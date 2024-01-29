import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean, scatter_max
import numpy as np
from e3nn.nn import BatchNorm
from torch_geometric.data import Batch
# from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims

import time
from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        # 把残基类型，lm_embedding，时间因子进行 embedding
        x = x.to(self.atom_embedding_list[0].weight.device)
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
        return x_embedding



class TensorProductAttention(torch.nn.Module):
    def __init__(self, in_irreps, in_tp_irreps, out_tp_irreps,
                 sh_irreps, out_irreps, n_edge_features,
                 batch_norm=False, dropout=0.0,
                 fc_dim=128, lin_self=False, # 残差项,可有可无
                 attention=False # 注意力机制项
                 ):
        super(TensorProductAttention, self).__init__()
        
        
        self.lin_in = o3.Linear(in_irreps, in_tp_irreps, internal_weights=True) # 在张量直积的前后都加入一个线性层
        self.tp = tp = o3.FullyConnectedTensorProduct(in_tp_irreps, sh_irreps, out_tp_irreps, shared_weights=False)      
        self.lin_out = o3.Linear(out_tp_irreps, out_irreps, internal_weights=True) # 在张量直积的前后都加入一个线性层
        if lin_self:
            self.lin_self = o3.Linear(in_irreps, out_irreps, internal_weights=True)
        else: self.lin_self = False
        if attention:
            self.attention = True
            key_irreps = [(mul//2, ir) for mul, ir in in_tp_irreps]
            self.h_q = o3.Linear(in_tp_irreps, key_irreps)
            self.tp_k = tp_k = o3.FullyConnectedTensorProduct(in_tp_irreps, sh_irreps, key_irreps, shared_weights=False)
            self.fc_k = self.fc = nn.Sequential(
                nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, fc_dim),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, tp_k.weight_numel)
            )
            self.dot = o3.FullyConnectedTensorProduct(key_irreps, key_irreps, "0e")
        else: self.attention = False
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, fc_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    
    def forward(self, node_attr, edge_index, edge_attr, edge_sh, ones=None, residual=True, out_nodes=None, reduce='mean'):
        node_attr_in = self.lin_in(node_attr)
        edge_src, edge_dst = edge_index
        out_nodes_bool = out_nodes!=None
        out_nodes = out_nodes or node_attr.shape[0]
        if self.attention:
            # 蛋白 配体
            def ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr):
                q = self.h_q(node_attr_in) # [res,24]
                k = self.tp_k(node_attr_in[edge_src], edge_sh, self.fc_k(edge_attr)) # [edge,24]
                v = self.tp(node_attr_in[edge_src], edge_sh, self.fc(edge_attr)) # [edge,78]
                a = self.dot(q[edge_dst], k) # 用蛋白去取配体的下标 (X) todo [edge,1]
                max_ = scatter_max(a, edge_dst, dim=0, dim_size=out_nodes)[0] # 
                a = (a - max_[edge_dst]).exp() # [edge,1]
                z = scatter(a, edge_dst, dim=0, dim_size=out_nodes)
                a = a / z[edge_dst]
                return scatter(a * v, edge_dst, dim=0, dim_size=out_nodes)
        else:
            # 蛋白 配体
            def ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr):
                tp = self.tp(node_attr_in[edge_src], edge_sh, self.fc(edge_attr))
                return scatter(tp, edge_dst, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.training:        
            out = torch.utils.checkpoint.checkpoint(ckpt_forward,
                    node_attr_in, edge_dst, edge_src, edge_sh, edge_attr)
        else:
            # 蛋白edge_dst, 配体edge_src
            out = ckpt_forward(node_attr_in, edge_dst, edge_src, edge_sh, edge_attr)
        
        out = self.lin_out(out)
        
        if not residual:
            return out
        if self.lin_self: 
            out = out + self.lin_self(node_attr)
        else:
            if out_nodes_bool:
                return out
            out = out + F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1])) 
        if self.batch_norm:
            out = self.batch_norm(out)
        return out



class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index.to(edge_attr.device)
        # 节点表征，边的向量的球谐表征，边的表征
        # [1352,48],[1352,9]> [1352,78]                  [1352,144]
        tp = self.tp(node_attr[edge_dst], edge_sh.to(edge_attr.device), self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out
        
        
class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(TensorProductScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = 0 # 噪声映射
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            rec_parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**rec_parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)

        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2*self.ns if num_conv_layers >= 3 else self.ns,ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs)
            )
        else:
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            
            self.res_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_res_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=3 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            if not no_torsion:
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
                
                
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

    def forward(self, data):
        # if False:
        # data['ligand'].pos = -data['ligand'].pos
        # data['receptor'].pos = -data['receptor'].pos
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr) 
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

    
        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)


        # 根据坐标建立残基-配体图,节点表示,节点idx,边的表示,边的球谐表示
        # 所有残基>>>所有配体
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        # 每层都在交互
        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing  节点的表示空间随着l不断增加
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh) # 配体内部进行张量直积(节点表示*边的球谐表示*权重)时,
            # 聚合（节点表示 直积 边向量的表示） >>>节点的表示(节点个数,out_irreps)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh) # 张量直积  残基的节点表示(节点个数,out_irreps)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        

        # compute confidence score
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)# 张量直积
        
        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb.to(tr_norm.device)], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb.to(rot_norm.device)], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1).to(tr_pred.device)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(rot_pred.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: return tr_pred, rot_pred, torch.empty(0, device=data['ligand'].x.device)
        
        
        if False:
            # 残基的平移旋转预测
            res_edge_index, res_edge_attr, res_edge_sh = self.build_res_cent_graph(data)
            res_edge_attr = self.res_edge_embedding(res_edge_attr)
            res_edge_attr = torch.cat([res_edge_attr, rec_node_attr[res_edge_index[0], :self.ns], rec_node_attr[res_edge_index[1], :self.ns]], -1)
            res_global_pred = self.final_res_conv(rec_node_attr, res_edge_index, res_edge_attr, res_edge_sh, out_nodes=data['receptor'].pos.shape[0])
            # 第一次预测就有很多0,300个残基的2*10+2*1e表示,有145个残基的表示都为0
            res_tr_pred = res_global_pred[:, :3] + res_global_pred[:, 6:9]
            res_rot_pred = res_global_pred[:, 3:6] + res_global_pred[:, 9:]
        else:
            res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 78:81]
            res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 81:84]
            
        data.res_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])[data['receptor'].batch]
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.cat([torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1), data.res_sigma_emb.to(res_tr_norm.device)], dim=1))
        
        mask_rot = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.cat([torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1), data.res_sigma_emb.to(res_rot_norm.device)], dim=1))

        res_tr_sigma = tr_sigma[data['receptor'].batch]
        res_rot_sigma = rot_sigma[data['receptor'].batch]
        if self.scale_by_sigma:
            res_tr_pred = res_tr_pred / res_tr_sigma.unsqueeze(1).to(res_tr_pred.device)
            res_rot_pred = res_rot_pred * so3.score_norm(res_rot_sigma.cpu()).unsqueeze(1).to(rot_pred.device)
        

        # torsional components
        # 中心向量:键中心到周围5A以内原子节点的向量
        # 扭转键向量:可扭转键两端原子的向量
        # tor_edge_index
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data) # 键中心到周围5A内原子节点的边向量球谐表示,高斯长度映射
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]] # 扭转键向量
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]] # 扭转键的表示,由两个节点的表示加和得到

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec.to(tor_bond_attr.device), normalize=True, normalization='component') # 扭转键球谐表示
        tor_edge_sh = self.final_tp_tor(tor_edge_sh.to(tor_bonds_sh.device), tor_bonds_sh[tor_edge_index.to(tor_bonds_sh.device)[0]]) # 中心向量表示和扭转键向量表示直积

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1) #中心向量长度映射,节点表示取end节点, 扭转键取start节点
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        # tor_pred_0o = self.tor_bond_conv_1x0o(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
        #                           out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        # tor_pred_0e = self.tor_bond_conv_1x0e(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
        #                           out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]


        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma .cpu().numpy())).float().to(tor_pred.device))
        # print(mask_rot.sum(),mask_rot.size(),mask_tr.sum(),mask_tr.size())
        return tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred

    def build_lig_conv_graph(self, data):
        
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr']) # 对每个节点做64维度的映射
        
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        
        # 配体原子特征+每个节点的噪声
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # da
        # builds the receptor initial node and edge embeddings
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        # esm特征
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]
        # [300,3,3] AB AC AD
        # data['receptor'].beta_vec = torch.cat( [ (data['receptor'].pos - data['receptor'].N_pos).unsqueeze(1), 
        #                                         (data['receptor'].pos - data['receptor'].beta_C_pos).unsqueeze(1), 
        #                                         (data['receptor'].pos - data['receptor'].O_pos).unsqueeze(1)] ,  1)
        
        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb.to(edge_length_emb.device), edge_length_emb], 1)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1)
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb.to(edge_attr.device)], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        # 所有扭转
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]] # 扭转角的起始节点，bsz=4,
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
    
    def build_res_cent_graph(self, data):
        edge_index = radius_graph(data['receptor'].pos, 5, data['receptor'].batch) # 建立虚拟边
        
        edge_vec = data['receptor'].pos[edge_index[1]] - data['receptor'].pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb.to(edge_attr.device)], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    # # 只建模周围4个原子,alpha-c, beta-C, N, O
    # def build_res_alphaC_graph(self, data):
    #     # data['receptor'].pos
    #     # data['receptor'].x  原本是残基的类型,残基的esm特征 >>> 4个原子的类型,原子的位置,残基类型,残基的esm2特征
    #     # [70,]
    #     return
def apply_euclidean(x, R):
    """
    R [..., 3, 3]
    x [..., Na, 3]
    """
    Rx = torch.einsum('...kl,...ml->...mk', R, x)
    return Rx 


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1).to(self.offset.device) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))



     
class FlexibleDockingModel(torch.nn.Module):
    def __init__(self, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2,num_encoder_layers=6,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(FlexibleDockingModel, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        lig_conv_encoders,pocket_conv_encoders = [],[]
        lig_encoder_parameters = {
                'in_irreps': f'{ns}x0e',
                'sh_irreps': self.sh_irreps,
                'out_irreps': f'{ns}x0e',
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
        pocket_encoder_parameters = {
                'in_irreps': f'{ns}x0e',
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': f'{ns}x0e',
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
        
        for i in range(num_encoder_layers):
            lig_conv_encoder = TensorProductConvLayer(**lig_encoder_parameters)
            pocket_conv_encoder = TensorProductConvLayer(**pocket_encoder_parameters)
            lig_conv_encoders.append(lig_conv_encoder)
            pocket_conv_encoders.append(pocket_conv_encoder)
        self.lig_conv_encoders = nn.ModuleList(lig_conv_encoders)
        self.pocket_conv_encoders = nn.ModuleList(pocket_conv_encoders)
         
        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            rec_parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**rec_parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)


        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        self.res_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_res_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))


    def forward(self, data):
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        
        # 配体的单塔编码
        for i in range(len(self.lig_conv_encoders)):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_node_attr = self.lig_conv_encoders[i](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh) 
        
        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        # 蛋白的单塔编码
        for i in range(len(self.pocket_conv_encoders)):
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
            rec_node_attr = self.pocket_conv_encoders[i](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh) 

        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing  节点的表示空间随着l不断增加
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh) 
            # 聚合（节点表示 直积 边向量的表示） >>>节点的表示(节点个数,out_irreps)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh) # 张量直积  残基的节点表示(节点个数,out_irreps)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update


        res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 128:131]
        res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 131:134]
            
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1))
        
        mask_rot = torch.logical_not(torch.all(res_rot_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1) ) # 方向和模长分开学


        tr_pred = lig_node_attr[:, 48:51] + lig_node_attr[:, 128:131]
        mask_tr = torch.logical_not(torch.all(tr_pred==0, dim=1))
        tr_pred_norm = tr_pred[mask_tr]/torch.norm(tr_pred[mask_tr], dim=1, keepdim=True)
        tr_norm = torch.zeros_like(tr_pred)
        tr_norm[mask_tr] = tr_pred_norm
        tr_pred = tr_norm * self.tr_final_layer(torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1))
        
        res_atoms_pos = self.predict_pocket_atoms_pos(data=data, res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
        lig_atoms_pos = self.predict_ligand_atoms_pos(data=data, tr_pred=tr_pred)
        
        return tr_pred, res_tr_pred, res_rot_pred, res_atoms_pos, lig_atoms_pos

    
    def predict_pocket_atoms_pos(self, data, res_tr_pred, res_rot_pred):
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(res_rot_pred) # 轴角到旋转矩阵 [res,3,3]  这个应该是一个限制  旋转向量的模长应该在0-2pi之间
        aboslute_rotation = False
        if aboslute_rotation:
            # 平移旋转
            atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + res_tr_pred.unsqueeze(1) 
        else:
            # 平移旋转
            atoms_pos = self.predict_pos_using_cent(data=data, tr_update=res_tr_pred, rot_update=res_rot_pred)
            
        # R@pos+T,现在是绕着原点旋转，再加入偏移量，这种方式的旋转对pos的影响很大.之后可以改为
        return atoms_pos
    
    def predict_ligand_atoms_pos(self, data, tr_pred):
        x = data['ligand'].pos
        assert x.shape==tr_pred.shape
        return x + tr_pred
    
    def predict_pos_using_cent(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_cent = torch.sum(atoms_pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着点云中心做旋转,再把点云中心加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    
    def build_lig_conv_graph(self, data):
              
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        node_attr = data['ligand'].x
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # esm特征
        node_attr = data['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = edge_length_emb.to(torch.float32)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1)
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh





 
class EncoderDecoder2(torch.nn.Module):
    def __init__(self, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2,num_encoder_layers=6,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(EncoderDecoder2, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]



        lig_encoder_irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        lig_conv_encoders,pocket_conv_encoders = [],[]
        for i in range(num_encoder_layers):
            lig_in_irreps = lig_encoder_irrep_seq[min(i, len(irrep_seq) - 1)]
            lig_out_irreps = lig_encoder_irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            lig_encoder_parameters = {
                'in_irreps': lig_in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': lig_out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            pocket_encoder_parameters = {
                'in_irreps': lig_in_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': lig_out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }        
            lig_conv_encoder = TensorProductConvLayer(**lig_encoder_parameters)
            pocket_conv_encoder = TensorProductConvLayer(**pocket_encoder_parameters)
            lig_conv_encoders.append(lig_conv_encoder)
            pocket_conv_encoders.append(pocket_conv_encoder)
        self.lig_conv_encoders = nn.ModuleList(lig_conv_encoders)
        self.pocket_conv_encoders = nn.ModuleList(pocket_conv_encoders)
         
        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            out_irreps = f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)

        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        self.res_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_res_conv = TensorProductConvLayer(
            in_irreps=f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps=f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))


    def forward(self, data):
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        
        # 配体的单塔编码
        for i in range(len(self.lig_conv_encoders)):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_node_attr_update = self.lig_conv_encoders[i](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)
            lig_node_attr = F.pad(lig_node_attr,(0, lig_node_attr_update.shape[-1]-lig_node_attr.shape[-1])) +  lig_node_attr_update # [atoms, 156]
            
        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        # 蛋白的单塔编码
        for i in range(len(self.pocket_conv_encoders)):
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
            rec_node_attr_update = self.pocket_conv_encoders[i](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)  # [atoms, 156]
            rec_node_attr = F.pad(rec_node_attr, (0, rec_node_attr_update.shape[-1] - rec_node_attr.shape[-1])) + rec_node_attr_update
            
            
        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        # 交互式编码
        for l in range(len(self.rec_to_lig_conv_layers)):
            
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            
            lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_inter_update
            rec_node_attr = rec_node_attr + rec_inter_update

        res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 128:131]
        res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 131:134]
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1))
        
        mask_rot = torch.logical_not(torch.all(res_rot_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1) ) # 方向和模长分开学


        tr_pred = lig_node_attr[:, 48:51] + lig_node_attr[:, 128:131]
        mask_tr = torch.logical_not(torch.all(tr_pred==0, dim=1))
        tr_pred_norm = tr_pred[mask_tr]/torch.norm(tr_pred[mask_tr], dim=1, keepdim=True)
        tr_norm = torch.zeros_like(tr_pred)
        tr_norm[mask_tr] = tr_pred_norm
        tr_pred = tr_norm * self.tr_final_layer(torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1))
        
        res_atoms_pos = self.predict_pocket_atoms_pos(data=data, res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
        lig_atoms_pos = self.predict_ligand_atoms_pos(data=data, tr_pred=tr_pred)
        
        return tr_pred, res_tr_pred, res_rot_pred, res_atoms_pos, lig_atoms_pos

    
    def predict_pocket_atoms_pos(self, data, res_tr_pred, res_rot_pred):
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(res_rot_pred) # 轴角到旋转矩阵 [res,3,3]  这个应该是一个限制  旋转向量的模长应该在0-2pi之间
        aboslute_rotation = False
        if aboslute_rotation:
            # 平移旋转
            atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + res_tr_pred.unsqueeze(1) 
        else:
            # 平移旋转
            atoms_pos = self.predict_pos_using_cent(data=data, tr_update=res_tr_pred, rot_update=res_rot_pred)
            
        # R@pos+T,现在是绕着原点旋转，再加入偏移量，这种方式的旋转对pos的影响很大.之后可以改为
        return atoms_pos
    
    def predict_ligand_atoms_pos(self, data, tr_pred):
        x = data['ligand'].pos
        assert x.shape==tr_pred.shape
        return x + tr_pred
    
    def predict_pos_using_cent(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_cent = torch.sum(atoms_pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着点云中心做旋转,再把点云中心加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    
    def build_lig_conv_graph(self, data):
              
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        node_attr = data['ligand'].x
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # esm特征
        node_attr = data['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = edge_length_emb.to(torch.float32)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1)
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh





class AttentionEncoderDecoder(torch.nn.Module):
    def __init__(self, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2, num_encoder_layers=6,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(AttentionEncoderDecoder, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]


        
        def fill_mults(ns, nv, irs, is_in=False):
            irreps = [(ns, (l, p)) if (l == 0 and p == 1) else [nv, (l, p)] for l, p in irs]
            return irreps
        
        # 双塔
        irrep_seq = [
                [(0, 1)],
                [(0, 1), (1, -1)],
                [(0, 1), (1, -1), (1, 1)],
                [(0, 1), (1, -1), (1, 1), (0, -1)]
            ]
        lig_conv_encoders,pocket_conv_encoders = [],[]
        for i in range(num_encoder_layers):
            
            in_seq, out_seq = irrep_seq[min(i, len(irrep_seq) - 1)], irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            lig_in_irreps = fill_mults(ns, nv, in_seq, is_in=(i==0))
            lig_out_irreps = fill_mults(ns, nv, out_seq)
            lig_in_tp_irreps = fill_mults(ns, nv, in_seq)
            lig_out_tp_irreps = fill_mults(ns, nv, out_seq)
            
            lig_encoder_parameters = {
                'in_irreps': lig_in_irreps,
                'in_tp_irreps': lig_in_tp_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': lig_out_irreps,
                'out_tp_irreps': lig_out_tp_irreps,
                'n_edge_features': 3 * ns,
                'lin_self': False,
                'batch_norm': batch_norm,
                'dropout': dropout,
                'attention':True
            }
            pocket_encoder_parameters = {
                'in_irreps': lig_in_irreps,
                'in_tp_irreps': lig_in_tp_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': lig_out_irreps,
                'out_tp_irreps': lig_out_tp_irreps,
                'n_edge_features': 3 * ns,
                'lin_self': False,
                'batch_norm': batch_norm,
                'dropout': dropout,
                'attention':True
            }        
            lig_conv_encoder = TensorProductAttention(**lig_encoder_parameters)
            pocket_conv_encoder = TensorProductAttention(**pocket_encoder_parameters)
            lig_conv_encoders.append(lig_conv_encoder)
            pocket_conv_encoders.append(pocket_conv_encoder)
        self.lig_conv_encoders = nn.ModuleList(lig_conv_encoders)
        self.pocket_conv_encoders = nn.ModuleList(pocket_conv_encoders)
         
        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            # in_irreps = f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            # out_irreps = f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            in_irreps = fill_mults(ns, nv, irrep_seq[-1])
            out_irreps = fill_mults(ns, nv, irrep_seq[-1])
            parameters = {
                'in_irreps': in_irreps,
                'in_tp_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'out_tp_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'lin_self': False,
                'batch_norm': batch_norm,
                'dropout': dropout,
                'attention':False
            }
            
            lig_to_rec_layer = TensorProductAttention(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductAttention(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)
        
        
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        
        
        # 平移和旋转的表示
        parity = True
        self.lig_final_tp = o3.FullyConnectedTensorProduct(out_irreps, out_irreps, '1x1o + 1x1e' if parity else '1x1o', internal_weights=True)
        self.pocket_tr_final_tp = o3.FullyConnectedTensorProduct(out_irreps, out_irreps, '1x1o + 1x1e' if parity else '1x1o', internal_weights=True)
        self.pocket_rot_final_tp = o3.FullyConnectedTensorProduct(out_irreps, out_irreps, '1x1o + 1x1e' if parity else '1x1o', internal_weights=True)

        # 模长的拟合
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        
    def forward(self, data):
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        
        # 配体的单塔编码(配体半径图)
        for i in range(len(self.lig_conv_encoders)):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_node_attr_update = self.lig_conv_encoders[i](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)
            lig_node_attr = F.pad(lig_node_attr,(0, lig_node_attr_update.shape[-1]-lig_node_attr.shape[-1])) +  lig_node_attr_update # [atoms, 156]
            
        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        # 蛋白的单塔编码(蛋白半径图)
        for i in range(len(self.pocket_conv_encoders)):
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
            rec_node_attr_update = self.pocket_conv_encoders[i](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)  # [atoms, 156]
            rec_node_attr = F.pad(rec_node_attr, (0, rec_node_attr_update.shape[-1] - rec_node_attr.shape[-1])) + rec_node_attr_update
            
        
        # 交互式编码 
        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        # 交互式编码(交互半径图)
        for l in range(len(self.rec_to_lig_conv_layers)):
            
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            
            lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_inter_update
            rec_node_attr = rec_node_attr + rec_inter_update

        
        pocket_rot = self.pocket_rot_final_tp(rec_node_attr,rec_node_attr)
        pocket_tr = self.pocket_tr_final_tp(rec_node_attr,rec_node_attr)
        res_rot_pred = pocket_rot[:, :3] + pocket_rot[:, 3:]
        res_tr_pred = pocket_tr[:, :3] + pocket_tr[:, 3:]
        # res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 131:134]
        # res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 128:131]
        
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1))
        
        mask_rot = torch.logical_not(torch.all(res_rot_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1) ) # 方向和模长分开学

        tr_pred = self.lig_final_tp(lig_node_attr, lig_node_attr)
        tr_pred = lig_node_attr[:, 0:3] + lig_node_attr[:, 3:6]
        # tr_pred = lig_node_attr[:, 48:51] + lig_node_attr[:, 128:131]
        mask_tr = torch.logical_not(torch.all(tr_pred==0, dim=1))
        tr_pred_norm = tr_pred[mask_tr]/torch.norm(tr_pred[mask_tr], dim=1, keepdim=True)
        tr_norm = torch.zeros_like(tr_pred)
        tr_norm[mask_tr] = tr_pred_norm
        tr_pred = tr_norm * self.tr_final_layer(torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1))
        
        res_atoms_pos = self.predict_pocket_atoms_pos(data=data, res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
        lig_atoms_pos = self.predict_ligand_atoms_pos(data=data, tr_pred=tr_pred)
        
        return tr_pred, res_tr_pred, res_rot_pred, res_atoms_pos, lig_atoms_pos

    
    def predict_pocket_atoms_pos(self, data, res_tr_pred, res_rot_pred):
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(res_rot_pred) # 轴角到旋转矩阵 [res,3,3]  这个应该是一个限制  旋转向量的模长应该在0-2pi之间
        aboslute_rotation = False
        if aboslute_rotation:
            # 平移旋转
            atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + res_tr_pred.unsqueeze(1) 
        else:
            # 平移旋转
            atoms_pos = self.predict_pos_using_cent(data=data, tr_update=res_tr_pred, rot_update=res_rot_pred)
            
        # R@pos+T,现在是绕着原点旋转，再加入偏移量，这种方式的旋转对pos的影响很大.之后可以改为
        return atoms_pos
    
    def predict_ligand_atoms_pos(self, data, tr_pred):
        x = data['ligand'].pos
        assert x.shape==tr_pred.shape
        return x + tr_pred
    
    def predict_pos_using_cent(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_cent = torch.sum(atoms_pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着点云中心做旋转,再把点云中心加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    
    def build_lig_conv_graph(self, data):
              
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        node_attr = data['ligand'].x
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # esm特征
        node_attr = data['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = edge_length_emb.to(torch.float32)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1)
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

from datasets.pdbbind_pocket import full_id_to_idx,ResidueSelector

def set_coord_structure(structure,full_id_list,pocket_res_list):
    for full_id, pred_res in zip (full_id_list,pocket_res_list):
        chain, res_id = full_id[2],full_id[3]
        res = structure[chain][res_id]
        for atom,pred_atom in zip(res,pred_res):
            if atom.name==pred_atom.name:
                new_coord = pred_atom.coord
                atom.set_coord( new_coord.reshape((3,)) )
            else:
                print('atom.name != pred_atom.name')
                raise ValueError
    return structure

def save_pocket_in(structure, path, pocket_res_list, set_coord=True):
    from Bio.PDB import PDBIO
    indices = []
    full_id_list = []
    for res in pocket_res_list:
        indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
        full_id_list.append(res.full_id)
    if set_coord:
        structure = set_coord_structure(structure,full_id_list,pocket_res_list)
    selector = ResidueSelector(indices)
    io = PDBIO()
    io.set_structure(structure)
    io.save(path,select=selector)
    return structure

import os

class FlexibleDocking_resTR_atomT(torch.nn.Module):
    def __init__(self, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2,inference=False,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(FlexibleDocking_resTR_atomT, self).__init__()
        self.inference = inference
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            rec_parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**rec_parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)


        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)

        
        self.res_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_res_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # 权值0初始化
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1)).apply(init_weights)
        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1)).apply(init_weights)
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1)).apply(init_weights)

                
    def forward(self, data):
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)


        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing  节点的表示空间随着l不断增加
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh) 
            # 聚合（节点表示 直积 边向量的表示） >>>节点的表示(节点个数,out_irreps)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh) # 张量直积  残基的节点表示(节点个数,out_irreps)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        
        # 残基的平移和旋转
        res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 78:81]
        res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 81:84] 
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1))
        mask_rot = torch.logical_not(torch.all(res_rot_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1) ) # 方向和模长分开
        res_atoms_pos = self.predict_pocket_atoms_pos(data=data, res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
        
        
        if self.inference:
            self.save_predict_pocket(data=data,res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
      
        # 配体每个原子的平移
        lig_atoms_tr_pred = lig_node_attr[:, 48:51] + lig_node_attr[:, 78:81]
        mask_tr = torch.logical_not(torch.all(lig_atoms_tr_pred==0, dim=1))
        lig_atoms_tr_pred_norm = lig_atoms_tr_pred[mask_tr]/torch.norm(lig_atoms_tr_pred[mask_tr], dim=1, keepdim=True)
        lig_atoms_tr_norm = torch.zeros_like(lig_atoms_tr_pred)
        lig_atoms_tr_norm[mask_tr] = lig_atoms_tr_pred_norm
        lig_atoms_tr_pred = lig_atoms_tr_norm * self.tr_final_layer(torch.linalg.vector_norm(lig_atoms_tr_pred, dim=1).unsqueeze(1))
        lig_atoms_pos = self.predict_ligand_atoms_pos(lig_atoms_pos=data['ligand'].pos, tr_pred=lig_atoms_tr_pred)
        
        return lig_atoms_tr_pred, res_tr_pred, res_rot_pred, res_atoms_pos, lig_atoms_pos

    def save_predict_pocket(self, data, res_tr_pred, res_rot_pred):
        bsz = data['receptor'].batch[-1].item()+1
        
        for i in range(bsz):
            d = data[i]
            mask_ith_pocket = data['receptor'].batch==i
            # 第i个口袋上所有残基的平移和旋转
            ith_pocket_res_tr_pred, ith_pocket_res_rot_pred = res_tr_pred[mask_ith_pocket], res_rot_pred[mask_ith_pocket]
            rot_mat = axis_angle_to_matrix(ith_pocket_res_rot_pred) #[n,3,3]
            import copy
            pocket_res_list =  d['receptor'].in_pocket_res
            pocket_cent = d.original_center.cpu().numpy().reshape(3,)
             
            for res_tr,res_rot,res in zip(ith_pocket_res_tr_pred, rot_mat, pocket_res_list):
                # 每个残基的平移和旋转
                old_coord = torch.from_numpy( np.array([atom.coord-pocket_cent for atom in res])) #[N_atoms,3]
                # 残基点云的中心
                pos_cent = torch.mean(old_coord, dim=0, keepdim=True) #[1,3]
                old_coord_norm = old_coord - pos_cent
                roted_coord = apply_euclidean(old_coord_norm.to(res_rot), res_rot) + pos_cent.to(res_rot)
                new_coord = roted_coord + res_tr
                # 加入口袋中心
                for atom, atom_coord in zip(res,new_coord):
                    atom.set_coord( atom_coord.cpu().numpy().reshape(3,)+pocket_cent )
            
            induce_pocket_path = f'./out_file/esmFold_{d.datatype}/induced_pocket'
            os.makedirs(induce_pocket_path, exist_ok=True)
            
            structure = save_pocket_in(structure=d['receptor'].in_structure, 
                                path=induce_pocket_path + f'/{d.name}_pocket_10A.pdb', 
                                pocket_res_list=pocket_res_list)
            save_pocket_in(structure=structure, 
                                path=induce_pocket_path + f'/{d.name}_pocket_6A.pdb', 
                                pocket_res_list=d['receptor'].in_pocket_res_6A, set_coord=False)
    

                
        
        
    def predict_pocket_atoms_pos(self, data, res_tr_pred, res_rot_pred):
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(res_rot_pred) # 轴角到旋转矩阵 [res,3,3]  这个应该是一个限制  旋转向量的模长应该在0-2pi之间
        aboslute_rotation = False
        if aboslute_rotation:
            # 平移旋转
            atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + res_tr_pred.unsqueeze(1) 
        else:
            # 平移旋转
            atoms_pos = self.predict_pos_using_cent(data=data, tr_update=res_tr_pred, rot_update=res_rot_pred)
            
        # R@pos+T,现在是绕着原点旋转，再加入偏移量，这种方式的旋转对pos的影响很大.之后可以改为
        return atoms_pos
    
    def predict_ligand_atoms_pos(self, lig_atoms_pos, tr_pred):
        x = lig_atoms_pos
        assert x.shape==tr_pred.shape
        return x + tr_pred
    
    def predict_pos_using_cent(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_cent = torch.sum(atoms_pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着点云中心做旋转,再把点云中心加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    
    def build_lig_conv_graph(self, data):
              
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        node_attr = data['ligand'].x
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # esm特征
        node_attr = data['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()] # 边向量的球谐表示

        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = edge_length_emb.to(torch.float32)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1) # 边向量的球谐表示
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh





























class FlexibleDocking_resTR_ligTRatomT(torch.nn.Module):
    def __init__(self, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2, pretrain_method ='pretrain_method1',
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(FlexibleDocking_resTR_ligTRatomT, self).__init__()
        self.pretrain_method = pretrain_method
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            rec_parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**rec_parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)


        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        self.res_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_res_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.lig_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.lig_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.lig_atoms_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))


    def forward(self, data):
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)


        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing  节点的表示空间随着l不断增加
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh) 
            # 聚合（节点表示 直积 边向量的表示） >>>节点的表示(节点个数,out_irreps)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh) # 张量直积  残基的节点表示(节点个数,out_irreps)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        
        # 残基的平移和旋转
        res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 78:81]
        res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 81:84] 
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1))
        mask_rot = torch.logical_not(torch.all(res_rot_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1) ) # 方向和模长分开
        res_atoms_pos = self.predict_pocket_atoms_pos(data=data, res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
        
        
        # 配体整体的平移,旋转
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)# 张量直积
        
        lig_tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        lig_rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        lig_tr_norm = torch.linalg.vector_norm(lig_tr_pred, dim=1).unsqueeze(1)
        lig_tr_pred = lig_tr_pred / lig_tr_norm * self.lig_tr_final_layer(lig_tr_norm)
        lig_rot_norm = torch.linalg.vector_norm(lig_rot_pred, dim=1).unsqueeze(1)
        lig_rot_pred = lig_rot_pred / lig_rot_norm * self.lig_rot_final_layer(lig_rot_norm)
        lig_atoms_pos = self.predict_ligand_RT(data=data, tr_pred=lig_tr_pred, rot_pred=lig_rot_pred)
        
        lig_atoms_tr_pred = None
        if self.pretrain_method=='pretrain_method2':
            # 配体每个原子的平移
            lig_atoms_tr_pred = lig_node_attr[:, 48:51] + lig_node_attr[:, 78:81]
            mask_tr = torch.logical_not(torch.all(lig_atoms_tr_pred==0, dim=1))
            lig_atoms_tr_pred_norm = lig_atoms_tr_pred[mask_tr]/torch.norm(lig_atoms_tr_pred[mask_tr], dim=1, keepdim=True)
            lig_atoms_tr_norm = torch.zeros_like(lig_atoms_tr_pred)
            lig_atoms_tr_norm[mask_tr] = lig_atoms_tr_pred_norm
            lig_atoms_tr_pred = lig_atoms_tr_norm * self.lig_atoms_tr_final_layer(torch.linalg.vector_norm(lig_atoms_tr_pred, dim=1).unsqueeze(1))
            lig_atoms_pos = self.predict_ligand_atoms_pos(lig_atoms_pos=lig_atoms_pos, tr_pred=lig_atoms_tr_pred)
        
        return lig_atoms_tr_pred, res_tr_pred, res_rot_pred, res_atoms_pos, lig_atoms_pos

    def predict_ligand_RT(self, data, tr_pred, rot_pred):
        # 绕着配体点云中心做旋转
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)
        pos_cent = center_pos[data['ligand'].batch] #[bsz*lig, 3] 计算配体的点云中心
        
        rot_mat = axis_angle_to_matrix(rot_pred)[data['ligand'].batch] # [bsz*lig, 3, 3]# 计算旋转矩阵
        atoms_pos = data['ligand'].pos - pos_cent # 将配体去中心化 [bsz*lig, 3]
        atoms_pos = apply_euclidean( atoms_pos.unsqueeze(1) , rot_mat.float() ).squeeze(1) #[bsz*lig, 3]
        atoms_pos = atoms_pos + pos_cent
        # 对配体做平移
        atoms_pos = atoms_pos + tr_pred[data['ligand'].batch]
        return atoms_pos
    
    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    def predict_pocket_atoms_pos(self, data, res_tr_pred, res_rot_pred):
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(res_rot_pred) # 轴角到旋转矩阵 [res,3,3]  这个应该是一个限制  旋转向量的模长应该在0-2pi之间
        aboslute_rotation = False
        if aboslute_rotation:
            # 平移旋转
            atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + res_tr_pred.unsqueeze(1) 
        else:
            # 平移旋转
            atoms_pos = self.predict_pos_using_cent(data=data, tr_update=res_tr_pred, rot_update=res_rot_pred)
            
        # R@pos+T,现在是绕着原点旋转，再加入偏移量，这种方式的旋转对pos的影响很大.之后可以改为
        return atoms_pos
    
    def predict_ligand_atoms_pos(self, lig_atoms_pos, tr_pred):
        x = lig_atoms_pos
        assert x.shape==tr_pred.shape
        return x + tr_pred
    
    def predict_pos_using_cent(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_cent = torch.sum(atoms_pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着点云中心做旋转,再把点云中心加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    
    def build_lig_conv_graph(self, data):
              
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        node_attr = data['ligand'].x
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # esm特征
        node_attr = data['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()] # 边向量的球谐表示

        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = edge_length_emb.to(torch.float32)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1) # 边向量的球谐表示
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh
    
class FlexibleTorsionModel(torch.nn.Module):
    def __init__(self, in_lig_edge_features=4, sigma_embed_dim=0, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1):
        super(FlexibleTorsionModel, self).__init__()
        
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            rec_parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': '+'.join(['0e+1o+2e' for _ in range(5)]),
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**rec_parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)


        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        
        self.res_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_res_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # 配体和残基 平移,旋转 最后一层
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.res_rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        
        # 配体的扭转角预测
        self.final_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))
        self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                    )
        self.tor_final_layer = nn.Sequential(nn.Linear(2 * ns, ns, bias=False),nn.Tanh(),nn.Dropout(dropout),nn.Linear(ns, 1, bias=False))

    def forward(self, data):
        
        if type(data)==list:
            data = Batch.from_data_list(data)

        # build ligand graph 
        # 根据坐标建立配体图,节点表示,节点idx,边的表示,边的球谐表示
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # 根据坐标建立口袋图,节点表示,节点idx,边的表示,边的球谐表示
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)


        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing  节点的表示空间随着l不断增加
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh) 
            # 聚合（节点表示 直积 边向量的表示） >>>节点的表示(节点个数,out_irreps)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0]) # 张量直积   残基向配体节点聚合   配体节点的表示(节点个数,out_irreps)

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh) # 张量直积  残基的节点表示(节点个数,out_irreps)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0]) # 张量直积   配体向残基聚合 残基的节点表示(节点个数,out_irreps)

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update


        res_tr_pred = rec_node_attr[:, 48:51] + rec_node_attr[:, 78:81]
        res_rot_pred = rec_node_attr[:, 51:54] + rec_node_attr[:, 81:84]
            
        #res_mask = res_tr_norm>0.00001
        mask_tr = torch.logical_not(torch.all(res_tr_pred==0, dim=1))
        res_tr_pred_norm = res_tr_pred[mask_tr]/torch.norm(res_tr_pred[mask_tr], dim=1, keepdim=True)
        res_tr_norm = torch.zeros_like(res_tr_pred)
        res_tr_norm[mask_tr] = res_tr_pred_norm
        res_tr_pred = res_tr_norm * self.res_tr_final_layer(torch.linalg.vector_norm(res_tr_pred, dim=1).unsqueeze(1))
        
        mask_rot = torch.logical_not(torch.all(res_rot_pred==0, dim=1))
        res_rot_pred_norm = res_rot_pred[mask_rot]/torch.norm(res_rot_pred[mask_rot], dim=1, keepdim=True)
        res_rot_norm = torch.zeros_like(res_rot_pred)
        res_rot_norm[mask_rot] = res_rot_pred_norm
        res_rot_pred = res_rot_norm * self.res_rot_final_layer(torch.linalg.vector_norm(res_rot_pred, dim=1).unsqueeze(1) ) # 方向和模长分开学


        # 配体部分的平移,旋转,扭转
        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)# 张量直积
        
        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        
        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(tr_norm)
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(rot_norm)
        
        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data) # 键中心到周围5A内原子节点的边向量球谐表示,高斯长度映射
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]] # 扭转键向量
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]] # 扭转键的表示,由两个节点的表示加和得到
        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec.to(tor_bond_attr.device), normalize=True, normalization='component') # 扭转键球谐表示
        tor_edge_sh = self.final_tp_tor(tor_edge_sh.to(tor_bonds_sh.device), tor_bonds_sh[tor_edge_index.to(tor_bonds_sh.device)[0]]) # 中心向量表示和扭转键向量表示直积
        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns], tor_bond_attr[tor_edge_index[0], :self.ns]], -1) #中心向量长度映射,节点表示取end节点, 扭转键取start节点
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh, out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        
        
        res_atoms_pos = self.predict_pocket_atoms_pos(data=data, res_tr_pred=res_tr_pred, res_rot_pred=res_rot_pred)
        lig_atoms_pos = self.predict_ligand_atoms_pos(data=data, tr_pred=tr_pred, rot_pred=rot_pred, tor_pred=tor_pred)
        
        return tr_pred, res_tr_pred, res_rot_pred, res_atoms_pos, lig_atoms_pos

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        # 所有扭转
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long() # 所有的扭转键组成的边
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2 # 键的中心坐标
        bond_batch = data['ligand'].batch[bonds[0]] # 扭转边的起始节点，bsz=4, 0 1 2 3,所有以0,1,2,3组成的一个tensor 【0，0，0，0，0，1，1，1，1，2，2，2，2】扭转键的个数
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    
    def predict_pocket_atoms_pos(self, data, res_tr_pred, res_rot_pred):
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(res_rot_pred) # 轴角到旋转矩阵 [res,3,3]  这个应该是一个限制  旋转向量的模长应该在0-2pi之间
        aboslute_rotation = False
        if aboslute_rotation:
            # 平移旋转
            atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + res_tr_pred.unsqueeze(1) 
        else:
            # 平移旋转
            atoms_pos = self.predict_pos_using_cent(data=data, tr_update=res_tr_pred, rot_update=res_rot_pred)
            
        # R@pos+T,现在是绕着原点旋转，再加入偏移量，这种方式的旋转对pos的影响很大.之后可以改为
        return atoms_pos
    
    def predict_ligand_atoms_pos(self, data, tr_pred, rot_pred, tor_pred):
        # 绕着配体点云中心做旋转
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)
        pos_cent = center_pos[data['ligand'].batch] #[bsz*lig, 3] 计算配体的点云中心
        
        rot_mat = axis_angle_to_matrix(rot_pred)[data['ligand'].batch] # [bsz*lig, 3, 3]# 计算旋转矩阵
        atoms_pos = data['ligand'].pos - pos_cent # 将配体去中心化 [bsz*lig, 3]
        atoms_pos = apply_euclidean( atoms_pos.unsqueeze(1) , rot_mat.float() ).squeeze(1) #[bsz*lig, 3]
        atoms_pos = atoms_pos + pos_cent
        
        # 对配体做平移
        atoms_pos = atoms_pos + tr_pred[data['ligand'].batch]
        
        # 对配体做扭转角的更新
        data['ligand'].pos = atoms_pos
        atoms_pos = self.modify_torsion(Batchdata=data, torsion_updates=tor_pred)
        data['ligand'].pos = atoms_pos
        
        return atoms_pos
    
    def modify_torsion(self, Batchdata, torsion_updates):
        
        data_list = Batchdata.to_data_list()
        atoms_pos_list = []
        torsion_start = 0
        for idx, data in enumerate(data_list):
            torsion_end = torsion_start + data['ligand'].edge_mask.sum()
            if data['ligand'].edge_mask.sum()>0:
                # 判断扭转角是否更新
                atoms_pos = data['ligand'].pos
                flexible_new_pos = self.modify_conformer_torsion_angles(atoms_pos.clone(),
                                                                data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                                data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                                torsion_updates[torsion_start:torsion_end]).to(atoms_pos.device)
                R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, atoms_pos.T)
                aligned_flexible_pos = flexible_new_pos @ R.T + t.T
                atoms_pos_list.append(aligned_flexible_pos)
            else:
                atoms_pos_list.append(data['ligand'].pos)
            
            torsion_start = torsion_end
        atoms_pos = torch.cat(atoms_pos_list, 0)
        return atoms_pos
    
    def modify_conformer_torsion_angles(self, pos, edge_index, mask_rotate, torsion_updates):
        for idx_edge, e in enumerate(edge_index):
            # if torsion_updates[idx_edge] == 0:
            #     continue
            u, v = e[0], e[1]

            # check if need to reverse the edge, v should be connected to the part that gets rotated
            # assert not mask_rotate[idx_edge, u]
            # assert mask_rotate[idx_edge, v]

            rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
            rot_vec = rot_vec / torch.norm(rot_vec)  # idx_edge!
            rot_mat = gen_matrix_from_rot_vec(rot_vec, torsion_updates[idx_edge])

            pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
        return pos
    
    def predict_pos_using_cent(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_cent = torch.sum(atoms_pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着点云中心做旋转,再把点云中心加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_cent.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    
    def build_lig_conv_graph(self, data):
              
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,# 边的类型
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device) # 将半径边condcat进来
        ], 0)
        
        node_attr = data['ligand'].x
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # esm特征
        node_attr = data['receptor'].x
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]
#.side_chain_vecs 
        beta_edge_vec = data['receptor'].side_chain_vecs[dst.long()] - data['receptor'].side_chain_vecs[src.long()]# [edge_num, 4, 3]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = edge_length_emb.to(torch.float32)
        beta_edge_sh = o3.spherical_harmonics('0e+1o+2e', beta_edge_vec, normalize=True, normalization='component').view([-1, 4*9])
        edge_sh = o3.spherical_harmonics('0e+1o+2e', edge_vec, normalize=True, normalization='component')
        edge_sh = torch.cat([edge_sh, beta_edge_sh], 1)
        # alphaC, betaC, N, O原子 >>> ABCD四个原子
        # AB,AC,AD向量

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

def gen_matrix_from_rot_vec(k, theta):
    K = torch.zeros((3, 3), device=k.device, dtype=k.dtype)
    K[[1, 2, 0], [2, 0, 1]] = -k
    K[[2, 0, 1], [1, 2, 0]] = k
    R = torch.eye(3, device=k.device) + K * torch.sin(theta) + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R



