import copy

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
# from torch_geometric.nn.data_parallel import DataParallel
from confidence.dataset import ListDataset
from utils import so3, torus
from utils.sampling import randomize_position, sampling,sampling_batch, randomize_res_position, sampling_flexible_batch
import torch
from utils.diffusion_utils import get_t_schedule
import pickle
from utils.utils import get_symmetry_rmsd
from rdkit import Chem
import json
import os
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
import networkx as nx
import random
from Bio.PDB import PDBParser, Superimposer
import torch.nn.functional as F

def loss_function(tr_pred, rot_pred, tor_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, apply_mean=True, no_torsion=False):
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred - tr_score.to(tr_pred.device)) ** 2 * tr_sigma.to(tr_pred.device) ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1).to(device)
    rot_loss = (((rot_pred - rot_score.to(rot_pred.device)) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.to(rot_score_norm.device) / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy()), device=device).float()
        tor_loss = ((tor_pred - tor_score.to(tor_pred.device)) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score.to(tor_score_norm2.device) ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float, device=device), tor_base_loss.mean() * torch.ones(1, dtype=torch.float, device=device)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.to(tor_loss.device).index_add_(0, index.to(tor_loss.device), tor_loss)
            t_b_l.to(tor_base_loss.device).index_add_(0, index.to(tor_base_loss.device), tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss.to(rot_loss.device) * tor_weight 
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), tr_base_loss, rot_base_loss, tor_base_loss



def loss_function_res(tr_pred, res_tr_pred, res_rot_pred, data, device, tr_weight=1, res_tr_weight=1, res_rot_weight=1, apply_mean=True):

    mean_dims = (0, 1) if apply_mean else 1
    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    res_tr_score = torch.cat([d.res_tr_score for d in data], dim=0) # [bsz*res, 3] 
    res_rot_score = torch.cat([d.res_rot_score for d in data], dim=0)
    
    # 配体原子坐标
    tr_loss = F.smooth_l1_loss(tr_pred, tr_score.to(tr_pred.device))

    # 残基平移
    res_tr_loss =F.smooth_l1_loss(res_tr_pred , res_tr_score.to(tr_pred.device)) 
    
    # 残基旋转的loss
    res_rot_loss =F.smooth_l1_loss(res_rot_pred , res_rot_score.to(res_rot_pred.device)) 
    loss = res_rot_loss*res_rot_weight + res_tr_loss*res_tr_weight + tr_loss * tr_weight
    return loss, tr_loss.detach(),res_tr_loss.detach(), res_rot_loss.detach()

def loss_function_rmsd(pocket_atoms_pos, lig_atoms_pos, data,  pocket_weight=1, ligand_weight=1):

    
    tr_score = torch.cat([d['ligand'].ref_pos for d in data], dim=0) # [bsz*atoms, 3] 
    res_tr_score = torch.cat([d['receptor'].ref_res_atoms_pos for d in data], dim=0) # [bsz*res, 14, 3] 
    mask = torch.cat([d['receptor'].res_atoms_mask for d in data], dim=0) # [bsz*res, 14] 
    
    # 配体原子坐标rmsd loss
    lig_loss = rmsd_loss(lig_atoms_pos, tr_score.to(lig_atoms_pos.device))

    # 残基原子坐标rmsd loss
    pocket_loss = rmsd_loss(pocket_atoms_pos, res_tr_score.to(pocket_atoms_pos.device), mask=mask) 
    
    # 肽键不合理的loss
    # c_idx_list = [d['receptor'].c_idx for d in data]# idx [[],[],[]] mask
    # n_mask_batch = torch.cat([d['receptor'].n_mask for d in data], dim=0)# [bsz*res, 14] N原子
    
    # peptide_bond_loss = pocket_atoms_pos[n_mask_batch], #
    
    loss = pocket_loss*pocket_weight + lig_loss*ligand_weight
    return loss, pocket_loss.detach(), lig_loss.detach()

def rmsd_loss(x, y, mask=None):
    # x y [res, atoms, 3]
    if mask is not None:
        x_ = x[mask]
        y_ = y[mask]
    else:
        x_ = x
        y_ = y
    # x y [atoms, 3]
    
    return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        # self.count = 1 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.cpu().sum() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out
import time  

def train_step(model, data, optimizer, device, loss_fn, ema_weigths, scheduler):
    t1 = time.time()
    if device.type == 'cuda' and len(data) == 1:
        print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        return None
    
    optimizer.zero_grad()
    try:
        tr_pred, res_tr_pred, res_rot_pred, pocket_atoms_pos, ligand_atoms_pos = model(data)
        t2 = time.time()
        print('model using ',t2-t1,'s')
        loss, pocket_loss, lig_loss = loss_fn(pocket_atoms_pos, ligand_atoms_pos, data=data)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_weigths.update(model.parameters())
        return [loss.cpu().detach(), pocket_loss, lig_loss]
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('| WARNING: ran out of memory, skipping batch')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            return None
        elif 'Input mismatch' in str(e):
            print('| WARNING: weird torch_cluster error, skipping batch')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            return None
        else:
            print('data error forward', e)
            #torch.cuda.empty_cache()
            return None

def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths):# to do
    model.train()
    meter = AverageMeter(['loss','pocket_loss', 'lig_loss'])
    t1 = time.time()
    for data in tqdm(loader, total=len(loader)):
        #continue
        # 测试GPU
        # print('test gpu use')
        # while True:
        t2 = time.time()
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            tr_pred, res_tr_pred, res_rot_pred, pocket_atoms_pos, ligand_atoms_pos = model(data)
            t3 = time.time()
            print('model using ',t3-t2,' s')
            loss, pocket_loss, lig_loss = loss_fn(pocket_atoms_pos, ligand_atoms_pos, data=data)
            
            loss.backward()
            optimizer.step()
            ema_weigths.update(model.parameters())
            meter.add([loss.cpu().detach(), pocket_loss, lig_loss])
            print(f'total_loss {loss.cpu().detach().item():.3f} pocket_loss {pocket_loss:.3f} ligand_loss {lig_loss.item():.3f}')
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print('data error forward', e)
                torch.cuda.empty_cache()
                continue

    return meter.summary()

def train_epoch_uni(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths):# to do
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'])
    train_loss_list, tr_loss_list, rot_loss_list, tor_loss_list = [],[],[],[]
    pocket_nan_list = []
    for i, data in tqdm(enumerate(loader), total=len(loader)):
        # i+=1
        # if (i+1)%8000==0:
        #     break
        # if i != 1358:
        #     continue
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            tr_pred, rot_pred, tor_pred = model(data)
            loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, data=data['net_input']["pocket_lig_graph"], t_to_sigma=t_to_sigma, device=device)
            loss.backward()
            if (not torch.isnan(loss).any()):
                train_loss_list.append(loss.item())
                tr_loss_list.append(tr_loss.item())
                rot_loss_list.append(rot_loss.item())
                tor_loss_list.append(tor_loss.item())
                if (i+1) % 10 == 0:
                    print('train_loss: ', np.average(train_loss_list), 'tr_loss: ', np.average(tr_loss_list), 'rot_loss: ', np.average(rot_loss_list), 'tor_loss: ', np.average(tor_loss_list))
            #     break
            # if torch.isnan(loss).any():
            #     for g in data['net_input']['pocket_lig_graph'] :
            #         pocket_nan_list.append(g.name)
            #         print(pocket_nan_list)
                # with open('/home/hxq2/DiffDock/name_nan_loss','wb') as f:
                #     pickle.dump((pocket_nan_list), f)
                # pocket_nan_list.append(data[])
                # print('--------------------\n',nan_list)
                
            optimizer.step()
            ema_weigths.update(model.parameters())
            if not torch.isnan(loss).any():
                meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred = model(data)

            loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([d.complex_t[noise_type] for d in data]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add(
                    [loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_rot,
                     sigma_index_tor, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


    model.eval()
def test_epoch_uni(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
            unpooled_metrics=True, intervals=10)
    # i = 0
    for i,data in tqdm(enumerate(loader), total=len(loader)):
        # if i > 20:
        #     break
        
        try:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred = model(data)

            loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, data=data['net_input']["pocket_lig_graph"], t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([d.complex_t[noise_type] for d in data['net_input']["pocket_lig_graph"]]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add(
                    [loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_rot,
                     sigma_index_tor, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch(model, complex_graphs, device, t_to_sigma, args):
    
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds = []

    for orig_complex_graph in tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph) for i in range(10)]
        # xqup 没有扭转
        randomize_position(data_list, no_torsion=False, no_random=False, tr_sigma_max=args.tr_sigma_max)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, # 一个构象预测32个
                                                        model=model.module if device.type=='cuda' else model,
                                                        #  model= model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        if failed_convergence_counter > 5: continue
        if args.no_torsion: # 不运行
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                     orig_complex_graph.original_center.cpu().numpy()) 

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy() # 过滤H

        if isinstance(orig_complex_graph['ligand'].orig_pos, list): # 判断初始ground_truth配体构象是否只给了一个
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # 预测的配体坐标
        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list]) # 把所有预测的坐标
        # 初始配体坐标 减去蛋白中心
        orig_ligand_pos = np.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append( np.min( rmsd))

    rmsds = np.array(rmsds)
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))} # 如果只预测了一个就是对的,不然计算的rmsds偏高
    return losses


def inference_epoch_save_lig(model, complex_graphs_dataset, device, t_to_sigma, args, rot_bool=True, tr_bool=True, tor_bool=True, out_dir='out_pocket_tr_rot'):
    
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule
    loader = DataLoader(dataset=complex_graphs_dataset, batch_size=1, shuffle=False)
    rmsds = []
    rmsds_all_sample_average = []
    num = 100
    i = 0
    for orig_complex_graph in tqdm(loader):
        # if i>=1000:
        #     break
        i += 1 

        # # 使用ground truth 做推理(正常情况pos是rdkit生成的)
        # orig_complex_graph['ligand'].pos = (torch.tensor( orig_complex_graph['ligand'].orig_pos[0] ) - orig_complex_graph.original_center).to(orig_complex_graph['ligand'].pos.dtype)
        
        data_list = [copy.deepcopy(orig_complex_graph) for i in range(10)]

        import os
        lig = orig_complex_graph.mol[0]


        if out_dir is not None:
        # 保存输入
            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
            write_dir = f'./{out_dir}/{data_list[0]["name"][0].replace("/","-")}'
            os.makedirs(write_dir, exist_ok=True)
            from datasets.process_mols import write_mol_with_coords
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'input_rank{rank+1}.sdf'))


        # randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)
        # 固定扭转角和欧拉角
        randomize_position(data_list, no_torsion= not tor_bool, no_random= not tr_bool, tr_sigma_max=args.tr_sigma_max, no_rot=not rot_bool)
        

        if out_dir is not None:
            # 保存采样前的ligs(加完平移/旋转噪声)
            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
            write_dir = f'./{out_dir}/{data_list[0]["name"][0].replace("/","-")}'
            os.makedirs(write_dir, exist_ok=True)
            from datasets.process_mols import write_mol_with_coords
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'random_tr_rank{rank+1}.sdf'))
        
        
        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, # 猜测一个构象预测32个
                                                        model=model.module if hasattr(model, 'module') else model,
                                                        # model= model,
                                                        inference_steps=args.inference_steps,
                                                        tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                        tor_schedule=tor_schedule,
                                                        device=device, t_to_sigma=t_to_sigma, model_args=args, 
                                                        rot_bool=rot_bool, tr_bool=tr_bool, tor_bool=tor_bool)
                    

            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        
        import os

        if out_dir is not None:
            # 保存采样后的ligs
            data_list = predictions_list
            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
            write_dir = f'./{out_dir}/{data_list[0]["name"][0].replace("/","-")}'
            os.makedirs(write_dir, exist_ok=True)
            from datasets.process_mols import write_mol_with_coords
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'sample_rank{rank+1}.sdf'))


        if failed_convergence_counter > 5: continue
        if args.no_torsion: # 不运行
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                     orig_complex_graph.original_center.cpu().numpy()) 

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy() # 过滤H

        if isinstance(orig_complex_graph['ligand'].orig_pos, list): # 判断初始ground_truth配体构象是否只给了一个
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # 预测的配体坐标
        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list]) # 把所有预测的坐标
        # 初始配体坐标 减去蛋白中心
        orig_ligand_pos = np.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

        min_rmsd_idx = np.argmin(rmsd)
        if out_dir is not None:
            np.save(os.path.join(write_dir,f'min_idx{min_rmsd_idx}'), rmsd)
        rmsds.append(np.min( rmsd ))
        rmsds_all_sample_average.append(np.average(rmsd))

    rmsds = np.array(rmsds)
    rmsds_all_sample_average = np.array(rmsds_all_sample_average)
    print('average min rmsd',np.average(rmsds))
    print('average all rmsd',np.average(rmsds_all_sample_average))
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))} # 如果只预测了一个就是对的,不然计算的rmsds偏高
    return losses


import copy
import numpy as np
import torch
import math
from rdkit import Chem
import networkx as nx


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t


def modify_conformer(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # rot_update = values[3:6] % (np.pi * 2)
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # torsion_updates = values[6:] % (np.pi * 2)
    # rot_update = values[3:6]
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (coords - lig_center) @ rot_mat.T + tr_update + lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos.clone(),
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            return aligned_flexible_pos
        except:
            return flexible_new_pos
    else:
        return rigid_new_pos


def modify_conformer2(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # rot_update = values[3:6]
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rigid_new_pos = coords - lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            holo_pos = flexible_new_pos @ R.T + t.T
        except:
            holo_pos = flexible_new_pos
    else:
        holo_pos = rigid_new_pos
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    out_pos = holo_pos @ rot_mat.T + tr_update + lig_center
    return out_pos


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates):
    # pos = copy.deepcopy(pos)

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


def gen_matrix_from_rot_vec(k, theta):
    K = torch.zeros((3, 3), device=k.device)
    K[[1, 2, 0], [2, 0, 1]] = -k
    K[[2, 0, 1], [1, 2, 0]] = k
    R = torch.eye(3, device=k.device) + K * torch.sin(theta) + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R




def single_conf_gen(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=1)
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    except:
        pass
    
    mol = Chem.RemoveHs(mol)
    return mol


def single_conf_gen_no_MMFF(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=40
    )
    mol = Chem.RemoveHs(mol)
    return mol


def get_mask_rotate(mol, device='cpu'):
    mol = Chem.RemoveHs(mol)
    G = nx.Graph()
    nodes = range(len(mol.GetAtoms()))
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    assert nx.is_connected(G), "分子图必须为连通图"
    torsions = []
    masks = []
    torsion_smarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsion_smarts)
    matches = mol.GetSubstructMatches(torsion_query)
    for edge in matches:
        G2 = G.to_undirected()
        G2.remove_edge(*edge)
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                torsions.append(edge)
                mask = torch.zeros(len(nodes), dtype=torch.bool, device=device)
                mask[l] = True
                masks.append(mask)
    return torsions, masks




def modify_conformer(coords, values, edge_index, mask_rotate):
    tr_update = (torch.sigmoid(values[:3]) - 0.5) * 10
    rot_update = (torch.sigmoid(values[3:6]) - 0.5) * np.pi * 2
    # rot_update = values[3:6] % (np.pi * 2)
    # torsion_updates = (torch.sigmoid(values[6:]) - 0.5) * np.pi * 2
    # torsion_updates = values[6:] % (np.pi * 2)
    # rot_update = values[3:6]
    torsion_updates = values[6:]

    lig_center = torch.mean(coords, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (coords - lig_center) @ rot_mat.T + tr_update + lig_center

    if values.shape[0] > 6:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos.clone(),
                                                           edge_index,
                                                           mask_rotate,
                                                           torsion_updates).to(rigid_new_pos.device)
        try:
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            return aligned_flexible_pos
        except:
            return flexible_new_pos
    else:
        return rigid_new_pos


def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=40
    )
    mol = Chem.RemoveHs(mol)
    
    torsions, masks = get_mask_rotate(mol)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if len(torsions) > 0:
        with torch.no_grad():
            coords = torch.from_numpy(np.array([c.GetPositions() for c in mol.GetConformers()])).to(torch.float)
            values = torch.zeros(coords.shape[0], 6 + len(torsions))
            values[:, 6:] = torch.rand(coords.shape[0], len(torsions)) * np.pi * 2
            for i, (coord, value) in enumerate(zip(coords, values)):
                new_coord = modify_conformer(coord, value, torsions, masks).cpu().data.numpy()
                set_coord(mol, new_coord, i)
    return mol


def clustering2(mol, M=100, N=5):
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    M = min(N * 8, M)
    rdkit_mol = single_conf_gen(mol, num_confs=M)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    total_sz = 0
    sz = len(rdkit_mol.GetConformers())
    tgt_coords = rdkit_mol.GetConformers()[0].GetPositions().astype(np.float32)
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)
    rdkit_coords_list = []
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    ### add no MMFF optimize conformers
    rdkit_mol = single_conf_gen_no_MMFF(mol, num_confs=int(M // 4), seed=43)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    ### add uniform rotation bonds conformers
    rdkit_mol = single_conf_gen_bonds(mol, num_confs=int(M // 4), seed=43)
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    # 部分小分子生成的构象少于聚类数目
    if len(rdkit_coords_list) > N:
        rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(total_sz, -1)
        cluster_size = N
        ids = (
            KMeans(n_clusters=cluster_size, random_state=42, n_init=10)
            .fit_predict(rdkit_coords_flatten)
            .tolist()
        )
        # 部分小分子仅可聚出较少的类
        ids_set = set(ids)
        coords_list = [rdkit_coords_list[ids.index(i)] for i in range(cluster_size) if i in ids_set]
    else:
        coords_list = rdkit_coords_list[:N]
    return coords_list


def sample_pos_from_rdkit(mol, num):
    """根据mol生成构象,据类,返回一个构象列表"""
    
    position_list = clustering2(mol=mol, M=200, N=num)
    add_H_position_list = [ Chem.rdmolops.AddHs( copy.deepcopy(set_coord(mol=mol, coords=pos)), addCoords=True).GetConformer().GetPositions() for pos in position_list]
    return add_H_position_list


def generate_conformer(data, num_sample):
    """
    data 是一个图数据，data.mol是分子的mol对象, data['ligand'].pos是其坐标,键长键角在这里设置
    num_sample 是需要生成的构象数目,并且设置给pos
    该函数需要返回 data 的list
    """
    init_pos = torch.tensor( copy.deepcopy( data['ligand'].pos ) )
    sample_pos_list = sample_pos_from_rdkit(mol=data.mol, num=num_sample)
    data_list = []
    for pos in sample_pos_list:
        pos = torch.tensor(pos)
        delta_vector = torch.mean(init_pos, dim=0, keepdim=True)  - torch.mean(pos, dim=0, keepdim=True) 
        data['ligand'].pos = pos + delta_vector
        data_list.append(copy.deepcopy(data))
    return data_list



def inference_batch_evaluation(model, complex_graphs_dataset, device, t_to_sigma, args, bsz=2, num_sample=25, rot_bool=True, tr_bool=True, tor_bool=True, out_dir='out_pocket_tr_rot', gen_confomer_method='1conformer'):
    """
    gen_confomer_method: 1confomer, 10conformer, 100confomer
    """
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    loader = DataLoader(dataset=complex_graphs_dataset, batch_size=bsz, shuffle=False)
    rmsds = []
    rmsds_all_sample_average = []
    num = 100
    i = 0
    predictions_lists = []
    rmsd_dict = {}
    
    for orig_complex_graph in tqdm(loader):
        
        data_list = orig_complex_graph.to('cpu').to_data_list()
        
        if gen_confomer_method=='1conformer':# 推理100个，键长键角相同, 从ground truth加噪
            data_list_new = []
            for data in data_list:
                if not torch.is_tensor(data['ligand'].pos):
                    data['ligand'].pos = random.choice(data['ligand'].pos)
                for _ in range(num_sample):
                    data_list_new.append(copy.deepcopy(data))
            data_list = data_list_new
    
        elif gen_confomer_method=='10conformer':# 推理100个，每10个的键长键角相同, 从口袋中心加入加白噪声
            data_list_new = []
            
            for data in data_list:
                assert not torch.is_tensor(data['ligand'].pos)
                pos_list = copy.deepcopy(data['ligand'].pos)
                assert len(pos_list)==10
                for pos in pos_list:
                    # 将配体平移到蛋白中心
                    pos_cent = torch.mean(pos, dim=0, keepdim=True)
                    data['ligand'].pos = pos - pos_cent
                    for _ in range(10):
                        data_list_new.append(copy.deepcopy(data))
            data_list = data_list_new
            
        elif gen_confomer_method=='100conformer':# 生成任意个数的键长键角, 从ground truth加噪
            graph_mol_list = []
            for data in data_list: # 对所有的分子生成构象
                graph_mol_list += generate_conformer(data, num_sample)
            data_list = graph_mol_list
        
                
        # 从口袋中心加入加白噪声
        randomize_position(data_list, no_torsion= not tor_bool, no_random= not tr_bool, tr_sigma_max=args.tr_sigma_max, no_rot=not rot_bool)
        
        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling_batch(data_list=data_list, # 
                                                        # model=model.module if hasattr(model, 'module') else model,
                                                        model= model,
                                                        inference_steps=args.inference_steps,
                                                        tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                        tor_schedule=tor_schedule,
                                                        device=device, t_to_sigma=t_to_sigma, model_args=args, 
                                                        rot_bool=rot_bool, tr_bool=tr_bool, tor_bool=tor_bool)
                predictions_lists += predictions_list
                    

            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        
    
        # 评测rmsd
        # 对batch内的不同分子循环[[10] [10] [10]]
        
        data_size = len(predictions_lists)//num_sample
        
        for j in range(data_size):
            predict_list = predictions_lists[j*num_sample:(j+1)*num_sample]
            filterHs = torch.not_equal(predict_list[0]['ligand'].x[:, 0], 0).cpu().numpy() # 过滤H

            if isinstance(predict_list[0]['ligand'].orig_pos, list): # 判断初始ground_truth配体构象是否只给了一个
                predict_list[0]['ligand'].orig_pos = predict_list[0]['ligand'].orig_pos[0]
            # 预测的配体坐标
            ligand_pos = np.asarray(
                [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predict_list]) # 把所有预测的坐标
            
            if out_dir:
                mol_ = Chem.RemoveHs(predict_list[0].mol)
                # 将预测的坐标加上蛋白中心,转化为绝对坐标
                ligand_abs_pos = [pos + predict_list[0].original_center.cpu().numpy() for pos in ligand_pos]
                save_graphs_to_sdf(mol=mol_, predict_pos=ligand_abs_pos, out_dir=out_dir, name=predict_list[0].name)

            # 初始配体坐标 减去蛋白中心
            orig_ligand_pos = np.expand_dims(
                predict_list[0]['ligand'].orig_pos[filterHs] - predict_list[0].original_center.cpu().numpy(), axis=0)
            
            
            #使用对称化后的rmsd 计算方式
            # try:
            #     rmsd = get_symmetry_rmsd(mol, np.squeeze(orig_ligand_pos), [ligand_pos[i] for i in range(ligand_pos.shape[0])], mol)
            # except Exception as e:
            #     print(e)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
           
            rmsds.append( rmsd.reshape((num_sample)) )
            # rmsds_all_sample_average.append(np.average(rmsd))
            
        # break

    rmsds = np.array(rmsds).reshape((len(rmsds), num_sample))
    print(f'Test Data size {len(rmsds)}')
    print_suceess(rmsd_results=np.min(rmsds[:,:1], axis=1), top=1)
    print_suceess(rmsd_results= np.min(rmsds[:,:5], axis=1), top=5)
    print_suceess(rmsd_results= np.min(rmsds[:,:10], axis=1), top=10)
    print_suceess(rmsd_results= np.min(rmsds[:,:15], axis=1), top=15)
    print_suceess(rmsd_results= np.min(rmsds[:,:25], axis=1), top=25)
    print_suceess(rmsd_results=np.min(rmsds[:,:50], axis=1), top=50)
    print_suceess(rmsd_results= np.min(rmsds[:,:100], axis=1), top=100)
    
 

def print_suceess(rmsd_results,top=1, task_name='ligand'):
    print(f'----------------------{task_name}------------------------------')
    print(f'--------------------Top{top}-----------------------')
    print("RMSD < 0.5 : ", np.mean(rmsd_results < 0.5))
    print("RMSD < 1.0 : ", np.mean(rmsd_results < 1.0))
    print("RMSD < 1.5 : ", np.mean(rmsd_results < 1.5))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 2.5 : ", np.mean(rmsd_results < 2.5))
    print("RMSD < 3.0 : ", np.mean(rmsd_results < 3.0))
    print("RMSD < 4.0 : ", np.mean(rmsd_results < 4.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))
    print("avg RMSD : ", np.mean(rmsd_results))

def compute_pocket_rmsd(structure1, structure2):
    atoms1 = [atom for atom in structure1.get_atoms() if atom.name == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.name == 'CA']

    # 创建Superimposer对象，并进行结构对齐
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    super_imposer.apply(structure2.get_atoms())
    # 计算RMSD
    rmsd = super_imposer.rms
    return rmsd

def protein_rmsd(reference_structure, mobile_structure, CA=False, res_id=None):
    """直接计算蛋白的rmsd,不做对齐"""
    reference_atoms_list = []
    mobile_atoms_list = []
    if CA:
        for chain, full_id in res_id:
            reference_atoms_list += [list(a.coord) for a in reference_structure[chain][tuple(full_id)].get_atoms() if a.name=='CA'] 
            mobile_atoms_list +=  [list(a.coord) for a in mobile_structure[chain][tuple(full_id)].get_atoms() if a.name=='CA' ] 
    else:
        for chain, full_id in res_id:
            reference_atoms_list += [list(a.coord) for a in reference_structure[chain][tuple(full_id)].get_atoms() ] 
            mobile_atoms_list +=  [list(a.coord) for a in mobile_structure[chain][tuple(full_id)].get_atoms()  ] 
    reference_atoms = np.array(reference_atoms_list)
    mobile_atoms = np.array(mobile_atoms_list) 
    
    #print(mobile_atoms.shape)
    rmsd = np.mean( np.sqrt( np.sum((reference_atoms - mobile_atoms)**2, axis=1) ) )
    return rmsd


def compute_rmsd_data_list(data_list):
    CA_rmsd = []
    rmsd_list = []
    for data in data_list:
        CA_rmsd.append(protein_rmsd(data['receptor'].orig_rec, data['receptor'].rec, CA=True, res_id=data['receptor'].res_chain_full_id_list))
        rmsd_list.append(protein_rmsd(data['receptor'].orig_rec, data['receptor'].rec, CA=False, res_id=data['receptor'].res_chain_full_id_list))
    return rmsd_list, CA_rmsd

import Bio.PDB
import io

def save_pdb(data, pdb_infer_name):
    name = data.name 
    if pdb_infer_name != 'ground':
        structure = data['receptor'].rec
    else:
        structure = data['receptor'].orig_rec
    selector = data['receptor'].selector
    os.makedirs(f'./out_file/predict/', exist_ok=True)
    os.makedirs(f'./out_file/predict/{name}', exist_ok=True)
    pdb_io = Bio.PDB.PDBIO()
    pdb_io.set_structure(structure)
    # f = io.StringIO()
    pdb_io.save(f'./out_file/predict/{name}/{pdb_infer_name}.pdb', select=selector)
    


def inference_flexible_batch_evaluation(model, complex_graphs_dataset, device, t_to_sigma, args, bsz=2, num_sample=25, rot_bool=True, tr_bool=True, tor_bool=True, out_dir='out_pocket_tr_rot', gen_confomer_method='1conformer'):
    """
    gen_confomer_method: 1confomer, 10conformer, 100confomer
    """
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    loader = DataLoader(dataset=complex_graphs_dataset, batch_size=bsz, shuffle=False)
    rmsds = []
    predictions_lists = []
    pocket_ca_rmsd = []
    pocket_atoms_rmsd = []
    for orig_complex_graph in tqdm(loader):
        
        data_list = orig_complex_graph.to('cpu').to_data_list()
        if os.path.exists(f'./out_file/predict/{data_list[0].name}'):
            continue
        
        if gen_confomer_method=='1conformer':# 推理100个，键长键角相同, 从ground truth加噪
            data_list_new = []
            for data in data_list:
                if not torch.is_tensor(data['ligand'].pos):
                    data['ligand'].pos = random.choice(data['ligand'].pos)
                for _ in range(num_sample):
                    data_list_new.append(copy.deepcopy(data))
            data_list = data_list_new
    
        elif gen_confomer_method=='10conformer':# 推理100个，每10个的键长键角相同, 从口袋中心加入加白噪声
            data_list_new = []
            
            for data in data_list:
                assert not torch.is_tensor(data['ligand'].pos)
                pos_list = copy.deepcopy(data['ligand'].pos)
                assert len(pos_list)==10
                for pos in pos_list:
                    # 将配体平移到蛋白中心
                    pos_cent = torch.mean(pos, dim=0, keepdim=True)
                    data['ligand'].pos = pos - pos_cent
                    for _ in range(10):
                        data_list_new.append(copy.deepcopy(data))
            data_list = data_list_new
            
        elif gen_confomer_method=='100conformer':# 生成任意个数的键长键角, 从ground truth加噪
            graph_mol_list = []
            for data in data_list: # 对所有的分子生成构象
                graph_mol_list += generate_conformer(data, num_sample)
            data_list = graph_mol_list
        

        # 从口袋中心加入加白噪声
        randomize_position(data_list, no_torsion= not tor_bool, no_random= not tr_bool, tr_sigma_max=args.tr_sigma_max, no_rot=not rot_bool)
        # 对残基加平移旋转噪声
        noise_rmsd, noise_ca_rmsd = compute_rmsd_data_list(data_list=data_list)
        print('noise rmsd ', noise_rmsd, noise_ca_rmsd)
        randomize_res_position(data_list=data_list, tr_sigma_max=args.tr_sigma_max)
        #输出初始加噪时,蛋白的rmsd
        noise_rmsd, noise_ca_rmsd = compute_rmsd_data_list(data_list=data_list)
        print('noise rmsd ', noise_rmsd, noise_ca_rmsd)
        
        save_pdb(data=data, pdb_infer_name=f'ground')
        for idx, (rmsd, ca_rmsd, data) in enumerate(zip(noise_rmsd, noise_ca_rmsd, data_list)):
            rmsd = str(rmsd)[:5]
            ca_rmsd = str(ca_rmsd)[:5]
            save_pdb(data=data, pdb_infer_name=f'{idx}_noise_rmsd_{rmsd}_rmsd_ca_{ca_rmsd}')
        
        predictions_list = None
        failed_convergence_counter = 0
        

        try:
            predictions_list, confidences = sampling_flexible_batch(data_list=data_list, # 
                                                    # model=model.module if hasattr(model, 'module') else model,
                                                    model= model,
                                                    inference_steps=args.inference_steps,
                                                    tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                    tor_schedule=tor_schedule,
                                                    device=device, t_to_sigma=t_to_sigma, model_args=args, 
                                                    rot_bool=rot_bool, tr_bool=tr_bool, tor_bool=tor_bool)
            denoise_rmsd, denoise_ca_rmsd = compute_rmsd_data_list(data_list=predictions_list)
            
            
            print('denoise rmsd ', denoise_rmsd, denoise_ca_rmsd)
            
            for idx, (de_rmsd, de_ca_rmsd, data) in enumerate(zip(denoise_rmsd, denoise_ca_rmsd, predictions_list)):
                de_rmsd = str(de_rmsd)[:5]
                de_ca_rmsd = str(de_ca_rmsd)[:5]
                save_pdb(data=data, pdb_infer_name=f'{idx}_infer_rmsd_{de_rmsd}_rmsd_ca_{de_ca_rmsd}')
            
            predictions_lists += predictions_list
                

        except Exception as e:
            continue
            if 'failed to converge' in str(e):
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                    break
                print('| WARNING: SVD failed to converge - trying again with a new sample')
            else:
                raise e
        
        pocket_ca_rmsd.append(np.array(denoise_ca_rmsd))
        pocket_atoms_rmsd.append(np.array(denoise_rmsd))
        # 评测rmsd
        # 对batch内的不同分子循环[[10] [10] [10]]
        
        data_size = len(predictions_lists)//num_sample
        
        for j in range(data_size):
            predict_list = predictions_lists[j*num_sample:(j+1)*num_sample]
            filterHs = torch.not_equal(predict_list[0]['ligand'].x[:, 0], 0).cpu().numpy() # 过滤H

            if isinstance(predict_list[0]['ligand'].orig_pos, list): # 判断初始ground_truth配体构象是否只给了一个
                predict_list[0]['ligand'].orig_pos = predict_list[0]['ligand'].orig_pos[0]
            # 预测的配体坐标
            ligand_pos = np.asarray(
                [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predict_list]) # 把所有预测的坐标
            
            if out_dir:
                mol_ = Chem.RemoveHs(predict_list[0].mol)
                # 将预测的坐标加上蛋白中心,转化为绝对坐标
                ligand_abs_pos = [pos + predict_list[0].original_center.cpu().numpy() for pos in ligand_pos]
                save_graphs_to_sdf(mol=mol_, predict_pos=ligand_abs_pos, out_dir=out_dir, name=predict_list[0].name)

            # 初始配体坐标 减去蛋白中心
            orig_ligand_pos = np.expand_dims(
                predict_list[0]['ligand'].orig_pos[filterHs] - predict_list[0].original_center.cpu().numpy(), axis=0)
            
            
            #使用对称化后的rmsd 计算方式
            # try:
            #     rmsd = get_symmetry_rmsd(mol, np.squeeze(orig_ligand_pos), [ligand_pos[i] for i in range(ligand_pos.shape[0])], mol)
            # except Exception as e:
            #     print(e)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
           
            rmsds.append( rmsd.reshape((num_sample)) )
            # rmsds_all_sample_average.append(np.average(rmsd))
            
        # break

    # rmsds = np.array(rmsds).reshape((len(rmsds), num_sample))
    # pocket_ca_rmsd = np.array(pocket_ca_rmsd).reshape((len(rmsds), num_sample))
    # pocket_atoms_rmsd = np.array(pocket_atoms_rmsd).reshape((len(rmsds), num_sample))
    
    # name1, name2 = data_list[0].name, data_list[-1].name
    # os.makedirs('./out_file/rmsd')
    # with open(f'./out_file/rmsd/{name1}_{name2}_rmsd_result.pkl','wb') as f:
    #     pickle.dump([rmsds, pocket_ca_rmsd, pocket_atoms_rmsd], f)
        
    # print(f'Test Data size {len(rmsds)}')
    print_suceess(rmsd_results=np.min(rmsds[:,:1], axis=1), top=1)
    # print_suceess(rmsd_results= np.min(rmsds[:,:5], axis=1), top=5)
    # print_suceess(rmsd_results= np.min(rmsds[:,:10], axis=1), top=10)
    
    # print_suceess(rmsd_results=np.min(pocket_ca_rmsd[:,:1], axis=1), top=1, task_name='pocket_ca')
    # print_suceess(rmsd_results= np.min(pocket_ca_rmsd[:,:5], axis=1), top=5, task_name='pocket_ca')
    # print_suceess(rmsd_results= np.min(pocket_ca_rmsd[:,:10], axis=1), top=10, task_name='pocket_ca')
    
    # print_suceess(rmsd_results=np.min(pocket_atoms_rmsd[:,:1], axis=1), top=1, task_name='pocket_atoms')
    # print_suceess(rmsd_results= np.min(pocket_atoms_rmsd[:,:5], axis=1), top=5, task_name='pocket_atoms')
    # print_suceess(rmsd_results= np.min(pocket_atoms_rmsd[:,:10], axis=1), top=10, task_name='pocket_atoms')
    # print_suceess(rmsd_results= np.min(rmsds[:,:15], axis=1), top=15)
    # print_suceess(rmsd_results= np.min(rmsds[:,:25], axis=1), top=25)
    # print_suceess(rmsd_results=np.min(rmsds[:,:50], axis=1), top=50)
    # print_suceess(rmsd_results= np.min(rmsds[:,:100], axis=1), top=100)
    



def sanitize_mols(mols):
    for mol in mols:
        # Chem.SanitizeMol(mol)
        # Chem.DetectBondStereochemistry(mol)
        Chem.AssignStereochemistryFrom3D(mol)

def save_sdf(mol_list, output_path, name=''):
    sanitize_mols(mol_list)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with Chem.SDWriter(output_path) as w:
        for i, mol in enumerate(mol_list):
            mol.SetProp('_Name', f'{name}_ligand_{i}')
            w.write(mol)

def set_coord(mol, coords, idx=0):
    _mol = copy.deepcopy(mol)
    for i in range(coords.shape[0]): # 对原子循环
        _mol.GetConformer(idx).SetAtomPosition(i, coords[i].tolist())
    return _mol


def save_graphs_to_sdf(mol, predict_pos, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    mol_list = [set_coord(mol, coords) for coords in predict_pos]
    save_sdf(mol_list, output_path=out_dir + f"/{name}.sdf")


def inference_batch(model, complex_graphs_dataset, device, t_to_sigma, args, rot_bool=True, tr_bool=True, tor_bool=True, out_dir='out_pocket_tr_rot', out_rmsd=None):
    bsz = 16
    num_sample = 10
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    loader = DataLoader(dataset=complex_graphs_dataset, batch_size=bsz, shuffle=False)
    rmsds = []
    rmsds_all_sample_average = []
    num = 100
    i = 0
    predictions_lists = []
    rmsd_dict = {}
    for orig_complex_graph in tqdm(loader):
        # i += 1
        # if i==2:
        #     break
        data_list = orig_complex_graph.to('cpu').to_data_list()
        data_list = [copy.deepcopy(data) for data in data_list for _ in range(num_sample)]
        
        # 随机初始化平移旋转和扭转角
        randomize_position(data_list, no_torsion= not tor_bool, no_random= not tr_bool, tr_sigma_max=args.tr_sigma_max, no_rot=not rot_bool)
        
        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling_batch(data_list=data_list, # 
                                                        # model=model.module if hasattr(model, 'module') else model,
                                                        model= model,
                                                        inference_steps=args.inference_steps,
                                                        tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                        tor_schedule=tor_schedule,
                                                        device=device, t_to_sigma=t_to_sigma, model_args=args, 
                                                        rot_bool=rot_bool, tr_bool=tr_bool, tor_bool=tor_bool)
                predictions_lists += predictions_list
                    

            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        
    
        # 评测rmsd
        # 对batch内的不同分子循环[[10] [10] [10]]
        
        data_size = len(predictions_lists)//num_sample
        
        for j in range(data_size):
            predict_list = predictions_lists[j*num_sample:(j+1)*num_sample]
            filterHs = torch.not_equal(predict_list[0]['ligand'].x[:, 0], 0).cpu().numpy() # 过滤H

            if isinstance(predict_list[0]['ligand'].orig_pos, list): # 判断初始ground_truth配体构象是否只给了一个
                predict_list[0]['ligand'].orig_pos = predict_list[0]['ligand'].orig_pos[0]
            # 预测的配体坐标
            ligand_pos = np.asarray(
                [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predict_list]) # 把所有预测的坐标
            # 初始配体坐标 减去蛋白中心
            orig_ligand_pos = np.expand_dims(
                predict_list[0]['ligand'].orig_pos[filterHs] - predict_list[0].original_center.cpu().numpy(), axis=0)
            
            mol = Chem.RemoveHs(predict_list[0].mol)
            #使用对称化后的rmsd 计算方式
            # try:
            #     rmsd = get_symmetry_rmsd(mol, np.squeeze(orig_ligand_pos), [ligand_pos[i] for i in range(ligand_pos.shape[0])], mol)
            # except Exception as e:
            #     print(e)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            if out_rmsd is not None:
                rmsd_dict[predict_list[0].name] = list(rmsd)
                with open(out_rmsd,'w') as f:
                    json.dump(rmsd_dict, f, indent=2)

            rmsds.append(np.min( rmsd ))
            rmsds_all_sample_average.append(np.average(rmsd))

    rmsds = np.array(rmsds)
    rmsds_all_sample_average = np.array(rmsds_all_sample_average)
    print('average min rmsd',np.average(rmsds))
    print('average all rmsd',np.average(rmsds_all_sample_average))
    losses = {'rmsds_lt1': (100 * (rmsds < 1).sum() / len(rmsds)),
            'rmsds_lt1.5': (100 * (rmsds < 1.5).sum() / len(rmsds)),
            'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
            'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))} # 如果只预测了一个就是对的,不然计算的rmsds偏高
    print(losses)
    return losses