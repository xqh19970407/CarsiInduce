import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta
from Bio.PDB import Selection, PDBIO
from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch, R_from_quaternion_u
from utils.torsion import modify_conformer_torsion_angles
import copy

def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma

def apply_euclidean(x, R):
    """
    R [..., 3, 3]
    x [..., Na, 3]
    """
    Rx = torch.einsum('...kl,...ml->...mk', R, x)
    return Rx

def modify_pocket_res_conformer(data, tr_update, rot_update):
    """
    只修改了CA的坐标和向量特征,模型的训练时只,加噪会用到
    """
    res_pos = data['receptor'].pos.to(torch.float32)
    assert res_pos.shape==tr_update.shape
    data['receptor'].pos = res_pos + tr_update
    if rot_update is not None:
        rot_mat = R_from_quaternion_u(rot_update) # [res_num, 3, 3]
        data['receptor'].side_chain_vecs = apply_euclidean(data['receptor'].side_chain_vecs, rot_mat.float())       # [res_num, 4, 3] 4个向量特征
    return data

def modify_pocket_atoms_conformer(data, tr_update, rot_update):
    """
    修改了全部口袋原子的坐标
    """
    res_pos = data['receptor'].pos.to(torch.float32)
    assert res_pos.shape==tr_update.shape
    # ca原子坐标平移更新
    data['receptor'].pos = res_pos + tr_update # [res_num, 3]
    
    res_atoms_mask = data['receptor'].res_atoms_mask # [res_num, atoms]
    res_atoms_pos = data['receptor'].res_atoms_pos # [res_num, atoms, 3]
    max_atom_num = res_atoms_pos.shape[1]
    
    res_pos = data['receptor'].pos
    res_pos = res_pos.unsqueeze(1).repeat(1, max_atom_num, 1)
    tr_update = tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)
    
    rec = data['receptor'].rec
    
    # 所有原子的坐标平移更新
    data['receptor'].res_atoms_pos[res_atoms_mask] = data['receptor'].res_atoms_pos[res_atoms_mask] + tr_update[res_atoms_mask]
    
    if rot_update is not None:
        rot_mat = R_from_quaternion_u(rot_update) # [res_num, 3, 3]
        # 向量特征的旋转更新
        data['receptor'].side_chain_vecs = apply_euclidean(data['receptor'].side_chain_vecs, rot_mat.float())       # [res_num, 4, 3] 4个向量特征
        # 所有坐标绕CA旋转更新：更新前要减去CA的坐标, 再做旋转, 然后再把CA的坐标加回来
        data['receptor'].res_atoms_pos[res_atoms_mask] = data['receptor'].res_atoms_pos[res_atoms_mask] - res_pos[res_atoms_mask] # [res_num*atoms, 3]
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        data['receptor'].res_atoms_pos[res_atoms_mask] = apply_euclidean(data['receptor'].res_atoms_pos[res_atoms_mask].unsqueeze(1), rot_mat[res_atoms_mask].float()).squeeze(1) + res_pos[res_atoms_mask]
        
    data['receptor'].rec = set_res_atom_pos(data, rec=rec, atoms_pos=data['receptor'].res_atoms_pos[res_atoms_mask])
    
    return data


def set_res_atom_pos(data, rec, atoms_pos):
    res_id = data['receptor'].res_chain_full_id_list
    idx = 0
    for chain,full_id in res_id:
        for atom in rec[chain][tuple(full_id)]:
            atom_pos = tuple(atoms_pos[idx].numpy() + data.original_center.squeeze(0).numpy())
            atom.set_coord(atom_pos)
            idx+=1  
    return rec




def set_atom_pos(rec, atoms_pos):
    rec = copy.deepcopy(rec)
    atoms = Selection.unfold_entities(rec, 'A')
    assert len(atoms)==len(atoms_pos)
    for idx, atom in enumerate(atoms):
        atom_pos = tuple(atoms_pos[idx].numpy())
        atom.set_coord(atom_pos)
    return rec
        
        

def modify_conformer(data, tr_update, rot_update, torsion_updates):
    data['ligand'].pos = data['ligand'].pos.to(torch.float32)
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    
    device = data['ligand'].pos.device


    if rot_update is not None:
        # 旋转
        rot_mat = axis_angle_to_matrix(rot_update.squeeze()).to(torch.float32)
        # 所有原子减去中心原子坐标,进行旋转 再平移 再加中心原子
        rigid_new_pos = (data['ligand'].pos - lig_center.to(device)) @ rot_mat.to(device).T + lig_center.to(device)
        # rigid_new_pos = (data['ligand'].pos - lig_center.to(device))  + lig_center.to(device)
        
    # 平移
    if (tr_update is not None) and (rot_update is not None):
        rigid_new_pos += tr_update.to(device)
    elif (tr_update is not None) and (rot_update is None):
        rigid_new_pos = data['ligand'].pos + tr_update.to(device)
    else:
        rigid_new_pos = data['ligand'].pos

    if torsion_updates is not None:
        # 判断扭转角是否更新
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                           torsion_updates).to(rigid_new_pos.device)
        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        data['ligand'].pos = aligned_flexible_pos
    else:
        data['ligand'].pos = rigid_new_pos
        
    return data



def modify_ligand_pocket_conformer(data, tr_update, rot_update, torsion_updates, res_tr_update, res_rot_update):
    data['ligand'].pos = data['ligand'].pos.to(torch.float32)
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    
    device = data['ligand'].pos.device


    if rot_update is not None:
        # 旋转
        rot_mat = axis_angle_to_matrix(rot_update.squeeze()).to(torch.float32)
        # 所有原子减去中心原子坐标,进行旋转 再平移 再加中心原子
        rigid_new_pos = (data['ligand'].pos - lig_center.to(device)) @ rot_mat.to(device).T + lig_center.to(device)
        # rigid_new_pos = (data['ligand'].pos - lig_center.to(device))  + lig_center.to(device)
        
    # 平移
    if (tr_update is not None) and (rot_update is not None):
        rigid_new_pos += tr_update.to(device)
    elif (tr_update is not None) and (rot_update is None):
        rigid_new_pos = data['ligand'].pos + tr_update.to(device)
    else:
        rigid_new_pos = data['ligand'].pos

    if torsion_updates is not None:
        # 判断扭转角是否更新
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                           torsion_updates).to(rigid_new_pos.device)
        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        data['ligand'].pos = aligned_flexible_pos
    else:
        data['ligand'].pos = rigid_new_pos
    
    modify_pocket_atoms_conformer(data=data, tr_update=res_tr_update, rot_update=res_rot_update)
    
    return data




def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]



def set_listbatch_time(complex_graphs_list, t_tr, t_rot, t_tor, all_atoms, device):
    # 循环代替bsz
    for complex_graphs in complex_graphs_list:
        complex_graphs['ligand'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}
        complex_graphs['receptor'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}
        complex_graphs.complex_t = {'tr': t_tr * torch.ones(1).to(device),
                                'rot': t_rot * torch.ones(1).to(device),
                                'tor': t_tor * torch.ones(1).to(device)}
        if all_atoms:
            complex_graphs['atom'].node_t = {
                'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
                'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
                'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device)}



def set_time(complex_graphs, t_tr, t_rot, t_tor, batchsize, all_atoms, device):
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device)
        }
    
    complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                               'rot': t_rot * torch.ones(batchsize).to(device),
                               'tor': t_tor * torch.ones(batchsize).to(device)}
    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device)}