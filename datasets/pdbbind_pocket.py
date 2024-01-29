import binascii
import glob
import hashlib
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
import json
from typing import Callable, Optional
import numpy as np
import torch
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
import pandas as pd
from datasets.process_mols_pocket import read_molecule, get_rec_graph, generate_conformer, \
                                         get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, \
                                         parse_pdb_from_path,extract_receptor_pocket_structure,parse_esm_PDB,\
                                         extract_esmProtein_crystalProtein, read_sdf_to_mol_list,add_rec_vector_infor,\
                                         get_lig_feature
                                         
from utils.diffusion_utils import modify_conformer, set_time, modify_pocket_res_conformer
# from utils.utils import read_strings_from_txt

from utils.geometry import axis_angle_to_matrix
from utils.geometry import rigid_transform_Kabsch_3D_torch
import warnings
from typing import List, Union
from Bio.PDB import PDBParser
import torch
from rdkit import Chem
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from Bio.PDB import Selection
from scipy.spatial.distance import cdist
from .process_mols_pocket import safe_index, allowable_features
from collections import Counter
import Bio.PDB
import io
three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

residue_atoms = {
    "ALA": {"C", "CA", "CB", "N", "O"},# 丙氨酸
    "ARG": {"C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"},
    "ASP": {"C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"},
    "ASN": {"C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"},
    "CYS": {"C", "CA", "CB", "N", "O", "SG"},
    "GLU": {"C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"},
    "GLN": {"C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"},
    "GLY": {"C", "CA", "N", "O"}, # 甘氨酸
    "HIS": {"C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"},
    "ILE": {"C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"},
    "LEU": {"C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"},# 缬氨酸
    "LYS": {"C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"},# 赖氨酸
    "MET": {"C", "CA", "CB", "CG", "CE", "N", "O", "SD"},
    "PHE": {"C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"},#苯丙氨酸
    "PRO": {"C", "CA", "CB", "CG", "CD", "N", "O"},
    "SER": {"C", "CA", "CB", "N", "O", "OG"},# 苏氨酸
    "THR": {"C", "CA", "CB", "CG2", "N", "O", "OG1"},
    "TRP": {"C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O", },
    "TYR": {"C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH", },
    "VAL": {"C", "CA", "CB", "CG1", "CG2", "N", "O"},
}




def collate_fn(data_list):
    graph_list = []
    for graph in data_list:
        if graph is not None:
            graph_list.append(graph)
    return graph_list


class DataListLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, List[BaseData]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)






class TRLabel(BaseTransform):
    def __init__(self):
        
        pass
    
    def __call__(self, data):
        
        return self.compute_RT(data)
    
    def compute_RT(self, data):
        # 传入的配体坐标如果是一个列表,那么从列表中抽取一个坐标
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
            
        data.tr_score = self.get_lig_translation(data)
        lig_rmsd = self.rmsd_test(x=data['ligand'].pos, y=data['ligand'].ref_pos)
        sota_lig_rmsd = self.rmsd_test(x=self.modify_lig_conformer(x=data['ligand'].pos, tr_update=data.tr_score), y=data['ligand'].ref_pos)
        print(f'{data.name} lig rmsd/sota_rmsd: ', lig_rmsd.item(), sota_lig_rmsd.item())
        data.res_tr_score = self.get_res_translation(data)
        data.res_rot_score = self.get_res_rotation_vector(data)
        
        pocket_rmsd = self.rmsd_test(x=data['receptor'].res_atoms_pos, 
                                     y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        sota_pocket_rmsd = self.rmsd_test(x=self.modify_pocket_conformer(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                     y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        
        print('pocket rmsd/sota_rmsd: ', pocket_rmsd, sota_pocket_rmsd)
        return data
    
    def get_lig_translation(self, data):
        assert data['ligand'].ref_pos.shape[0] == data['ligand'].pos.shape[0]
        lig_tr = data['ligand'].ref_pos - data['ligand'].pos
        return lig_tr # [lig_atoms,3]
    
    def get_res_translation(self, data):
        res_tr = data['receptor'].ref_pos - data['receptor'].pos
        return res_tr
    
    def get_res_rotation_vector(self, data):
        
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        ref_atoms_pos = data['receptor'].ref_res_atoms_pos    # ground truth 所有原子坐标
        pos_mask = data['receptor'].res_atoms_mask
        pos_ca = data['receptor'].pos
        ref_ca = data['receptor'].ref_pos
        rot_mat =self.point_cloud_to_ror_matrix(pos=atoms_pos, ref=ref_atoms_pos, pos_mask=pos_mask, pos_ca=pos_ca, ref_ca=ref_ca) #[res, 3, 3]
        rot_vector = self.R_batch_to_axis_vec(R=rot_mat) #[res, 3]
        return rot_vector
    
    def point_cloud_to_ror_matrix(self, pos, ref, pos_mask, pos_ca=None, ref_ca=None):
        # 两堆点云计算旋转矩阵
        # pos [N,M,3] [res, atoms, 3]
        # ref [N,M,3]
        # pos_mask [N,M]
        # pos_ca, ref_ca [N,3]
        # N : number of examples
        # M : number of atoms
        # R,T maps local reference onto global pos
        if pos_mask is None:
            pos_mask = torch.ones(pos.shape[:2], device=pos.device)
        else:
            if pos_mask.shape[0] != pos.shape[0]:
                raise ValueError("pos_mask should have same number of rows as number of input vectors.")
            if pos_mask.shape[1] != pos.shape[1]:
                raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
            if pos_mask.ndim != 2:
                raise ValueError("pos_mask should be 2 dimensional.")
        if pos_ca is None:
            # 使用点云中心为旋转中心
            denom = torch.sum(pos_mask, dim=1, keepdim=True)
            denom[denom == 0] = 1.
            pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
            ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
            pos_c = pos - pos_mu
            ref_c = ref - ref_mu
        else:
            # 使用CA为旋转中心
            pos_c = pos - pos_ca.unsqueeze(1)
            ref_c = ref - ref_ca.unsqueeze(1)
        # Covariance matrix
        H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c.to(torch.float32))
        U, S, Vh = torch.linalg.svd(H)
        # Decide whether we need to correct rotation matrix to ensure right-handed coord system
        locs = torch.linalg.det(U @ Vh) < 0
        S[locs, -1] = -S[locs, -1]
        U[locs, :, -1] = -U[locs, :, -1]
        # Rotation matrix
        R = torch.einsum('bji,bkj->bik', Vh, U)
        
        return R
    def R_batch_to_axis_vec(self, R):
        """
        旋转矩阵到旋转向量
        [N,3,3] >>> [N,3]
        """
        J = (R - R.permute([0,2,1])) / 2
        t = torch.cat( [ (-J[:, 1, 2]).unsqueeze(1) , (J[:, 0, 2]).unsqueeze(1),  (-J[:, 0, 1]).unsqueeze(1)], dim=1)
        t_norm = torch.norm(t, dim=1, keepdim=True)
        theta = torch.asin(t_norm)
        angle = theta / np.pi * 180
        r = t / t_norm
        v = r * theta
        return v
    
    def rmsd_test(self, x, y, mask=None):
        # x y [res, atoms, 3]
        if mask is not None:
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y
        # x y [atoms, 3]
        
        return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))
    
    def modify_lig_conformer(self, x, tr_update):
        return x + tr_update
    
    def modify_pocket_conformer(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        pos_ca = data['receptor'].pos # [res, 3] 
        max_atom_num = atoms_pos.shape[1]
        
        # 先绕着CA做旋转,再把CA加回来
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        atoms_pos[pos_mask] = atoms_pos[pos_mask] - (pos_ca.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = apply_euclidean(atoms_pos[pos_mask].unsqueeze(1) , rot_mat[pos_mask].float()).squeeze(1) 
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (pos_ca.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 对每个残基做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    

class TRLabel_Point_Cloud_Center(BaseTransform):
    def __init__(self):
        pass
    
    def __call__(self, data):
        
        return self.compute_RT(data)
    
    def compute_RT(self, data):
        # 传入的配体坐标如果是一个列表,那么从列表中抽取一个坐标
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
            
        data.tr_score = self.get_lig_translation(data)
        lig_rmsd = self.rmsd_test(x=data['ligand'].pos, y=data['ligand'].ref_pos)
        sota_lig_rmsd = self.rmsd_test(x=self.modify_lig_conformer(x=data['ligand'].pos, tr_update=data.tr_score), y=data['ligand'].ref_pos)
        print(f'{data.name} lig rmsd/sota_rmsd: ', lig_rmsd.item(), sota_lig_rmsd.item())
        data.res_rot_score, data.res_tr_score = self.get_res_rotation_vector(data)
        
        
        pocket_rmsd = self.rmsd_test(x=data['receptor'].res_atoms_pos, 
                                     y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()

        sota_pocket_rmsd = self.rmsd_test(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                     y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()
        data['receptor'].sota_pocket_rmsd = sota_pocket_rmsd
        
        print(f'{data.name} pocket rmsd/sota_rmsd: ', pocket_rmsd, sota_pocket_rmsd)
        print(f'{data.name} esm-fold rmsd: ',data['receptor'].esm_rmsd)
        
        
        
        return data
    
    def get_lig_translation(self, data):
        assert data['ligand'].ref_pos.shape[0] == data['ligand'].pos.shape[0]
        lig_tr = data['ligand'].ref_pos - data['ligand'].pos
        return lig_tr # [lig_atoms,3]
    
    def get_res_translation(self, data):
        res_tr = data['receptor'].ref_pos - data['receptor'].pos
        return res_tr
    
    def get_res_rotation_vector(self, data):
        
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        ref_atoms_pos = data['receptor'].ref_res_atoms_pos    # ground truth 所有原子坐标
        pos_mask = data['receptor'].res_atoms_mask
        
        rot_mat, tr_vec =self.point_cloud_to_ror_matrix(ref_atoms_pos, atoms_pos, pos_mask=pos_mask) #[res, 3, 3]
        rot_vector = self.R_batch_to_axis_vec(R=rot_mat) #[res, 3]
        return rot_vector, tr_vec
    
    def point_cloud_to_ror_matrix(self, pos, ref, pos_mask, pos_ca=None, ref_ca=None):
        # 两堆点云计算旋转矩阵
        # pos [N,M,3] [res, atoms, 3]
        # ref [N,M,3]
        # pos_mask [N,M]
        # pos_ca, ref_ca [N,3]
        # N : number of examples
        # M : number of atoms
        # R,T maps local reference onto global pos
        if pos_mask is None:
            pos_mask = torch.ones(pos.shape[:2], device=pos.device)
        else:
            if pos_mask.shape[0] != pos.shape[0]:
                raise ValueError("pos_mask should have same number of rows as number of input vectors.")
            if pos_mask.shape[1] != pos.shape[1]:
                raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
            if pos_mask.ndim != 2:
                raise ValueError("pos_mask should be 2 dimensional.")
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        pos_c = pos - pos_mu
        ref_c = ref - ref_mu
       
        # Covariance matrix
        H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c.to(torch.float32))
        U, S, Vh = torch.linalg.svd(H)
        # Decide whether we need to correct rotation matrix to ensure right-handed coord system
        locs = torch.linalg.det(U @ Vh) < 0
        S[locs, -1] = -S[locs, -1]
        U[locs, :, -1] = -U[locs, :, -1]
        # Rotation matrix
        R = torch.einsum('bji,bkj->bik', Vh, U)
        T = pos_mu - torch.einsum('bij,bkj->bki', R, ref_mu)
        return R,T.squeeze(1)
    
    def R_batch_to_axis_vec(self, R):
        """
        旋转矩阵到旋转向量
        [N,3,3] >>> [N,3]
        """
        J = (R - R.permute([0,2,1])) / 2
        t = torch.cat( [ (-J[:, 1, 2]).unsqueeze(1) , (J[:, 0, 2]).unsqueeze(1),  (-J[:, 0, 1]).unsqueeze(1)], dim=1)
        t_norm = torch.norm(t, dim=1, keepdim=True)
        theta = torch.asin(t_norm)
        angle = theta / np.pi * 180
        r = t / t_norm
        v = r * theta
        return v
    
    def rmsd_test(self, x, y, mask=None):
        # x y [res, atoms, 3]
        if mask is not None:
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y
        # x y [atoms, 3]
        
        return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))
    
    def pocket_rmsd(self, x,y,mask):
        rmsd = []
        for i in range(len(mask)):
            rmsd.append( self.rmsd_test(x[i],y[i],mask[i]))
        max_idx = rmsd.index(max(rmsd))
        print('res max/mean rmsd ', max(rmsd), self.rmsd_test(x,y,mask))
        return max_idx
    
    def modify_lig_conformer(self, x, tr_update):
        return x + tr_update
    
    def modify_pocket_conformer(self, data, tr_update, rot_update):
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
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
    def modify_pocket_conformer2(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = copy.deepcopy( data['receptor'].res_atoms_pos )   # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        max_atom_num = atoms_pos.shape[1]
        
        # 做旋转
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        atoms_pos[pos_mask] = torch.einsum('bij,bkj->bki',rot_mat[pos_mask].float(), atoms_pos[pos_mask].unsqueeze(1)).squeeze(1) 
        
        # 做平移
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
    
def apply_euclidean(x, R):
    """
    R [..., 3, 3]
    x [..., Na, 3]
    """
    Rx = torch.einsum('...kl,...ml->...mk', R, x)
    return Rx



class Datalist_to_PDBBind(Dataset):
    def __init__(self, data_list):
        super(Datalist_to_PDBBind, self).__init__()
        self.complex_graphs = data_list

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        return copy.deepcopy(self.complex_graphs[idx])

    
class FlexiDataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.protein_ligand_paths = [('1aqp.pdb','1aqp.sdf'),...]
        pass

    def len(self):
        pass

    def get(self,idx):
        data_path = self.protein_ligand_paths[idx]
        data = self.from_pdb_and_sdf_to_graph(data_path=data_path)
        return data

    def from_pdb_and_sdf_to_graph(self, data_path):
        data = None
        return data
    
# pdb
# 
# train_dataset
# test_datset

class PDBBind(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0, data_type='train',
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15, task_name='protein_ligand_flexible_docking_320',
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False, max_data_size=10000000, start_idx=None):

        super(PDBBind, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.data_type = data_type
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures

        self.max_data_size = max_data_size
        cache_path  = os.path.join(cache_path, task_name)
        self.full_cache_path = os.path.join(cache_path, data_type)
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        
        print('loading data from memory: ', os.path.join(self.full_cache_path, "heterographs.pkl"))
        with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
            complex_graphs = pickle.load(f)[start_idx:self.max_data_size]
        if require_ligand:
            with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                rdkit_ligands = pickle.load(f)[start_idx:self.max_data_size]
        
        self.complex_graphs = []
        self.rdkit_ligands = []
        for complex_graph,rdkit_ligand in zip(complex_graphs, rdkit_ligands):
            complex_graph['receptor'].esm_rmsd = self.get_pocket_rmsd(complex_graph=complex_graph)
            if complex_graph['receptor'].esm_rmsd<5:
                data = self.compute_RT(complex_graph)
                # if data['receptor'].esm_rmsd > data['receptor'].sota_pocket_rmsd:
                self.complex_graphs.append(data)
                self.rdkit_ligands.append(rdkit_ligand)

        print_statistics(self.complex_graphs)

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        if self.require_ligand:
            complex_graph = copy.deepcopy(self.complex_graphs[idx])
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
            return complex_graph
        else:
            return copy.deepcopy(self.complex_graphs[idx])

    
    def get_pocket_rmsd(self, complex_graph):
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask
        res_atoms_pos = complex_graph['receptor'].res_atoms_pos[res_atoms_mask]
        ref_res_atoms_pos = complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask]
        return torch.sqrt(torch.sum((res_atoms_pos-ref_res_atoms_pos)**2,dim=1)).mean().item()
    
    def compute_RT(self, data):
        # 传入的配体坐标如果是一个列表,那么从列表中抽取一个坐标
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
            
        data.tr_score = self.get_lig_translation(data)
        lig_rmsd = self.rmsd_test(x=data['ligand'].pos, y=data['ligand'].ref_pos)
        sota_lig_rmsd = self.rmsd_test(x=self.modify_lig_conformer(x=data['ligand'].pos, tr_update=data.tr_score), y=data['ligand'].ref_pos)
        # print(f'{data.name} lig rmsd/sota_rmsd: ', lig_rmsd.item(), sota_lig_rmsd.item())
        data.res_rot_score, data.res_tr_score = self.get_res_rotation_vector(data)
        
        pocket_rmsd = self.rmsd_test(x=data['receptor'].res_atoms_pos, 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()

        print(f'{data.name} pocket rmsd: ', pocket_rmsd)
        res_rmsd = self.pocket_rmsd(x=data['receptor'].res_atoms_pos, y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        
        sota_pocket_rmsd = self.rmsd_test(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()
        
        aligned_res_rmsd = self.pocket_rmsd(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        deca_list = [(x1-x2)/x1 for x1,x2 in zip(res_rmsd,aligned_res_rmsd) if x1>x2]
        deca = np.average(deca_list)
        rate = len(deca_list)/len(res_rmsd)
        
        data['receptor'].deca = deca
        data['receptor'].rate = rate
        data['receptor'].sota_pocket_rmsd = sota_pocket_rmsd
        print('rate',rate, 'deca rate ', np.average(deca))
        
        # print(f'{data.name} esm-fold rmsd: ',data['receptor'].esm_rmsd)
        
        
        
        return data
        
    def get_lig_translation(self, data):
        assert data['ligand'].ref_pos.shape[0] == data['ligand'].pos.shape[0]
        lig_tr = data['ligand'].ref_pos - data['ligand'].pos
        return lig_tr # [lig_atoms,3]
    
    def get_res_translation(self, data):
        res_tr = data['receptor'].ref_pos - data['receptor'].pos
        return res_tr
    
    def get_res_rotation_vector(self, data):
        
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        ref_atoms_pos = data['receptor'].ref_res_atoms_pos    # ground truth 所有原子坐标
        pos_mask = data['receptor'].res_atoms_mask
        
        rot_mat, tr_vec =self.point_cloud_to_ror_matrix(ref_atoms_pos, atoms_pos, pos_mask=pos_mask) #[res, 3, 3]
        rot_vector = self.R_batch_to_axis_vec(R=rot_mat) #[res, 3]
        return rot_vector, tr_vec
        
    def point_cloud_to_ror_matrix(self, pos, ref, pos_mask, pos_ca=None, ref_ca=None):
        # 两堆点云计算旋转矩阵
        # pos [N,M,3] [res, atoms, 3]
        # ref [N,M,3]
        # pos_mask [N,M]
        # pos_ca, ref_ca [N,3]
        # N : number of examples
        # M : number of atoms
        # R,T maps local reference onto global pos
        if pos_mask is None:
            pos_mask = torch.ones(pos.shape[:2], device=pos.device)
        else:
            if pos_mask.shape[0] != pos.shape[0]:
                raise ValueError("pos_mask should have same number of rows as number of input vectors.")
            if pos_mask.shape[1] != pos.shape[1]:
                raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
            if pos_mask.ndim != 2:
                raise ValueError("pos_mask should be 2 dimensional.")
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        pos_c = pos - pos_mu
        ref_c = ref - ref_mu
    
        # Covariance matrix
        H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c.to(torch.float32))
        U, S, Vh = torch.linalg.svd(H)
        # Decide whether we need to correct rotation matrix to ensure right-handed coord system
        locs = torch.linalg.det(U @ Vh) < 0
        S[locs, -1] = -S[locs, -1]
        U[locs, :, -1] = -U[locs, :, -1]
        # Rotation matrix
        R = torch.einsum('bji,bkj->bik', Vh, U)
        T = pos_mu - torch.einsum('bij,bkj->bki', R, ref_mu)
        return R,T.squeeze(1)
        
    def R_batch_to_axis_vec(self, R):
        """
        旋转矩阵到旋转向量
        [N,3,3] >>> [N,3]
        """
        J = (R - R.permute([0,2,1])) / 2
        t = torch.cat( [ (-J[:, 1, 2]).unsqueeze(1) , (J[:, 0, 2]).unsqueeze(1),  (-J[:, 0, 1]).unsqueeze(1)], dim=1)
        t_norm = torch.norm(t, dim=1, keepdim=True)
        theta = torch.asin(t_norm)
        angle = theta / np.pi * 180
        r = t / t_norm
        v = r * theta
        return v
    
    def rmsd_test(self, x, y, mask=None):
        # x y [res, atoms, 3]
        if mask is not None:
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y
        # x y [atoms, 3]
        
        return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))
        
    def pocket_rmsd(self, x,y,mask):
        rmsd = []
        for i in range(len(mask)):
            rmsd.append( self.rmsd_test(x[i],y[i],mask[i]).item() )
        # max_idx = rmsd.index(max(rmsd))
        # print('res max/mean rmsd ', max(rmsd), self.rmsd_test(x,y,mask))
        return rmsd
    
    def modify_lig_conformer(self, x, tr_update):
        return x + tr_update
        
    def modify_pocket_conformer(self, data, tr_update, rot_update):
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
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
        
    def modify_pocket_conformer2(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = copy.deepcopy( data['receptor'].res_atoms_pos )   # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        max_atom_num = atoms_pos.shape[1]
        
        # 做旋转
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = vec_to_R(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        # atoms_pos[pos_mask] = torch.einsum('bij,bkj->bki',rot_mat[pos_mask].float(), atoms_pos[pos_mask].unsqueeze(1)).squeeze(1) 
        
        # 做平移
        # atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 平移旋转
        atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + tr_update.unsqueeze(1)
        
        return atoms_pos

def vec_to_R(vec):
    vec = vec.numpy()
    res_num = len(vec)
    R_list = []
    for i in range(res_num):
        R_list.append( rodrigues_rotation_vec_to_R(vec[i])[:3,:3])
    return torch.from_numpy( np.array(R_list) )


def rodrigues_rotation_vec_to_R(v):
    # r旋转向量[3x1]
    theta = np.linalg.norm(v)
    r = np.array(v).reshape(3, 1) / theta
    return rodrigues_rotation(r, theta)

def rodrigues_rotation(r, theta):
    # n旋转轴[3x1]
    # theta为旋转角度
    # 旋转是过原点的，n是旋转轴
    r = np.array(r).reshape(3, 1)
    rx, ry, rz = r[:, 0]
    M = np.array([
        [0, -rz, ry],
        [rz, 0, -rx],
        [-ry, rx, 0]
    ])
    R = np.eye(4)
    R[:3, :3] = np.cos(theta) * np.eye(3) +        \
                (1 - np.cos(theta)) * r @ r.T +    \
                np.sin(theta) * M
    return R
def apply_euclidean(x, R):
    """
    R [..., 3, 3]
    x [..., Na, 3]
    """
    Rx = torch.einsum('...kl,...ml->...mk', R, x)
    return Rx

class PDBBind_Graph_Save(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', data_type='train', task_name='protein_ligand_flexible_docking',
                 split_path='data/', limit_complexes=0,start_from_pdbid=None,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False, data_test=True):

        super(PDBBind_Graph_Save, self).__init__(root, transform)
        self.pdbbind_dir = root # pdb文件路径
        self.generate_ligand_dir = '/userdata/xiaoqi/EsmFoldPredict/ligand10conformers'
        self.max_lig_size = max_lig_size # 配体原子的最大尺寸
        self.split_path = split_path # 训练集测试集划分
        self.data_type = data_type
        self.start_from_pdbid = start_from_pdbid
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.data_test = data_test
        cache_path  = os.path.join(cache_path, task_name)
        self.full_cache_path = os.path.join(cache_path, data_type)
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.esm_pdb_path = '/userdata/xiaoqi/EsmFoldPredict/align_pdb_0529'
        os.makedirs(self.full_cache_path, exist_ok=True)
        self.preprocessing()
        
        print('loading data from memory: ', os.path.join(self.full_cache_path, "heterographs.pkl"))
        with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
            self.complex_graphs = pickle.load(f)
        if require_ligand:
            with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                self.rdkit_ligands = pickle.load(f)

        print_statistics(self.complex_graphs)

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        if self.require_ligand:
            complex_graph = copy.deepcopy(self.complex_graphs[idx])
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
            return complex_graph
        else:
            return copy.deepcopy(self.complex_graphs[idx])

    
    def load_json(self, data_type='train'):
        with open(self.split_path, 'r') as f:
            dict_data = json.load(f)
        return dict_data[data_type]
    
    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        complex_names_all = self.load_json(data_type=self.data_type) # 读取所有蛋白 训练集/验证集
        if self.start_from_pdbid is not None:
            print(f'start from {self.start_from_pdbid}')
            start_idx = complex_names_all.index(self.start_from_pdbid)
            print(f'start from {start_idx}')
            complex_names_all = complex_names_all[start_idx:]
            
        
        if self.limit_complexes is not None and self.limit_complexes != 0:# 只处理特定条
            complex_names_all = complex_names_all[:self.limit_complexes]
        
        if self.esm_pdb_path is not None:
            esm_pdb_names = set( [file_name[:4] for file_name in os.listdir(self.esm_pdb_path)] )
            complex_names_all = list(set(complex_names_all) & esm_pdb_names)
        
        if self.data_test:# 只处理前10条
            complex_names_all = complex_names_all[:16]
            
        print(f'Loading {len(complex_names_all)} complexes.')

        if self.esm_embeddings_path is not None:
            # id_to_embeddings = torch.load('/mnt/d/data/esm2_3billion_embeddings.pt')
            id_to_embeddings = torch.load(self.esm_embeddings_path)

            chain_embeddings_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                tmp = key.split('_')
                key_name, chain_id = tmp[0], int(tmp[-1])
                if key_name in complex_names_all:
                    chain_embeddings_dictlist[key_name].append((chain_id, embedding))
            new_embeddings_dict = dict()
            for key, embedding in chain_embeddings_dictlist.items():
                sorted_embedding = sorted(embedding, key=lambda x: x[0])
                new_embeddings_dict[key] = [i[1] for i in sorted_embedding]
            chain_embeddings_dictlist = new_embeddings_dict
            
            lm_embeddings_chains_all = []
            esm_not_exit_embedding = []
            for name in complex_names_all:
                try:
                    lm_embeddings_chains_all.append(chain_embeddings_dictlist[name])
                except:
                    esm_not_exit_embedding.append(name)
            for name in esm_not_exit_embedding:
                del complex_names_all[complex_names_all.index(name)]
            
            print(esm_not_exit_embedding)
            print('esm not exit embedding num ',len(esm_not_exit_embedding))

        else:
            raise ValueError
            
        complex_graphs, rdkit_ligands = [], []
        with tqdm(total=len(complex_names_all), desc=f' loading complexes ') as pbar:
            idx = 0
            for t in map(self.get_complex, zip(complex_names_all, lm_embeddings_chains_all, [None] * len(complex_names_all), [None] * len(complex_names_all))):
            # for t in map(self.get_complex, zip(complex_names_all, lm_embeddings_chains_all[start_idx:(start_idx+1000 )], [None] * len(complex_names_all), [None] * len(complex_names_all))):
                complex_graphs.extend(t[0])
                rdkit_ligands.extend(t[1])
                print(len(complex_graphs))
                pbar.update()
                idx += 1
                # 每隔100条存一次
                if idx%100==0:
                    print('saveing graph')
                    with open(os.path.join(self.full_cache_path, f"heterographs.pkl"), 'wb') as f:
                        pickle.dump((complex_graphs), f)
                    with open(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), 'wb') as f:
                        pickle.dump((rdkit_ligands), f)
            # 最终再存一次
            with open(os.path.join(self.full_cache_path, f"heterographs.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)

    def assert_tor(self, edge_index, mask_rotate):

        for idx_edge, e in enumerate(edge_index.cpu().numpy()):
            u, v = e[0], e[1]
            # check if need to reverse the edge, v should be connected to the part that gets rotated
            if  not ( (not mask_rotate[idx_edge, u]) and mask_rotate[idx_edge, v]):
                raise ValueError('torsion assert error')
        

    def get_complex(self, par):
        name, lm_embedding_chains, ligand, ligand_description = par
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
            print("Folder not found", name)
            return [], []
        # 模型的输入
        # 配体: rdkit生成,坐标norm化
        # 蛋白: esmFold生成,
        # 模型的label:
        # 配体: 晶体数据,坐标根据蛋白口袋中心norm化
        # 蛋白: 晶体数据,根据配体割口袋,根据口袋中心norm化
        if ligand is not None:
            rec_model = parse_pdb_from_path(name)
            name = f'{name}____{ligand_description}'
            ligs = [ligand]
        else:
            try:
                rec_model = parse_receptor(name, self.pdbbind_dir) # PDB文件读取晶体数据
                esm_rec = parse_esm_PDB(name, pdbbind_dir=self.esm_pdb_path) # pdb文件读取esm结构数据
                rec_model, esm_rec = extract_esmProtein_crystalProtein(rec=rec_model, esm_rec=esm_rec) # 同时前处理
                
            except Exception as e:
                print(f'Skipping {name} (bio load) because of the error:')
                print(e)
                return [], []
        # ligs_rdkit_gen = read_sdf_to_mol_list(os.path.join(self.generate_ligand_dir, f'{name}_ligand.sdf'), sanitize=True, remove_hs=False)
        ligs = read_all_mols(self.pdbbind_dir, name, remove_hs=False)
        
        
        complex_graphs = []
        failed_indices = []
        # print('ligs len ',len(ligs))
        # 有的配体可能失败,如果sdf和mol2有一个成功,那么就可结束
        for k, lig in ligs.items():
            
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                break
            complex_graph = HeteroData()# 定义异构图
            complex_graph['name'] = name
            # try:
            # 在图中添加配体信息
            
            useful_key = None
            try:
                print(f'using {name} {k}')
                
                
                get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                            self.num_conformers, remove_hs=self.remove_hs)
                self.assert_tor(complex_graph['ligand', 'ligand'].edge_index.T[complex_graph['ligand'].edge_mask], complex_graph['ligand'].mask_rotate if isinstance(complex_graph['ligand'].mask_rotate, np.ndarray) else complex_graph['ligand'].mask_rotate[0])
                useful_key = k
                    
                # except Exception as e:
                #     print(e)
                #     print(f'{name} {k} is not using')
                #     continue
                
                # 利用bio 返回蛋白相关的信息，alpha_C坐标，embeding等
                # rec是残基类型
                feature_res, rec_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, lm_embeddings,\
                    rec, close_residues, selector, res_chain_full_id_list = \
                        extract_receptor_pocket_structure(copy.deepcopy(rec_model), lig, lm_embedding_chains=lm_embedding_chains)
                
                
                if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                    print(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                    break
                # 对graph添加残基ground_truth的标量和向量信息
                get_rec_graph(feature_res, rec_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, complex_graph, 
                                rec=rec, res_list=close_residues, selector=selector, res_chain_full_id_list=res_chain_full_id_list,
                                rec_radius=self.receptor_radius,c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                                atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
                add_rec_vector_infor(complex_graph=complex_graph, res_chain_full_id_list=res_chain_full_id_list, 
                                    pdb_rec=esm_rec, ref_sorted_atom_names=complex_graph['receptor'].ref_sorted_atom_names)
            except Exception as e:
                print(f'Skipping {name} (process) because of the error:')
                print(e)
                break

            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
            
            complex_graph['receptor'].pos -= protein_center
            complex_graph['receptor'].ref_pos -= protein_center # 蛋白CA
               
            res_atoms_mask = complex_graph['receptor'].res_atoms_mask # [N_res,]
            complex_graph['receptor'].res_atoms_pos[res_atoms_mask] -= protein_center # esm-fold生成的蛋白坐标以晶体蛋白口袋中心为坐标系  [N_res, atoms, 3]
            complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask] -= protein_center  # 晶体蛋白坐标以蛋白口袋中心为坐标系
            complex_graph['receptor'].esm_rmsd = self.get_pocket_rmsd(complex_graph)
            print('esmfold and crystal pocket rmsd: ', complex_graph['receptor'].esm_rmsd)
            
            if (not self.matching) or self.num_conformers == 1:
                # complex_graph['ligand'].pos -= protein_center
                complex_graph['ligand'].pos -= ligand_center # 随机生成的构象进行中心化
                complex_graph['ligand'].ref_pos -= protein_center # ground_truth的配体坐标以蛋白口袋中心为坐标系
                
            else:
                # 多构象
                for p in complex_graph['ligand'].pos:
                    p -= protein_center

            complex_graph.original_center = protein_center
                    
            complex_graphs.append(complex_graph)
            break
        
        if useful_key is not None and len(complex_graphs)>0:
            ligs = [ligs[useful_key]]
        else:
            ligs = []
        return complex_graphs, ligs
    def get_pocket_rmsd(self,complex_graph):
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask
        res_atoms_pos = complex_graph['receptor'].res_atoms_pos[res_atoms_mask]
        ref_res_atoms_pos = complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask]
        return torch.sqrt(torch.sum((res_atoms_pos-ref_res_atoms_pos)**2,dim=1)).mean().item()


def load_json(path):
    with open(path, 'r') as f:
        dict_data = json.load(f)
    return dict_data

# 2913 同源蛋白,根据文件夹名称排序,前2800个同源蛋白作为训练集
# 最后113个同源蛋白作为测试集

# 肽链条数比对，排除肽链条数不对应的蛋白,
# 去掉水，去掉非20类人体氨基酸，得到两个氨基酸序列
# 求出氨基酸的最大公共子序列，排除T小于0.9的蛋白
# 根据氨基酸最大子序列得到esm2的特征，这一步可以先不做
# 根据氨基酸最大子序列给出口袋内的idx——完成了割口袋
# 对两个口袋进行α-C的对齐,大于5A的的数据进行排除
# 对五个关键原子不存在的残基进行标注mask,不对该系列残基训练,但其它残基可以训练

# 真正的蛋白配体
# (pdb_in, lig_in, pdb_out, lig_out)
# 65w pdb_in - lig_out
# 11w pdb_out - lig_out   ground_truth
# 2w lig_out
# 根据同源蛋白,划分100个同源蛋白出来
# 2900=2800+100
# (pdb_in, lig_in, pdb_out, lig_out) 这是一条数据

def load_data_type(data_type='train'):
    train_len = 2800
    test_len = 113
    if data_type=='train':
        print('train Homologous protein num is ',train_len)
    else:
        print('test Homologous protein num is ',test_len)
    cross_dock_names_path = '/home/tianye/MolPretrain/scripts/crossdock2020.json'
    cross_dock_dict = load_json(path=cross_dock_names_path)
    data_split = []
    for homologous_protein_dir, val in cross_dock_dict.items():
        data_split.append(homologous_protein_dir)
    soted_data_split = sorted(data_split)
    if data_type=='train':
        return soted_data_split[:train_len]
    elif data_type=='test':
        return soted_data_split[train_len:]
    
    
    
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_match_information(continuos=True):
    data_list = []
    for i in range(66):
        if continuos:# 最大连续子序列
            data_list += load_pickle(f'./out_file/cross_dock_data_infor/matching_infor_continuous/lig_protein_continuous_{i}_match_information.pkl')
        else:# 最大公共子序列
            data_list += load_pickle(f'./out_file/cross_dock_data_infor/lig_protein_{i}_match_information.pkl')
            
    return data_list

def get_path(match_information):
    # 从匹配信息中获取 [pdb_in， lig_sdf, pdb_out，lig_pdb]
    # 如果lig_sdf不存在则使用lig_pdb作为输入
    cs2020_path = '/home/tianye/cs2020'
    lig_dir = '/userdata/xiaoqi/crossdock_lig10conformers'
    data_path = [None,None,None,None]
    data_path[0],data_path[2] = match_information['pdb_in_out_path']
    data_path[3] =  os.path.join(cs2020_path, match_information['cs_name'], match_information['lig_name']+'.pdb')
    sdf_path = os.path.join(lig_dir, match_information['lig_name']+'.sdf')
    if os.path.exists(sdf_path):
        data_path[1] = sdf_path
    return data_path
        

def filter_data(data_type='train',continuos=True):
    protein_match_infor_list = load_match_information(continuos=continuos) # 读取匹配信息
    print('pair data number is ',len(protein_match_infor_list))
    # 过滤训练集
    data_dir = load_data_type(data_type=data_type)
    data_path = [] 
    # 对60w数据排序
    protein_match_infor_list = sorted( protein_match_infor_list ,key=lambda x: x['cs_name'])
    homo_data = []
    for match_information in protein_match_infor_list:
        if match_information['cs_name'] in data_dir:# 过滤训练集/测试集
            if match_information['usable']==True: #如果两个蛋白完全匹配
                data_path.append(get_path(match_information))
                homo_data.append( match_information['cs_name'] )
            else: # 不完全匹配
                if match_information['reason'] == 'chains_num_is_not_match': # 肽链不一样多
                    pass
                elif match_information['reason'] == 'chains_length_is_not_match_or_chains_seqence_is_not_match': # 肽链序列不匹配
                    same_rate = np.average(match_information['same_rate'])
                    if same_rate>0.9:# 最大公共子序列的占据目标序列的比值(T值)
                        data_path.append(get_path(match_information))
                        homo_data.append( match_information['cs_name'] )
          
    print(f'{data_type} data size ',len(data_path))
    print(f'{data_type} data homo num is ',len(set(homo_data)))
    return data_path

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def filter_dataABCD(DATA_TYPE='train',continuos=True):
    dict_data = load_json('./split_crossdock_dataset.json')
    train,valid,test_A,test_B,test_C,test_D = dict_data['train'],dict_data['valid'],dict_data['testA'],dict_data['testB'],dict_data['testC'],dict_data['testD']
    
    if DATA_TYPE=='train':data_dir = train
    if DATA_TYPE=='valid':data_dir = valid
    if DATA_TYPE=='testA':data_dir = test_A
    if DATA_TYPE=='testB':data_dir = test_C
    if DATA_TYPE=='testC':data_dir = test_B
    if DATA_TYPE=='testD':data_dir = test_D
    
    protein_match_infor_list = load_match_information(continuos=continuos) # 读取匹配信息
    print('pair data number is ',len(protein_match_infor_list))

    data_path = [] 
    # 对60w数据排序
    protein_match_infor_list = sorted( protein_match_infor_list ,key=lambda x: x['lig_name'])
    homo_data = []
    lig_name_list = []
    data_test_path = {}
    
    for match_information in protein_match_infor_list:
        if match_information['lig_name'] in data_dir:# 过滤数据集
            if match_information['usable']==True: #如果两个蛋白完全匹配
                path_tuple = get_path(match_information)
                if match_information['lig_name'] not in data_test_path.keys():
                    data_test_path[match_information['lig_name']] = path_tuple
                data_path.append(path_tuple)
                homo_data.append( match_information['cs_name'] )
                lig_name_list.append(match_information['lig_name'])
            else: 
                if match_information['reason'] == 'chains_length_is_not_match_or_chains_seqence_is_not_match': # 肽链序列不匹配
                    same_rate = np.average(match_information['same_rate'])
                    if same_rate>0.9:# 最大公共子序列的占据目标序列的比值(T值)
                        path_tuple = get_path(match_information)
                        if match_information['lig_name'] not in data_test_path.keys():
                            data_test_path[match_information['lig_name']] = path_tuple
                        data_path.append(path_tuple)
                        homo_data.append( match_information['cs_name'] )
                        lig_name_list.append(match_information['lig_name'])
                        
    if 'test' in DATA_TYPE: data_path = list( data_test_path.values())
    if 'valid' in DATA_TYPE: data_path = list( data_test_path.values())
    
    print(f'{DATA_TYPE} data size ',len(data_path))
    print(f'{DATA_TYPE} data homo num is ',len(set(homo_data)))
    print(f'{DATA_TYPE} data pdbid num is ',len(set(lig_name_list)))
    return data_path



def get_pdbid_from_crossdock(protein_match_infor_list):
    pdbid = []
    for match_information in protein_match_infor_list:
        pdbid.append( match_information['lig_name'][:4] )
    return pdbid


def num_fragments_lig(mol):
    # 获取分子中的所有键
    bonds = mol.GetBonds()

    # 构建键的连接关系图
    bond_dict = {}
    for bond in bonds:
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        if atom1_idx not in bond_dict:
            bond_dict[atom1_idx] = []
        bond_dict[atom1_idx].append(atom2_idx)
        if atom2_idx not in bond_dict:
            bond_dict[atom2_idx] = []
        bond_dict[atom2_idx].append(atom1_idx)

    # 判断分子的连通部分个数
    visited_atoms = set()
    num_fragments = 0
    for atom_idx in bond_dict:
        if atom_idx not in visited_atoms:
            num_fragments += 1
            stack = [atom_idx]
            while stack:
                current_atom = stack.pop()
                visited_atoms.add(current_atom)
                connected_atoms = bond_dict[current_atom]
                for connected_atom in connected_atoms:
                    if connected_atom not in visited_atoms:
                        stack.append(connected_atom)
    return num_fragments
    
# 过滤
# 取出训练集
# [训练集, T<0.9, 肽链条数不一致, 输入配体的sdf不存在/输出配体无法读取]
# [(pdb_in, lig_in, pdb_out, lig_out)] 50w
def get_list_use_idx(idx_list, val_list):
    sorted_val = []
    for i in idx_list:
        sorted_val.append(val_list[i])
    return sorted_val

def get_dataset_and_condition():
    test_A = [] # 返回测试集的pdb_id
    origin_protein = [] # 该测试集的同源蛋白
    smiles_list = [] # 该测试集合的smiles
    coreset_test_pdbid = load_json('./unimol_dataset.json')['test'] # coreset的测试集
    cross_dock_dataset = load_json('./cross_dock_smi.json') # crossdock数据集
    lig_name_list = [lig_name[:4] for lig_name, (cs_name, smiles) in cross_dock_dataset.items()] # crossdock的所有pdbid
    test_pdbid =  set(lig_name_list)&set(coreset_test_pdbid) # 172个交叉的pdbid # 但由于那三个小字母,有185个配体,177个不重合的smiles,49个同源蛋白
    
    for pdbid in test_pdbid:
        for lig_name in cross_dock_dataset.keys():
            if pdbid in lig_name:
                test_A.append(lig_name)
                origin_protein.append(cross_dock_dataset[lig_name][0])
                smiles_list.append(cross_dock_dataset[lig_name][1])
    counter = Counter(smiles_list)
    smi_count = []
    for smi in smiles_list:
        smi_count.append(counter[smi]) # 个数
    smi_count_idx = sorted(range(len(smi_count)), key=lambda i: smi_count[i], reverse=True)# 把出现次数多的smiles排在前面
    
    core_set_test = get_list_use_idx(idx_list=smi_count_idx, val_list=test_A)
    origin_protein = get_list_use_idx(idx_list=smi_count_idx, val_list=origin_protein)
    smiles_list = get_list_use_idx(idx_list=smi_count_idx, val_list=smiles_list)
    
    test_A, testA_origin_protein, testA_smiles_list = core_set_test[:100], origin_protein[:100], smiles_list[:100]
    test_B, testB_origin_protein, testB_smiles_list = core_set_test[100:], origin_protein[100:], smiles_list[100:]
    
    # testA 蛋白和配体 F
    # testB 蛋白F, 配体 T
    # testC 配体F, 蛋白 T
    # testD 蛋白和配体 T
    for lig_name, (cs_name, smi) in cross_dock_dataset.items():
        if lig_name not in set(test_A+test_B)   :# 不能再选选过的
            if (smi not in testA_smiles_list) and ('.' not in smi): # 不能选testA里面的smiles,因为这些要求没见过, 而testB要求见过这些配体
                test_B.append(lig_name)
                testB_origin_protein.append(cs_name)
                testB_smiles_list.append(smi)
        if len(test_B)==100:
            break
        
    test_C, testC_origin_protein, testC_smiles_list = [], [], []
    for lig_name, (cs_name, smi) in cross_dock_dataset.items():
        if lig_name not in set(test_A+test_B)   :# 不能再选选过的
            if cs_name not in set(testA_origin_protein+testB_origin_protein) : #不能选testA+testB里面的pocket,因为这些要求没见过,而testC要求见过这些蛋白
                if (smi not in set(testB_smiles_list)) and ('.' not in smi): #不能选testB里面的smi, 因为这些要求见过, 而testC要求没见过配体
                    test_C.append(lig_name)
                    testC_origin_protein.append(cs_name)
                    testC_smiles_list.append(smi)
        if len(test_C)==100:
            break
    
    valid, valid_origin_protein, valid_smiles_list = [], [], []
    train, train_origin_protein, train_smiles_list = [], [], []
    for lig_name, (cs_name, smi) in cross_dock_dataset.items():
        if lig_name not in set(test_A+test_B+test_C)   :# 不能再选选过的
            #不能选testA+testB里面的pocket,因为这些要求没见过,而testD要求见过这些蛋白
            #不能选testA+C里面的smi, 因为这些要求没见过, 而testD要求见过这些配体
            if (cs_name not in set(testA_origin_protein+testB_origin_protein)) and (smi not in set(testA_smiles_list+testC_smiles_list)) and ('.' not in smi): 
                train.append(lig_name)
                train_origin_protein.append(cs_name)
                train_smiles_list.append(smi)
            elif ('.' not in smi):
                valid.append(lig_name)
                valid_origin_protein.append(cs_name)
                valid_smiles_list.append(smi)
    
    # 将训练集进行shuffled
    zipped = list(zip(train, train_origin_protein, train_smiles_list))
    random.shuffle(zipped)
    train, train_origin_protein, train_smiles_list = zip(*zipped)
    
    test_D, testD_origin_protein, testD_smiles_list = train[:100], train_origin_protein[:100], train_smiles_list[:100]
    train, train_origin_protein, train_smiles_list = train[100:], train_origin_protein[100:], train_smiles_list[100:]
    
    valid_nums = len(train)//10
    if valid_nums > len(valid): # 验证集有2855
        gap = valid_nums - len(valid)
        valid, valid_origin_protein, valid_smiles_list = valid+train[:gap], valid_origin_protein+train_origin_protein[:gap], valid_smiles_list+train_smiles_list[:gap]
        train, train_origin_protein, train_smiles_list = train[gap:], train_origin_protein[gap:], train_smiles_list[gap:]
    json_dict = {'train':train,'valid':valid,'testA':test_A,'testB':test_B,'testC':test_C,'testD':test_D}
    with open('./split_crossdock_dataset.json') as f:
        json.dump(json_dict, f)
        
    return train, valid, test_A, test_B, test_C, test_D


def get_paths_list(data_type='pretrain_train'):
    
    esmfold_match_json = './out_file/esmfold_match/esm_fold_match.json'  
    mol_match_json = './out_file/5000w_mol_match/mol_match.json'
    dataset_path = []
    
    with open(esmfold_match_json,'r') as f:
        esmfold_match_dict = json.load(f)
    with open(mol_match_json,'r') as f:
        mol_match_dict = json.load(f)
    pdbids_set = set(esmfold_match_dict.keys()) & set(mol_match_dict.keys())
    
    # 对蛋白pdbid循环
    for pdbid,esm_fold_match in esmfold_match_dict.items():
        if pdbid in pdbids_set:
            if esm_fold_match["usable"]:
                pdb_in, pdb_out = esm_fold_match["pdb_in_out_path"]
            elif esm_fold_match["reason"] == "chains_length_is_not_match_or_chains_seqence_is_not_match" \
                and np.average( esm_fold_match["same_rate"])>0.9:
                    pdb_in, pdb_out = esm_fold_match["pdb_in_out_path"]
                    
            # 对小分子循环
            for name,mol_match_information in mol_match_dict[pdbid].items():
                if mol_match_information["usable"]:
                    lig_in, lig_out = mol_match_information['sdf_in'],mol_match_information['sdf_out']
                    dataset_path.append([pdb_in, lig_in, pdb_out, lig_out])
    
    num = len(dataset_path)
    train_num = int(num*0.97)
    if data_type=='pretrain_train':
        dataset_path = dataset_path[:train_num]
        print('dataset size',len(dataset_path))
        return dataset_path
    elif data_type=='pretrain_valid':
        dataset_path = dataset_path[train_num:]
        print('dataset size',len(dataset_path))
        return dataset_path

def full_id_to_idx(full_id, res_name, offset=0):
    return f'{full_id[1]}-{full_id[2]}-{full_id[3][0]}{full_id[3][1]}{full_id[3][2]}-{res_name}'


class ResidueSelector(Bio.PDB.Select):
    def __init__(self, positions):
        self.positions = set(positions)

    def accept_model(self, model):
        return True

    def accept_residue(self, residue):
        flag = full_id_to_idx(residue.get_full_id(), residue.get_resname()) in self.positions
        return flag
    
def read_txt_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行末尾的换行符并添加到列表中
            lines.append(line.rstrip('\n'))
    return lines


def get_path_list(data_type):
    data_path = []
    
    file_path = f'./example_data/{data_type}'
    if data_type=='coreset_esmfold':
        coreset_pdbid_txt = file_path + '/pdbid.txt'
        with open(coreset_pdbid_txt, 'r', encoding='utf8') as f:
            pdbids = [line.strip() for line in f.readlines()]
        for pdbid in pdbids:
            esmfold_predict_protein_path = file_path + f'/esmfold_predict_protein/{pdbid}_protein.pdb'
            ground_ligand_path = file_path + f'/ground_ligand/{pdbid}'
            ground_protein_path = file_path + f'/ground_protein/{pdbid}_protein_processed.pdb'
            # pdb_in, lig_in, pdb_out, lig_out
            data_path.append([esmfold_predict_protein_path, ground_ligand_path, ground_protein_path, ground_ligand_path])
            
    elif data_type=='posebusters_esmfold':
        pdbids = read_txt_lines('./example_data/posebusters_esmfold/posebusters.txt')
        for pdbid in pdbids:
            esmfold_predict_protein_path = file_path + f'/esmfold_predict_protein/{pdbid}_protein.pdb'
            ground_ligand_path = file_path + f'/ground_ligand/{pdbid}_ligand.sdf'
            ground_protein_path = file_path + f'/ground_protein/{pdbid}_protein.pdb'
            init_ligand_path = file_path + f'/init_conformer_ligand/{pdbid}_ligand_start_conf.sdf'
            # pdb_in, lig_in, pdb_out, lig_out
            data_path.append([esmfold_predict_protein_path, init_ligand_path, ground_protein_path, ground_ligand_path])
    
    elif 'posebusters_esmfold_prepared' in data_type:
        file_path = f'./example_data/posebusters_esmfold'
        pdbids = ['_'.join(name.split('_')[:2]) for name in os.listdir( file_path + f'/esmfold_prepared/' )]
        for pdbid in pdbids:
            esmfold_predict_protein_path = file_path + f'/esmfold_prepared/{pdbid}_p.pdb'
            ground_ligand_path = file_path + f'/ground_ligand/{pdbid}_ligand.sdf'
            ground_protein_path = file_path + f'/ground_protein_prepared/{pdbid}_protein.pdb'
            # pdb_in, lig_in, pdb_out, lig_out
            data_path.append([esmfold_predict_protein_path, ground_ligand_path, ground_protein_path, ground_ligand_path])
    elif data_type=='glun1_af2':
        esmfold_predict_protein_path = '/home/xiaoqi/FlexiDock/example_data/GluN1/alphafold2_aligned.pdb'
        ground_ligand_path = '/home/xiaoqi/FlexiDock/example_data/GluN1/5UN1_ligand.sdf'
        ground_protein_path = '/home/xiaoqi/FlexiDock/example_data/GluN1/alphafold2_aligned.pdb'
        data_path.append([esmfold_predict_protein_path, ground_ligand_path, ground_protein_path, ground_ligand_path])
        return data_path
    elif data_type=='glun1_5un1':
        esmfold_predict_protein_path = '/home/xiaoqi/FlexiDock/example_data/GluN1/5UN1_protein.pdb'
        ground_ligand_path = '/home/xiaoqi/FlexiDock/example_data/GluN1/5UN1_ligand.sdf'
        ground_protein_path = '/home/xiaoqi/FlexiDock/example_data/GluN1/5UN1_protein.pdb'
        data_path.append([esmfold_predict_protein_path, ground_ligand_path, ground_protein_path, ground_ligand_path])
        return data_path
    
    else:
        pdbids = os.listdir(f'./example_data/{data_type}/ground_ligand')
        for pdbid in pdbids:
            esmfold_predict_protein_path = file_path + f'/esmfold_predict_protein/{pdbid}_protein.pdb'
            ground_ligand_path = file_path + f'/ground_ligand/{pdbid}'
            ground_protein_path = file_path + f'/ground_protein/{pdbid}_protein_processed.pdb'
            # pdb_in, lig_in, pdb_out, lig_out
            data_path.append([esmfold_predict_protein_path, ground_ligand_path, ground_protein_path, ground_ligand_path])
    return data_path


class CrossDockDataSet(Dataset):
    def __init__(self, root, transform=None, data_type='croeset_esmfold',continuos=True,pretrain_method='pretrain_method1',max_align_rmsd=20,cut_r=10,
                limit_complexes=0,start_from_pdbid=None,save_pdb=False,data_path=None,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False, data_test=True):

        super(CrossDockDataSet, self).__init__(root, transform)
        self.max_align_rmsd = max_align_rmsd
        self.cut_r = cut_r
        self.continuos = continuos
        self.data_type = data_type

        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = get_path_list(data_type=data_type)
        
        print(data_type, len(self.data_path))
        
        self.pretrain_method = pretrain_method
        self.save_pdb = save_pdb
        self.max_lig_size = max_lig_size # 配体原子的最大尺寸
        self.start_from_pdbid = start_from_pdbid
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.data_test = data_test
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        
    def len(self):
        return len(self.data_path)

    def get(self, idx):
        complex_graph, lig = None,None
        
        try:
            complex_graph, lig = self.preprocessing(idx)
        except Exception as e:
            print(e)
            
        if complex_graph is None:
            return None
        if self.require_ligand:
            complex_graph.mol = lig
        
        return complex_graph

    def preprocessing(self, idx):
        pdb_in, lig_in, pdb_out, lig_out = self.data_path[idx]
        if lig_in is None:
            lig_in = lig_out
        
        graph, lig = self.get_complex(pdb_in, lig_in, pdb_out, lig_out)
        return graph, lig
        
    def assert_tor(self, edge_index, mask_rotate):

        for idx_edge, e in enumerate(edge_index.cpu().numpy()):
            u, v = e[0], e[1]
            # check if need to reverse the edge, v should be connected to the part that gets rotated
            if  not ( (not mask_rotate[idx_edge, u]) and mask_rotate[idx_edge, v]):
                raise ValueError('torsion assert error')
    
    def get_structure(self, path=''):
        # 该函数读取蛋白质
        warnings.filterwarnings("ignore")
        biopython_parser = PDBParser()
        structure = biopython_parser.get_structure('random_id', path)
        structure = structure[0]
        return structure
    
    def get_name(self, pdb_in, pdb_out, lig_out):
        # 该函数获取图数据的名称
        n1 = pdb_in.split('/')[5].split('_')[0]
        n2 = pdb_out.split('/')[5].split('_')[0]
        n3 = '_'.join( lig_out.split('/')[5].split('_')[:2] )
        name = '_'.join([n1,n2,n3])
        return name
    
    def longest_common_subsequence_seq_idx(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 构造最长公共子序列
        lcs_length = dp[m][n]
        lcs = [''] * lcs_length
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs[lcs_length - 1] = s1[i - 1]
                i -= 1
                j -= 1
                lcs_length -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        # 寻找最长公共子序列在原始序列中的所有下标
        lcs_indices = []
        lcs_len = len(lcs)
        idx1 = 0
        idx2 = 0
        for idx in range(lcs_len):
            while s1[idx1] != lcs[idx]:
                idx1 += 1
            while s2[idx2] != lcs[idx]:
                idx2 += 1
            lcs_indices.append((idx1, idx2))
            idx1 += 1
            idx2 += 1
            
        s1_same_idx, s2_same_idx = zip(*lcs_indices)
        return ''.join(lcs), s1_same_idx, s2_same_idx
    

    def longest_continuos_subsequence_seq_idx(self, s1, s2):
        # 该函数返回最大连续子序列,以及在s1和s2的起始终止下标
        m = len(s1)
        n = len(s2)

        # 创建一个二维数组来保存最长连续子序列的长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        max_length = 0  # 最长连续子序列的长度
        end_index = 0  # 最长连续子序列的结束索引

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_index = i - 1  # 更新最长连续子序列的结束索引
                else:
                    dp[i][j] = 0

        # 根据最长连续子序列的长度和结束索引，提取出最长连续子序列
        lcs = s1[end_index - max_length + 1: end_index + 1]

        # 寻找最长公共子序列在原始序列中的起始和终止下标
        start_idx1 = s1.find(lcs)
        end_idx1 = start_idx1 + len(lcs) - 1
        
        start_idx2 = s2.find(lcs)
        end_idx2 = start_idx2 + len(lcs) - 1
    
        return lcs, (start_idx1, end_idx1), (start_idx2, end_idx2)
    
    
    def from_chain_list_to_sequences_list(self, chain_list):
        # 该函数去水，去掉20种非人体氨基酸
        sequences = []
        res_seq = []
        ids = []
        for i, chain in enumerate(chain_list):
            seq = ''
            chain_seq = []
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    continue
                if residue.get_resname() not in residue_atoms.keys():
                    continue
                seq += three_to_one[residue.get_resname()] # 得到20种氨基酸的序列
                chain_seq.append(residue)
            if len(seq)!=0:
                sequences.append(seq)
                res_seq.append(chain_seq)
        return sequences,res_seq
    
    
    def get_same_chain_idx(self, sequence_in, sequence_out):
        for i in range(len(sequence_out)):
            if sequence_in[0]==sequence_out[i]:
                return i
            
        return None
                
    def get_chain_res_list(self, in_structure, out_structure):
        out_chain_res_list, in_chain_res_list = [], []
        
        chain_list_a, chain_list_b = Selection.unfold_entities(in_structure, 'C'),Selection.unfold_entities(out_structure, 'C')
        # if len(chain_list_a) != len(chain_list_b):
        #     print('chain number are not same, in /out ',len(chain_list_a) , len(chain_list_b) )
        
        # 去掉H2O,去掉非20种人体氨基酸
        sequence_in, chain_res_list_in = self.from_chain_list_to_sequences_list(chain_list_a)
        sequence_out, chain_res_list_out = self.from_chain_list_to_sequences_list(chain_list_b)
        
        # if len(sequence_in)==1 and len(sequence_out)>1:
        #     same_chain_idx = self.get_same_chain_idx(sequence_in, sequence_out)
        #     if same_chain_idx is not None:
        #         print('find same chain successful')
        #         sequence_out = [sequence_out[same_chain_idx]]
        #         chain_res_list_out = [chain_res_list_out[same_chain_idx]]
        
        
        # 对所有的肽链求出最大连续公共子序列
        for s_in, s_out, chain_in, chain_out in zip(sequence_in, sequence_out, chain_res_list_in, chain_res_list_out):
            if self.continuos:# 最大连续子序列
                max_same_seq, (start_idx1, end_idx1), (start_idx2, end_idx2) = self.longest_continuos_subsequence_seq_idx(s1=s_in,s2=s_out)
                in_chain_res_list.append(chain_in[start_idx1:end_idx1+1])
                out_chain_res_list.append(chain_out[start_idx2:end_idx2+1])
                if  len(max_same_seq)/max(len(s_in),len(s_out))<0.9:
                    return None,None
            else: # 最大公共子序列
                max_same_seq, s1_same_idx, s2_same_idx = self.longest_common_subsequence_seq_idx(s1=s_in,s2=s_out)
                in_chain_res_list.append([chain_in[idx] for idx in s1_same_idx])
                out_chain_res_list.append([chain_out[idx] for idx in s2_same_idx])
        
        return in_chain_res_list, out_chain_res_list
    
    
    def get_dist(self, RES_atoms_pos, lig_atoms_pos):
        # [N,3] [M,3]
        dist_matrix = cdist(RES_atoms_pos, lig_atoms_pos)
        return dist_matrix
        
    def cut_pocket_from_pocket_cent(self, out_chain_res_list, lig_out_mol, max_r=[10,15,25]):
        pocket_idx = []
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        pocket_cent = np.average(lig_coords, axis=0, keepdims=True)
        noise_delta = np.random.normal(0, 5, size=pocket_cent.shape)
        pocket_cent = pocket_cent + noise_delta # 口袋中心
        max_r = random.choice(max_r) # 最大距离随机抽一个
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A')] )
                dist_mat = self.get_dist(res_atoms_coord, pocket_cent)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        return pocket_idx
    
    def cut_pocket_from_residues_ca_distance(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = []
        # 随机选取几个配体原子，求出坐标均值,以片段去求
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        atoms_num = random.choice([len(lig_coords)//i  for i in [3,4,5]] ) # 获取配体关键原子个数
        start_idx = np.random.randint(0, len(lig_coords)-atoms_num)
        random_frag_lig_cent = np.average( lig_coords[start_idx:start_idx+atoms_num], axis=0, keepdims=True)
        
        # 求出关键残基ca坐标
        ca_coord_list = []
        distance_list_of_frag_cent_dist_with_ca = []
        for res_list in out_chain_res_list:
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                ca_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') if atom.name=='CA'] )
                dist_mat = self.get_dist(ca_coord, random_frag_lig_cent)
                min_dist = np.min(dist_mat)
                ca_coord_list.append(ca_coord)
                distance_list_of_frag_cent_dist_with_ca.append(min_dist)
        min_dist_idx = np.argmin(distance_list_of_frag_cent_dist_with_ca)
        key_ca_coord = ca_coord_list[min_dist_idx]
        
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [ list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') if atom.name=='CA' ] )
                dist_mat = self.get_dist(res_atoms_coord, key_ca_coord)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        
        return pocket_idx
    
    def cut_pocket_from_residues_atom_distance(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = []
        # 随机选取几个配体原子，求出坐标均值,以片段去求
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        atoms_num = random.choice([len(lig_coords)//i  for i in [3,4,5]] ) # 获取配体关键原子个数
        start_idx = np.random.randint(0, len(lig_coords)-atoms_num)
        random_frag_lig_cent = np.average( lig_coords[start_idx:start_idx+atoms_num], axis=0, keepdims=True)
        
        # 求出关键原子坐标
        res_min_coord_list = []
        distance_min = []
        for res_list in out_chain_res_list:
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atom_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') ] )
                dist_mat = self.get_dist(res_atom_coord, random_frag_lig_cent)
                min_dist = np.min(dist_mat)
                res_min_coord_list.append(res_atoms_coord[np.argmin(dist_mat.reshape((-1)))])
                distance_min.append(min_dist)
        min_dist_idx = np.argmin(distance_min)
        key_atom_coord = res_min_coord_list[min_dist_idx]
        
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [ list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') ] )
                dist_mat = self.get_dist(res_atoms_coord, key_atom_coord)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        
        return pocket_idx
    
        
    def cut_pocket(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = []
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A')] )
                dist_mat = self.get_dist(res_atoms_coord, lig_coords)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        return pocket_idx

    def from_pocket_idx_to_res_list(self, chain_res_list, pocket_chain_idx_list):
        # 根据口袋残基的idx将口袋残基的列表提取出来
        pocket_res = []
        for chain_res,chain_idx_list in zip(chain_res_list,pocket_chain_idx_list):
            for idx in chain_idx_list:
                pocket_res.append(chain_res[idx])
        return pocket_res
    
    def get_ca_coord(self, res):
        for atom in Selection.unfold_entities(res, 'A'):
            if atom.name=='CA':
                return list(atom.get_vector())
        return None
                
        
    def align_pocket(self, out_pocket_res, in_pocket_res):
        # 对所有存在CA坐标的残基进行提取坐标
        in_coord = []
        out_coord = []
        for res_in, res_out in zip(in_pocket_res, out_pocket_res):
            CA_in = self.get_ca_coord(res_in)
            CA_out = self.get_ca_coord(res_out)
            if CA_in is not None and CA_out is not None:
                in_coord.append(CA_in)
                out_coord.append(CA_out)
            
        in_coord , out_coord = torch.from_numpy(np.array(in_coord)).float(), torch.from_numpy(np.array(out_coord)).float()
        if in_coord.shape[0] < 5:
            print(f'in_coord.shape[0] = {in_coord.shape[0]}')
            return None,None,None
        R, t = rigid_transform_Kabsch_3D_torch(in_coord.T, out_coord.T)
        aligned_in_coord = in_coord @ R.T + t.T

        rmsd, R_pocket, T_pocket = self.rmsd_test(aligned_in_coord, out_coord), R, t
        return rmsd, R_pocket, T_pocket
    

    def extract_pocket_structure(self, out_pocket_res):
        feature_list = []
        chain_c_alpha_coords, chain_n_coords, chain_c_coords, chain_c_beta_coords, chain_o_coords = [],[],[],[],[]
        for residue in out_pocket_res:
            feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
            c_alpha, n, c, c_beta, o = None, None, None, None, None
            for atom in residue:# 残基(氨基酸)对原子进行循环
                if atom.name == 'CA': # 残基alpha碳原子
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N': # 残基N原子
                    n = list(atom.get_vector())
                if atom.name == 'C': # 残基C原子
                    c = list(atom.get_vector())
                if atom.name == 'CB':# 残基beta碳原子
                    c_beta = list(atom.get_vector())
                if atom.name == 'O':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    o = list(atom.get_vector())
                if atom.name == 'O' and residue.resname == 'GLY':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    c_beta = list(atom.get_vector())
            none_res_atom_pos = [0.0, 0.0, 0.0]     
            # 五个原子都不为None
            if c_alpha != None and n != None and c != None and c_beta != None and  o != None:
                chain_c_alpha_coords.append(c_alpha) # 当前循环已经把 alpha c原子找到的时候就把它添加到alpha碳原子
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_c_beta_coords.append(c_beta)
                chain_o_coords.append(o)
                
    
        c_alpha_coords = torch.from_numpy(np.array(chain_c_alpha_coords)).float()
        n_coords = torch.from_numpy(np.array(chain_n_coords)).float()
        c_coords = torch.from_numpy(np.array(chain_c_coords)).float()
        c_beta_coords = torch.from_numpy(np.array(chain_c_beta_coords)).float()
        o_coords = torch.from_numpy(np.array(chain_o_coords)).float()
        feature_res = torch.from_numpy(np.array(feature_list)).float()
        return feature_res, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords
                        
    def out_feature_to_graph(self, complex_graph, feature_res, in_c_alpha_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, lm_embeddings=None):
        cutoff = self.receptor_radius #15
        max_neighbor = self.c_alpha_max_neighbors #24
        n_rel_pos = n_coords - c_alpha_coords
        c_rel_pos = c_coords - c_alpha_coords
        c_beta_rel_pos = c_beta_coords - c_alpha_coords
        o_rel_pos = o_coords - c_alpha_coords
        
        num_residues = len(c_alpha_coords)
        if num_residues <= 1:
            raise ValueError(f"rec contains only 1 residue!")
        
        # Build the k-NN graph
        import scipy.spatial as spa
        distances = spa.distance.cdist(in_c_alpha_coords, in_c_alpha_coords) #氨基酸alpha C的距离矩阵    该函数最
        src_list = []
        dst_list = []
        
        for i in range(num_residues):
            dst = list(np.where(distances[i, :] < cutoff)[0]) # 取出对于距离矩阵中小于15A的那些下标
            dst.remove(i) # 把对角元去掉,对角元是自己和自己的距离
            if max_neighbor != None and len(dst) > max_neighbor:
                dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1] # 距离小于15A的氨基酸可能有很多，甚至超过了24个，那么我们只取前24的下标
            if len(dst) == 0:
                dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
                print(f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                    f'So we connected it to the closest other c_alpha')
            assert i not in dst # 确保i没有在dst中，因为 是自己和自己
            src = [i] * len(dst)
            src_list.extend(src) # 最大邻接数是24, src_list [0]*24 + [1]*24 + [2]*22 + .....  
            dst_list.extend(dst) # 最大邻接数是24, dst_list [0距离小于15A的下标] + [1距离小于15A的下标].....  
    
        assert len(src_list) == len(dst_list)

        # node_feat = rec_residue_featurizer(rec)
        node_feat = feature_res  # [n_res, 1]
        side_chain_vecs = torch.cat([n_rel_pos.unsqueeze(1), c_rel_pos.unsqueeze(1), 
                                     c_beta_rel_pos.unsqueeze(1), o_rel_pos.unsqueeze(1)], axis=1)
        
        # 氨基酸为粒度， 氨基酸的embdeing和氨基酸种类 [N_res, 1280] [N_res, 1]
        complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1).float() if lm_embeddings is not None else node_feat
        # alpha_c的坐标
        complex_graph['receptor'].ref_pos = c_alpha_coords
        # [N_res,4,3]  先计算C原子相对alpha_C原子的位置差, N原子相对alpha_C原子的位置差，然后把它俩concat到一起得到
        complex_graph['receptor'].ref_side_chain_vecs = side_chain_vecs
        # [2，N<255*24] 最大邻接数是24, src_list [0]*24 + [1]*24 + [2]*22 + .....      最大邻接数是24, dst_list [0距离小于15A的下标] + [1距离小于15A的下标].....  
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))
        return complex_graph  
    
    def get_complete_information_idx(self, res_list):
        complete_information_idx = []
        for idx, residue in enumerate(res_list):
            c_alpha, n, c, c_beta, o = None, None, None, None, None
            for atom in residue:# 残基(氨基酸)对原子进行循环
                if atom.name == 'CA': # 残基alpha碳原子
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N': # 残基N原子
                    n = list(atom.get_vector())
                if atom.name == 'C': # 残基C原子
                    c = list(atom.get_vector())
                if atom.name == 'CB':# 残基beta碳原子
                    c_beta = list(atom.get_vector())
                if atom.name == 'O':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    o = list(atom.get_vector())
                if atom.name == 'O' and residue.resname == 'GLY':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    c_beta = list(atom.get_vector())
            if c_alpha != None and n != None and c != None and c_beta != None and  o != None:
                complete_information_idx.append(idx)
        return set( complete_information_idx )
    
    def cut_residues_of_miss_information(self, out_pocket_res, in_pocket_res):
        out_complete_information_idx = self.get_complete_information_idx(res_list=out_pocket_res)
        in_complete_information_idx = self.get_complete_information_idx(res_list=in_pocket_res)
        idx_list = sorted( list( out_complete_information_idx & in_complete_information_idx ) )
        out_res,in_res = [],[]
        for idx in idx_list:
            out_res.append(out_pocket_res[idx])
            in_res.append(in_pocket_res[idx])
        return out_res, in_res
    
    
    def in_feature_to_graph(self, complex_graph, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords, R_pocket, T_pocket):
        # 保存向量特征和输入CA的坐标,在保存之前需要做口袋对齐
        in_c_alpha_coords = in_c_alpha_coords @ R_pocket.T + T_pocket.T
        in_n_coords = in_n_coords @ R_pocket.T + T_pocket.T
        in_c_coords = in_c_coords @ R_pocket.T + T_pocket.T
        in_c_beta_coords = in_c_beta_coords @ R_pocket.T + T_pocket.T
        in_o_coords = in_o_coords @ R_pocket.T + T_pocket.T

        n_rel_pos = in_n_coords - in_c_alpha_coords
        c_rel_pos = in_c_coords- in_c_alpha_coords
        c_beta_rel_pos = in_c_beta_coords - in_c_alpha_coords
        o_rel_pos = in_o_coords - in_c_alpha_coords

        side_chain_vecs = torch.cat([n_rel_pos.unsqueeze(1), c_rel_pos.unsqueeze(1), 
                                     c_beta_rel_pos.unsqueeze(1), o_rel_pos.unsqueeze(1)], axis=1)
        
        complex_graph['receptor'].pos = in_c_alpha_coords
        complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
        return complex_graph
    
    def get_same_atoms_pos(self, res_in, res_out):
        # 获取非H原子
        in_atoms = {atom.name:list(atom.get_vector()) for atom in res_in if atom.name in residue_atoms[res_in.resname]}
        out_atoms = {atom.name:list(atom.get_vector()) for atom in res_out if atom.name in residue_atoms[res_out.resname]}
        common_keys = sorted( list( set(in_atoms.keys()) & set(out_atoms.keys())  ) )
        in_atoms_coord = []
        out_atoms_coord = []
        for atom_name in common_keys:
            in_atoms_coord.append(in_atoms[atom_name])
            out_atoms_coord.append(out_atoms[atom_name])
        return torch.from_numpy(np.array(in_atoms_coord)).float(), torch.from_numpy(np.array(out_atoms_coord)).float(), common_keys
            
    
    def atoms_pos_to_graph(self, complex_graph, out_pocket_res, in_pocket_res, R_pocket, T_pocket):
        # 将输入和输出的原子进行取交集,保存输入和输出的全原子坐标
        # 输入全原子坐标在保存之前需要做口袋对齐
        from .process_mols_pocket import get_pocket_infor
        res_num, max_atom_num = get_pocket_infor(res_list=out_pocket_res)
        res_atoms_mask = torch.zeros([res_num, max_atom_num], dtype=torch.bool)
        res_atoms_pos = torch.zeros([res_num, max_atom_num, 3])
        ref_res_atoms_pos = torch.zeros([res_num, max_atom_num, 3])
        ref_sorted_atom_names = []
        
        for idx, (res_in, res_out) in enumerate(zip(in_pocket_res,out_pocket_res)) :
            res_coord, ref_res_coord, res_sorted_atom_names = self.get_same_atoms_pos(res_in, res_out)
            res_atom_num = len(res_sorted_atom_names)
            res_atoms_pos[idx, :res_atom_num] = res_coord @ R_pocket.T + T_pocket.T # 输入进行对齐
            ref_res_atoms_pos[idx, :res_atom_num] = ref_res_coord
            res_atoms_mask[idx, :res_atom_num] = True
            ref_sorted_atom_names.append(res_sorted_atom_names)
    
        complex_graph['receptor'].res_atoms_mask = res_atoms_mask # [N_res,]
        complex_graph['receptor'].ref_res_atoms_pos = ref_res_atoms_pos.float() # [N_res, atoms, 3]
        complex_graph['receptor'].res_atoms_pos = res_atoms_pos.float() # [N_res, atoms, 3]
        complex_graph['receptor'].ref_sorted_atom_names = ref_sorted_atom_names 
        return complex_graph
    
    def set_coord_structure(self, pocket_R,pocket_T,structure,full_id_list):
        aligned_pocket_res_list = []
        for full_id in full_id_list:
            chain, res_id = full_id[2],full_id[3]
            res = structure[chain][res_id]
            for atom in res:
                old_coord = torch.from_numpy (atom.coord)
                new_coord = old_coord @ pocket_R.T + pocket_T.T
                atom.set_coord( new_coord.numpy().reshape((3,)) )
            aligned_pocket_res_list.append(res)
        return structure,aligned_pocket_res_list
    
    def get_6A_res_list_aligned(self,structure,full_id_list):
        aligned_pocket_res_list = []
        for full_id in full_id_list:
            chain, res_id = full_id[2],full_id[3]
            res = structure[chain][res_id]
            aligned_pocket_res_list.append(res)
        return aligned_pocket_res_list
    
    
    def save_pocket_out(self, structure, path, pocket_res_list):
        from Bio.PDB import PDBIO
        indices = []
        for res in pocket_res_list:
            indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
        selector = ResidueSelector(indices)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path,select=selector)
    
    def save_pocket_in_aligned(self, structure, path, pocket_res_list, pocket_R, pocket_T, pocket_res_list_6A=None, path_6A=None):
        from Bio.PDB import PDBIO
        indices = []
        full_id_list = []
        
        
        for res in pocket_res_list:
            indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
            full_id_list.append(res.full_id)
        
        
        structure,aligned_pocket_res_list = self.set_coord_structure(pocket_R,pocket_T,structure,full_id_list)
        selector = ResidueSelector(indices)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path,select=selector)
        if pocket_res_list_6A is not None:
            indices_6A = []
            full_id_list_6A = []
            for res in pocket_res_list_6A:
                indices_6A.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
                full_id_list_6A.append(res.full_id)
            aligned_pocket_res_list_6A = self.get_6A_res_list_aligned(structure=structure, full_id_list=full_id_list_6A)
            selector_6A = ResidueSelector(indices_6A)
            io.save(path_6A,select=selector_6A)
            return aligned_pocket_res_list, aligned_pocket_res_list_6A
        return aligned_pocket_res_list
    
    def save_predict_pocket(self, structure, path, pocket_res_list):
        # pocket_res_list 这个蛋白对象里存的是预测好的蛋白口袋
        from Bio.PDB import PDBIO
        indices = []
        for res in pocket_res_list:
            indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
        selector = ResidueSelector(indices)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path,select=selector)
    
    def get_mol_from_sdf_pdb_mol2(self,path_list):
        # path_list路径或者路径列表
        if type(path_list)==str:
            lig_out_path = path_list
            #获取配体
            if not os.path.exists(lig_out_path):
                print('load out ligand is not exist')
                return None
            if '.sdf' in lig_out_path:
                lig_out_mol = Chem.SDMolSupplier(lig_out_path, sanitize=True, removeHs=True)[0]
                return lig_out_mol
            elif '.pdb' in lig_out_path:
                lig_out_mol = Chem.MolFromPDBFile(lig_out_path, sanitize=True, removeHs=True)
                return lig_out_mol
        for path in path_list:
            lig_out_path = path
            #获取配体
            if not os.path.exists(lig_out_path):
                print('load out ligand is not exist')
                continue
            if '.sdf' in lig_out_path:
                lig_out_mol = Chem.SDMolSupplier(lig_out_path, sanitize=True, removeHs=True)[0]
                if lig_out_mol is not None:
                    return lig_out_mol
            elif '.pdb' in lig_out_path:
                lig_out_mol = Chem.MolFromPDBFile(lig_out_path, sanitize=True, removeHs=True)
                if lig_out_mol is not None:
                    return lig_out_mol
            elif '.mol2' in lig_out_path:
                lig_out_mol = Chem.MolFromMol2File(lig_out_path, sanitize=True, removeHs=True)
                if lig_out_mol is not None:
                    return lig_out_mol
        return None
    
    
    def get_complex(self, pdb_in_path, lig_in_path, pdb_out_path, lig_out_path):
        # 判断路径是目录还是文件
        if os.path.isdir(lig_in_path):
            lig_in_path = os.path.join(lig_in_path, random.choice(os.listdir(lig_in_path) )) # 随机选取一个输入
            
        # 将配体路径或者配体路径列表传入
        lig_out_path_list = lig_out_path
        if os.path.isdir(lig_out_path):
            lig_out_path_list = [os.path.join(lig_out_path, name) for name in os.listdir(lig_out_path) if '.sdf' in name or '.mol2' in name]
        if self.pretrain_method=='pretrain_method1':
            lig_in_path = copy.deepcopy(lig_out_path)
        
        if self.data_type in ['train', 'testA','testB','testC','testD','valid']:
            name = self.get_name(pdb_in_path, pdb_out_path, lig_out_path) # crossdock pdb_in pdb_out lig_out
        elif 'pretrain' in self.data_type:
            name = lig_out_path.split('/')[-1][:4]
        name = pdb_out_path.split('/')[-1][:4]
        complex_graph = HeteroData()# 定义异构图
        complex_graph['name'] = name
        if 'posebusters_esmfold' in self.data_type:
            name = '_'.join( pdb_out_path.split('/')[-1].split('_')[:2])
            complex_graph['name'] = name
        complex_graph.datatype = self.data_type
        #获取配体的特征
        
        lig_out_mol = self.get_mol_from_sdf_pdb_mol2(path_list=lig_out_path_list)
            
            
        if lig_out_mol is None: 
            print('load out ligand is None')
            return (None,None)
        complex_graph = get_lig_feature(mol_=lig_out_mol,complex_graph=complex_graph,keep_original=self.keep_original, remove_hs=self.remove_hs, sdf_path=lig_in_path, data_type=self.data_type, pretrain_method=self.pretrain_method) # 根据lig_out获取配体特征
        if complex_graph is None: # 配体特征获取失败
            return (None,None)
        #对于配体被分为多个部分的连通图,则跳过该类分子
        if not complex_graph.is_connected: 
            print('lig fragments num > 1')
            return (None,None)
        
        #获取蛋白特征
        out_structure = self.get_structure(pdb_out_path)
        in_structure = self.get_structure(pdb_in_path)
        
        # 去H2O,去非20类氨基酸,求出最大公共子序列的残基列表
        in_chain_res_list, out_chain_res_list = self.get_chain_res_list(in_structure, out_structure)
        
        if out_chain_res_list is None:
            print(f'{name} same sequence is None')
            return (None,None)
        
        # # 根据晶体割口袋
        # pocket_chain_idx_list = self.cut_pocket(out_chain_res_list, lig_out_mol, max_r=self.cut_r)
        # 根据晶体割口袋
        pocket_size = 10
        pocket_chain_idx_list = self.cut_pocket(out_chain_res_list, lig_out_mol, max_r=pocket_size)
        
        pocket_chain_idx_list_6A = self.cut_pocket(out_chain_res_list, lig_out_mol, max_r=6)
        in_pocket_res_6A = self.from_pocket_idx_to_res_list(chain_res_list=in_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list_6A)
        out_pocket_res = self.from_pocket_idx_to_res_list(chain_res_list=out_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list)
        in_pocket_res = self.from_pocket_idx_to_res_list(chain_res_list=in_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list)
        # 对于五个关键原子是否存在进行判断,并且对in和out进行取交集操作
        out_pocket_res, in_pocket_res = self.cut_residues_of_miss_information(out_pocket_res, in_pocket_res)
        
        if len(out_pocket_res)<=5:
            print('alpha-C num < 5')
            return (None,None)
        
        # 进行口袋对齐   
        ca_rmsd, R_pocket, T_pocket = self.align_pocket(out_pocket_res, in_pocket_res)
        
        if 'not_pocket_align' in self.data_type:
            R_pocket, T_pocket = torch.eye(3), torch.zeros_like(T_pocket)
        if self.save_pdb:
            complex_graph['receptor'].R_pocket, complex_graph['receptor'].T_pocket = copy.deepcopy(R_pocket), copy.deepcopy( T_pocket)
            complex_graph['receptor'].out_structure, complex_graph['receptor'].in_structure = copy.deepcopy (out_structure), copy.deepcopy( in_structure)
            complex_graph['receptor'].out_pocket_res  = out_pocket_res
        
            os.makedirs(f'./out_file/esmFold_{self.data_type}', exist_ok=True)
            os.makedirs(f'./out_file/esmFold_{self.data_type}/ground_pocket', exist_ok=True)
            os.makedirs(f'./out_file/esmFold_{self.data_type}/esm_pocket', exist_ok=True)
            self.save_pocket_out(structure=complex_graph['receptor'].out_structure, 
                                 path=f'./out_file/esmFold_{self.data_type}/ground_pocket/{name}_pocket_10A.pdb', 
                                 pocket_res_list=complex_graph['receptor'].out_pocket_res)
            complex_graph['receptor'].in_pocket_res, complex_graph['receptor'].in_pocket_res_6A = \
            self.save_pocket_in_aligned(structure=complex_graph['receptor'].in_structure, 
                                     path=f'./out_file/esmFold_{self.data_type}/esm_pocket/{name}_pocket_10A.pdb', 
                                    pocket_res_list=in_pocket_res, pocket_R=R_pocket, pocket_T=T_pocket, 
                                    pocket_res_list_6A=in_pocket_res_6A, path_6A=f'./out_file/esmFold_{self.data_type}/esm_pocket/{name}_pocket_6A.pdb'
                                        ) # 保存对齐后的esmFold预测的6A口袋和10A的口袋,
            
            # self.save_pocket_out(structure=complex_graph['receptor'].out_structure, 
            #             path=f'./out_file/esmFold_{self.data_type}/ground_pocket/{name}_pocket_12A.pdb', 
            #             pocket_res_list=complex_graph['receptor'].out_pocket_res)
            # complex_graph['receptor'].in_pocket_res, complex_graph['receptor'].in_pocket_res_6A = \
            # self.save_pocket_in_aligned(structure=complex_graph['receptor'].in_structure, 
            #                          path=f'./out_file/esmFold_{self.data_type}/esm_pocket/{name}_pocket_12A.pdb', 
            #                         pocket_res_list=in_pocket_res, pocket_R=R_pocket, pocket_T=T_pocket, 
            #                         pocket_res_list_6A=in_pocket_res_6A, path_6A=f'./out_file/esmFold_{self.data_type}/esm_pocket/{name}_pocket_6A.pdb'
            #                             ) # 保存对齐后的esmFold预测的6A口袋和10A的口袋,
             
            
            
        if ca_rmsd>30:
            print(f'aligned pocket rmsd > {self.max_align_rmsd}',ca_rmsd)
            return (None,None)
        
        complex_graph['receptor'].ca_rmsd = ca_rmsd.item()
        
        # 得到口袋残基的标量特征,口袋alphaC的坐标
        out_feature_res, out_c_alpha_coords, out_n_coords, out_c_coords, out_c_beta_coords, out_o_coords = self.extract_pocket_structure(out_pocket_res) 
        in_feature_res, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords = self.extract_pocket_structure(in_pocket_res)
        

        
        # 输出蛋白的信息————输出节点(CA)的坐标, 残基标量特征, 边的关系(根据输入的CA建立), 向量特征, 添加到图数据中
        complex_graph = self.out_feature_to_graph(complex_graph, in_feature_res, in_c_alpha_coords, out_c_alpha_coords, out_n_coords, out_c_coords, out_c_beta_coords, out_o_coords, lm_embeddings=None)
        # 输入蛋白的信息————输入节点(CA)的坐标, 向量特征, 全原子坐标(和全原子的符号做排序对齐),添加到图数据中. 所有的向量特征在传入图数据前都进行对齐
        complex_graph = self.in_feature_to_graph(complex_graph, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords, R_pocket, T_pocket)
        # 将输入和输出的原子取交集,方便之后求loss, 全原子坐标, 排序后的全原子的符号, 全原子mask, 添加到图数据        
        complex_graph = self.atoms_pos_to_graph(complex_graph, out_pocket_res, in_pocket_res, R_pocket, T_pocket)
        
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center# 去中心化,输入口袋
        complex_graph['receptor'].ref_pos -= protein_center # 去中心化,输出口袋
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask # [N_res,]
        complex_graph['receptor'].res_atoms_pos[res_atoms_mask] -= protein_center # esm-fold生成的蛋白坐标以晶体蛋白口袋中心为坐标系  [N_res, atoms, 3]
        complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask] -= protein_center  # 晶体蛋白坐标以蛋白口袋中心为坐标系
        complex_graph['receptor'].crossdock_rmsd = self.get_pocket_rmsd(complex_graph)
        complex_graph['ligand'].pos -= ligand_center # 随机生成的构象进行中心化
        complex_graph['ligand'].ref_pos -= protein_center # ground_truth的配体坐标以蛋白口袋中心为坐标系
        complex_graph.original_center = protein_center
            
        return complex_graph, lig_out_mol
    

    def get_pocket_rmsd(self, complex_graph):
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask
        res_atoms_pos = complex_graph['receptor'].res_atoms_pos[res_atoms_mask]
        ref_res_atoms_pos = complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask]
        return torch.sqrt(torch.sum((res_atoms_pos-ref_res_atoms_pos)**2,dim=1).mean()).item()
        
    def compute_RT(self, data):
        # 传入的配体坐标如果是一个列表,那么从列表中抽取一个坐标
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
            
        data.tr_score = self.get_lig_translation(data)
        lig_rmsd = self.rmsd_test(x=data['ligand'].pos, y=data['ligand'].ref_pos)
        sota_lig_rmsd = self.rmsd_test(x=self.modify_lig_conformer(x=data['ligand'].pos, tr_update=data.tr_score), y=data['ligand'].ref_pos)
        # print(f'{data.name} lig rmsd/sota_rmsd: ', lig_rmsd.item(), sota_lig_rmsd.item())
        data.res_rot_score, data.res_tr_score = self.get_res_rotation_vector(data)
        
        pocket_rmsd = self.rmsd_test(x=data['receptor'].res_atoms_pos, 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()

        print(f'{data.name} pocket rmsd: ', pocket_rmsd)
        res_rmsd = self.pocket_rmsd(x=data['receptor'].res_atoms_pos, y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        
        sota_pocket_rmsd = self.rmsd_test(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()
        
        aligned_res_rmsd = self.pocket_rmsd(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        deca_list = [(x1-x2)/x1 for x1,x2 in zip(res_rmsd,aligned_res_rmsd) if x1>x2]
        deca = np.average(deca_list)
        rate = len(deca_list)/len(res_rmsd)
        
        data['receptor'].deca = deca
        data['receptor'].rate = rate
        data['receptor'].sota_pocket_rmsd = sota_pocket_rmsd
        print('rate',rate, 'deca rate ', np.average(deca))
        
        # print(f'{data.name} esm-fold rmsd: ',data['receptor'].esm_rmsd)
        
        
        
        return data
        
    def get_lig_translation(self, data):
        assert data['ligand'].ref_pos.shape[0] == data['ligand'].pos.shape[0]
        lig_tr = data['ligand'].ref_pos - data['ligand'].pos
        return lig_tr # [lig_atoms,3]
    
    def get_res_translation(self, data):
        res_tr = data['receptor'].ref_pos - data['receptor'].pos
        return res_tr
    
    def get_res_rotation_vector(self, data):
        
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        ref_atoms_pos = data['receptor'].ref_res_atoms_pos    # ground truth 所有原子坐标
        pos_mask = data['receptor'].res_atoms_mask
        
        rot_mat, tr_vec =self.point_cloud_to_ror_matrix(ref_atoms_pos, atoms_pos, pos_mask=pos_mask) #[res, 3, 3]
        rot_vector = self.R_batch_to_axis_vec(R=rot_mat) #[res, 3]
        return rot_vector, tr_vec
        
    def point_cloud_to_ror_matrix(self, pos, ref, pos_mask, pos_ca=None, ref_ca=None):
        # 两堆点云计算旋转矩阵
        # pos [N,M,3] [res, atoms, 3]
        # ref [N,M,3]
        # pos_mask [N,M]
        # pos_ca, ref_ca [N,3]
        # N : number of examples
        # M : number of atoms
        # R,T maps local reference onto global pos
        if pos_mask is None:
            pos_mask = torch.ones(pos.shape[:2], device=pos.device)
        else:
            if pos_mask.shape[0] != pos.shape[0]:
                raise ValueError("pos_mask should have same number of rows as number of input vectors.")
            if pos_mask.shape[1] != pos.shape[1]:
                raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
            if pos_mask.ndim != 2:
                raise ValueError("pos_mask should be 2 dimensional.")
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        pos_c = pos - pos_mu
        ref_c = ref - ref_mu
    
        # Covariance matrix
        H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c.to(torch.float32))
        U, S, Vh = torch.linalg.svd(H)
        # Decide whether we need to correct rotation matrix to ensure right-handed coord system
        locs = torch.linalg.det(U @ Vh) < 0
        S[locs, -1] = -S[locs, -1]
        U[locs, :, -1] = -U[locs, :, -1]
        # Rotation matrix
        R = torch.einsum('bji,bkj->bik', Vh, U)
        T = pos_mu - torch.einsum('bij,bkj->bki', R, ref_mu)
        return R,T.squeeze(1)
        
    def R_batch_to_axis_vec(self, R):
        """
        旋转矩阵到旋转向量
        [N,3,3] >>> [N,3]
        """
        J = (R - R.permute([0,2,1])) / 2
        t = torch.cat( [ (-J[:, 1, 2]).unsqueeze(1) , (J[:, 0, 2]).unsqueeze(1),  (-J[:, 0, 1]).unsqueeze(1)], dim=1)
        t_norm = torch.norm(t, dim=1, keepdim=True)
        theta = torch.asin(t_norm)
        angle = theta / np.pi * 180
        r = t / t_norm
        v = r * theta
        return v
    
    def rmsd_test(self, x, y, mask=None):
        # x y [res, atoms, 3]
        if mask is not None:
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y
        # x y [atoms, 3]
        
        return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))
        
    def pocket_rmsd(self, x,y,mask):
        rmsd = []
        for i in range(len(mask)):
            rmsd.append( self.rmsd_test(x[i],y[i],mask[i]).item() )
        # max_idx = rmsd.index(max(rmsd))
        # print('res max/mean rmsd ', max(rmsd), self.rmsd_test(x,y,mask))
        return rmsd
    
    def modify_lig_conformer(self, x, tr_update):
        return x + tr_update
        
    def modify_pocket_conformer(self, data, tr_update, rot_update):
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
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
        
    def modify_pocket_conformer2(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = copy.deepcopy( data['receptor'].res_atoms_pos )   # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        max_atom_num = atoms_pos.shape[1]
        
        # 做旋转
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = vec_to_R(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        # atoms_pos[pos_mask] = torch.einsum('bij,bkj->bki',rot_mat[pos_mask].float(), atoms_pos[pos_mask].unsqueeze(1)).squeeze(1) 
        
        # 做平移
        # atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 平移旋转
        atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + tr_update.unsqueeze(1)
        
        return atoms_pos


class CrossInferDataSet(Dataset):
    def __init__(self, root, transform=None, data_type='train',continuos=True,pretrain_method='pretrain_method1',dekois=False,
                limit_complexes=0,start_from_pdbid=None,save_pdb=False,max_align_rmsd=20,cut_r=10,min_align_rmsd=0.2,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False, data_test=True):

        super(CrossInferDataSet, self).__init__(root, transform)
        self.dekois = dekois
        self.continuos = continuos
        self.data_type = data_type
        self.max_align_rmsd = max_align_rmsd
        self.min_align_rmsd = min_align_rmsd
        self.cut_pocket_r = cut_r
        if data_type in ['train', 'testA','testB','testC','testD','valid']:
            self.data_path = filter_dataABCD(DATA_TYPE=data_type, continuos=self.continuos) # [(pdb_in, lig_in, pdb_out, lig_out)]
        elif data_type=='crossdock_testAB':
            self.data_path = filter_dataABCD(DATA_TYPE='testA', continuos=self.continuos) + filter_dataABCD(DATA_TYPE='testB', continuos=self.continuos) # [(pdb_in, lig_in, pdb_out, lig_out)]
        elif data_type=='crossdock_testCD':
            self.data_path = filter_dataABCD(DATA_TYPE='testC', continuos=self.continuos) + filter_dataABCD(DATA_TYPE='testD', continuos=self.continuos)# [(pdb_in, lig_in, pdb_out, lig_out)]
        elif data_type=='pretrain_train':
            self.data_path = load_json('./out_file/pretrain_dataset.json')['train']
        elif data_type=='pretrain_valid':
            self.data_path = load_json('./out_file/pretrain_dataset.json')['valid']
        # elif 'coreset' in data_type:
        #     key = data_type.split('_')[-1]
        #     self.data_path = load_json('./out_file/core_set_half_flexi.json')[key]
        elif 'esmfold' in data_type: # esmfold_train
            key = data_type.split('_')[-1]
            self.data_path = load_json('./out_file/protein_flexi.json')[key]
        elif data_type=='testAB':
            dict_data = load_json('./split_crossdock_dataset.json')
            pdbids = [name[:4] for name in dict_data['testA']] + [name[:4] for name in dict_data['testB']]
            pdbids = set(pdbids)
            self.data_path = [path_list for path_list in load_json('./out_file/protein_flexi.json')['test'] if path_list[0].split('/')[-1][:4] in pdbids]
        elif data_type=='testCD':
            dict_data = load_json('./split_crossdock_dataset.json')
            pdbids = [name[:4] for name in dict_data['testC']] + [name[:4] for name in dict_data['testD']]
            pdbids = set(pdbids)
            self.data_path = [path_list for path_list in load_json('./out_file/protein_flexi.json')['train'] if path_list[0].split('/')[-1][:4] in pdbids]
        # elif data_type=='crossdock_coreset':
        #     self.data_path = load_json('./out_file/crossdock_57_5_4.json')['test']
        elif 'crossdock_coreset' in data_type:
            key = data_type.split('_')[-1]
            self.data_path = load_json('./out_file/pdbbind_cross2_coreset.json')[key]
        elif 'dekois'==data_type:
            self.data_path = load_json('./out_file/DEKOIS.json')['dekois_test']
            self.dekois = True
        print(data_type, len(self.data_path))
        
        
        self.save_pdb = save_pdb
        self.max_lig_size = max_lig_size # 配体原子的最大尺寸
        self.start_from_pdbid = start_from_pdbid
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.data_test = data_test
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        
    def len(self):
        return len(self.data_path)

    def get(self, idx):
        complex_graph, lig = None,None
        
        try:
            complex_graph, lig = self.preprocessing(idx)
        except Exception as e:
            print(e)
            
        if complex_graph is None:
            return None
        if self.require_ligand:
            complex_graph.mol = lig
        
        return complex_graph

    def preprocessing(self, idx):
        
        cut_pocket_path, cut_lig_path, pdb_in, lig_in, pdb_out, lig_out = self.data_path[idx]
        graph, lig = self.get_infer_complex(cut_pocket_path, cut_lig_path, pdb_in, lig_in, pdb_out, lig_out)
        
        return graph, lig
        
    def assert_tor(self, edge_index, mask_rotate):

        for idx_edge, e in enumerate(edge_index.cpu().numpy()):
            u, v = e[0], e[1]
            # check if need to reverse the edge, v should be connected to the part that gets rotated
            if  not ( (not mask_rotate[idx_edge, u]) and mask_rotate[idx_edge, v]):
                raise ValueError('torsion assert error')
    
    def get_structure(self, path=''):
        # 该函数读取蛋白质
        warnings.filterwarnings("ignore")
        biopython_parser = PDBParser()
        structure = biopython_parser.get_structure('random_id', path)
        structure = structure[0]
        return structure
    
    def get_name(self, pdb_in, pdb_out, lig_out):
        # 该函数获取图数据的名称
        n1 = pdb_in.split('/')[5].split('_')[0]
        n2 = pdb_out.split('/')[5].split('_')[0]
        n3 = '_'.join( lig_out.split('/')[5].split('_')[:2] )
        name = '_'.join([n1,n2,n3])
        return name
    
    def longest_common_subsequence_seq_idx(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 构造最长公共子序列
        lcs_length = dp[m][n]
        lcs = [''] * lcs_length
        i, j = m, n
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs[lcs_length - 1] = s1[i - 1]
                i -= 1
                j -= 1
                lcs_length -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        # 寻找最长公共子序列在原始序列中的所有下标
        lcs_indices = []
        lcs_len = len(lcs)
        idx1 = 0
        idx2 = 0
        for idx in range(lcs_len):
            while s1[idx1] != lcs[idx]:
                idx1 += 1
            while s2[idx2] != lcs[idx]:
                idx2 += 1
            lcs_indices.append((idx1, idx2))
            idx1 += 1
            idx2 += 1
            
        s1_same_idx, s2_same_idx = zip(*lcs_indices)
        return ''.join(lcs), s1_same_idx, s2_same_idx
    

    def longest_continuos_subsequence_seq_idx(self, s1, s2):
        # 该函数返回最大连续子序列,以及在s1和s2的起始终止下标
        m = len(s1)
        n = len(s2)

        # 创建一个二维数组来保存最长连续子序列的长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        max_length = 0  # 最长连续子序列的长度
        end_index = 0  # 最长连续子序列的结束索引

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_index = i - 1  # 更新最长连续子序列的结束索引
                else:
                    dp[i][j] = 0

        # 根据最长连续子序列的长度和结束索引，提取出最长连续子序列
        lcs = s1[end_index - max_length + 1: end_index + 1]

        # 寻找最长公共子序列在原始序列中的起始和终止下标
        start_idx1 = s1.find(lcs)
        end_idx1 = start_idx1 + len(lcs) - 1
        
        start_idx2 = s2.find(lcs)
        end_idx2 = start_idx2 + len(lcs) - 1
    
        return lcs, (start_idx1, end_idx1), (start_idx2, end_idx2)
    
    
    def from_chain_list_to_sequences_list(self, chain_list):
        # 该函数去水，去掉20种非人体氨基酸
        sequences = []
        res_seq = []
        ids = []
        for i, chain in enumerate(chain_list):
            seq = ''
            chain_seq = []
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    continue
                if residue.get_resname() not in residue_atoms.keys():
                    continue
                seq += three_to_one[residue.get_resname()] # 得到20种氨基酸的序列
                chain_seq.append(residue)
            if len(seq)!=0:
                sequences.append(seq)
                res_seq.append(chain_seq)
        return sequences,res_seq
    
    def get_chain_res_list(self, in_structure, out_structure):
        out_chain_res_list, in_chain_res_list = [], []
        
        chain_list_a, chain_list_b = Selection.unfold_entities(in_structure, 'C'),Selection.unfold_entities(out_structure, 'C')
        # 去掉H2O,去掉非20种人体氨基酸
        sequence_in, chain_res_list_in = self.from_chain_list_to_sequences_list(chain_list_a)
        sequence_out, chain_res_list_out = self.from_chain_list_to_sequences_list(chain_list_b)
        
        # 对所有的肽链求出最大连续公共子序列
        for s_in, s_out, chain_in, chain_out in zip(sequence_in, sequence_out, chain_res_list_in, chain_res_list_out):
            if self.continuos:# 最大连续子序列
                max_same_seq, (start_idx1, end_idx1), (start_idx2, end_idx2) = self.longest_continuos_subsequence_seq_idx(s1=s_in,s2=s_out)
                in_chain_res_list.append(chain_in[start_idx1:end_idx1+1])
                out_chain_res_list.append(chain_out[start_idx2:end_idx2+1])
                if  len(max_same_seq)/max(len(s_in),len(s_out))<0.9:
                    return None,None
            else: # 最大公共子序列
                max_same_seq, s1_same_idx, s2_same_idx = self.longest_common_subsequence_seq_idx(s1=s_in,s2=s_out)
                in_chain_res_list.append([chain_in[idx] for idx in s1_same_idx])
                out_chain_res_list.append([chain_out[idx] for idx in s2_same_idx])
        
        return in_chain_res_list, out_chain_res_list
    
    
    def get_dist(self, RES_atoms_pos, lig_atoms_pos):
        # [N,3] [M,3]
        dist_matrix = cdist(RES_atoms_pos, lig_atoms_pos)
        return dist_matrix
        
    def cut_pocket_from_pocket_cent(self, out_chain_res_list, lig_out_mol, max_r=[10,15,25]):
        pocket_idx = []
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        pocket_cent = np.average(lig_coords, axis=0, keepdims=True)
        noise_delta = np.random.normal(0, 5, size=pocket_cent.shape)
        pocket_cent = pocket_cent + noise_delta # 口袋中心
        
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A')] )
                dist_mat = self.get_dist(res_atoms_coord, pocket_cent)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        return pocket_idx
    
    def cut_pocket_from_pocket_key_res(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = self.cut_pocket(out_chain_res_list,lig_out_mol,max_r=6)
        ca_list = []
        for chain_idx,pocket_res_idx in enumerate( pocket_idx ):
            for idx in pocket_res_idx:
                ca_list += [list(atom.get_vector()) for atom in out_chain_res_list[chain_idx][idx] if atom.name=='CA']
        ca_coord_pocket = np.array(ca_list)
        N = ca_coord_pocket.shape[0]
        M = np.random.randint(1,5)
        sub_arr = ca_coord_pocket[np.random.choice(N, M, replace=False)]
        
        # 利用关键残基的坐标割口袋
        pocket_idx = []
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A')] )
                dist_mat = self.get_dist(res_atoms_coord, sub_arr)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        
        return pocket_idx
    
    def cut_pocket_from_residues_ca_distance(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = []
        # 随机选取几个配体原子，求出坐标均值,以片段去求
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        atoms_num = random.choice([len(lig_coords)//i  for i in [3,4,5]] ) # 获取配体关键原子个数
        start_idx = np.random.randint(0, len(lig_coords)-atoms_num)
        random_frag_lig_cent = np.average( lig_coords[start_idx:start_idx+atoms_num], axis=0, keepdims=True)
        
        # 求出关键残基ca坐标
        ca_coord_list = []
        distance_list_of_frag_cent_dist_with_ca = []
        for res_list in out_chain_res_list:
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                ca_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') if atom.name=='CA'] )
                dist_mat = self.get_dist(ca_coord, random_frag_lig_cent.reshape(-1,3))
                min_dist = np.min(dist_mat)
                ca_coord_list.append(ca_coord)
                distance_list_of_frag_cent_dist_with_ca.append(min_dist)
        min_dist_idx = np.argmin(distance_list_of_frag_cent_dist_with_ca)
        key_ca_coord = ca_coord_list[min_dist_idx]
        
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [ list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') if atom.name=='CA' ] )
                dist_mat = self.get_dist(res_atoms_coord, key_ca_coord.reshape(-1,3))
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        
        return pocket_idx
    
    def cut_pocket_from_residues_atom_distance(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = []
        # 随机选取几个配体原子，求出坐标均值,以片段去求
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        atoms_num = random.choice([len(lig_coords)//i  for i in [3,4,5]] ) # 获取配体关键原子个数
        start_idx = np.random.randint(0, len(lig_coords)-atoms_num)
        random_frag_lig_cent = np.average( lig_coords[start_idx:start_idx+atoms_num], axis=0, keepdims=True)
        
        # 求出关键原子坐标
        res_min_coord_list = []
        distance_min = []
        for res_list in out_chain_res_list:
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体片段中心的距离
                res_atoms_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') ] )
                dist_mat = self.get_dist(res_atoms_coord, random_frag_lig_cent)
                min_dist = np.min(dist_mat)
                res_min_coord_list.append(res_atoms_coord[np.argmin(dist_mat.reshape((-1)))])
                distance_min.append(min_dist)
        min_dist_idx = np.argmin(distance_min)
        key_atom_coord = res_min_coord_list[min_dist_idx]
        
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [ list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A') ] )
                dist_mat = self.get_dist(res_atoms_coord, key_atom_coord.reshape(-1,3))
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        
        return pocket_idx
    
    def cut_pocket(self, out_chain_res_list, lig_out_mol, max_r=10):
        pocket_idx = []
        conf = lig_out_mol.GetConformer()
        lig_coords = conf.GetPositions() # 获取配体三维坐标
        for res_list in out_chain_res_list:
            pocket_chain_idx = []
            for idx,res in enumerate(res_list):
                # 残基上所有的原子的坐标和配体所有原子坐标的距离
                res_atoms_coord = np.array( [list(atom.get_vector()) for atom in Selection.unfold_entities(res, 'A')] )
                dist_mat = self.get_dist(res_atoms_coord, lig_coords)
                min_dist = np.min(dist_mat)
                if min_dist<max_r:
                    pocket_chain_idx.append(idx)
            pocket_idx.append(pocket_chain_idx)
        return pocket_idx

    def from_pocket_idx_to_res_list(self, chain_res_list, pocket_chain_idx_list):
        # 根据口袋残基的idx将口袋残基的列表提取出来
        pocket_res = []
        for chain_res,chain_idx_list in zip(chain_res_list,pocket_chain_idx_list):
            for idx in chain_idx_list:
                pocket_res.append(chain_res[idx])
        return pocket_res
    
    def get_ca_coord(self, res):
        for atom in Selection.unfold_entities(res, 'A'):
            if atom.name=='CA':
                return list(atom.get_vector())
        return None
                
        
    def align_pocket(self, out_pocket_res, in_pocket_res):
        # 对所有存在CA坐标的残基进行提取坐标
        in_coord = []
        out_coord = []
        for res_in, res_out in zip(in_pocket_res, out_pocket_res):
            CA_in = self.get_ca_coord(res_in)
            CA_out = self.get_ca_coord(res_out)
            if CA_in is not None and CA_out is not None:
                in_coord.append(CA_in)
                out_coord.append(CA_out)
            
        in_coord , out_coord = torch.from_numpy(np.array(in_coord)).float(), torch.from_numpy(np.array(out_coord)).float()
        
        R, t = rigid_transform_Kabsch_3D_torch(in_coord.T, out_coord.T)
        aligned_in_coord = in_coord @ R.T + t.T

        rmsd, R_pocket, T_pocket = self.rmsd_test(aligned_in_coord, out_coord), R, t
        return rmsd, R_pocket, T_pocket
    

    def extract_pocket_structure(self, out_pocket_res):
        feature_list = []
        chain_c_alpha_coords, chain_n_coords, chain_c_coords, chain_c_beta_coords, chain_o_coords = [],[],[],[],[]
        for residue in out_pocket_res:
            feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
            c_alpha, n, c, c_beta, o = None, None, None, None, None
            for atom in residue:# 残基(氨基酸)对原子进行循环
                if atom.name == 'CA': # 残基alpha碳原子
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N': # 残基N原子
                    n = list(atom.get_vector())
                if atom.name == 'C': # 残基C原子
                    c = list(atom.get_vector())
                if atom.name == 'CB':# 残基beta碳原子
                    c_beta = list(atom.get_vector())
                if atom.name == 'O':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    o = list(atom.get_vector())
                if atom.name == 'O' and residue.resname == 'GLY':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    c_beta = list(atom.get_vector())
            none_res_atom_pos = [0.0, 0.0, 0.0]     
            # 五个原子都不为None
            if c_alpha != None and n != None and c != None and c_beta != None and  o != None:
                chain_c_alpha_coords.append(c_alpha) # 当前循环已经把 alpha c原子找到的时候就把它添加到alpha碳原子
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_c_beta_coords.append(c_beta)
                chain_o_coords.append(o)
                
    
        c_alpha_coords = torch.from_numpy(np.array(chain_c_alpha_coords)).float()
        n_coords = torch.from_numpy(np.array(chain_n_coords)).float()
        c_coords = torch.from_numpy(np.array(chain_c_coords)).float()
        c_beta_coords = torch.from_numpy(np.array(chain_c_beta_coords)).float()
        o_coords = torch.from_numpy(np.array(chain_o_coords)).float()
        feature_res = torch.from_numpy(np.array(feature_list)).float()
        return feature_res, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords
                        
    def out_feature_to_graph(self, complex_graph, feature_res, in_c_alpha_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, lm_embeddings=None):
        cutoff = self.receptor_radius #15
        max_neighbor = self.c_alpha_max_neighbors #24
        n_rel_pos = n_coords - c_alpha_coords
        c_rel_pos = c_coords - c_alpha_coords
        c_beta_rel_pos = c_beta_coords - c_alpha_coords
        o_rel_pos = o_coords - c_alpha_coords
        
        num_residues = len(c_alpha_coords)
        if num_residues <= 1:
            raise ValueError(f"rec contains only 1 residue!")
        
        # Build the k-NN graph
        import scipy.spatial as spa
        distances = spa.distance.cdist(in_c_alpha_coords, in_c_alpha_coords) #氨基酸alpha C的距离矩阵    该函数最
        src_list = []
        dst_list = []
        
        for i in range(num_residues):
            dst = list(np.where(distances[i, :] < cutoff)[0]) # 取出对于距离矩阵中小于15A的那些下标
            dst.remove(i) # 把对角元去掉,对角元是自己和自己的距离
            if max_neighbor != None and len(dst) > max_neighbor:
                dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1] # 距离小于15A的氨基酸可能有很多，甚至超过了24个，那么我们只取前24的下标
            if len(dst) == 0:
                dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
                print(f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. '
                    f'So we connected it to the closest other c_alpha')
            assert i not in dst # 确保i没有在dst中，因为 是自己和自己
            src = [i] * len(dst)
            src_list.extend(src) # 最大邻接数是24, src_list [0]*24 + [1]*24 + [2]*22 + .....  
            dst_list.extend(dst) # 最大邻接数是24, dst_list [0距离小于15A的下标] + [1距离小于15A的下标].....  
    
        assert len(src_list) == len(dst_list)

        # node_feat = rec_residue_featurizer(rec)
        node_feat = feature_res  # [n_res, 1]
        side_chain_vecs = torch.cat([n_rel_pos.unsqueeze(1), c_rel_pos.unsqueeze(1), 
                                     c_beta_rel_pos.unsqueeze(1), o_rel_pos.unsqueeze(1)], axis=1)
        
        # 氨基酸为粒度， 氨基酸的embdeing和氨基酸种类 [N_res, 1280] [N_res, 1]
        complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1).float() if lm_embeddings is not None else node_feat
        # alpha_c的坐标
        complex_graph['receptor'].ref_pos = c_alpha_coords
        # [N_res,4,3]  先计算C原子相对alpha_C原子的位置差, N原子相对alpha_C原子的位置差，然后把它俩concat到一起得到
        complex_graph['receptor'].ref_side_chain_vecs = side_chain_vecs
        # [2，N<255*24] 最大邻接数是24, src_list [0]*24 + [1]*24 + [2]*22 + .....      最大邻接数是24, dst_list [0距离小于15A的下标] + [1距离小于15A的下标].....  
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))
        return complex_graph  
    
    def get_complete_information_idx(self, res_list):
        complete_information_idx = []
        for idx, residue in enumerate(res_list):
            c_alpha, n, c, c_beta, o = None, None, None, None, None
            for atom in residue:# 残基(氨基酸)对原子进行循环
                if atom.name == 'CA': # 残基alpha碳原子
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N': # 残基N原子
                    n = list(atom.get_vector())
                if atom.name == 'C': # 残基C原子
                    c = list(atom.get_vector())
                if atom.name == 'CB':# 残基beta碳原子
                    c_beta = list(atom.get_vector())
                if atom.name == 'O':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    o = list(atom.get_vector())
                if atom.name == 'O' and residue.resname == 'GLY':# 残基O原子，或者甘氨酸没有beta-C时也使用O原子代替
                    c_beta = list(atom.get_vector())
            if c_alpha != None and n != None and c != None and c_beta != None and  o != None:
                complete_information_idx.append(idx)
        return set( complete_information_idx )
    
    def cut_residues_of_miss_information(self, out_pocket_res, in_pocket_res):
        out_complete_information_idx = self.get_complete_information_idx(res_list=out_pocket_res)
        in_complete_information_idx = self.get_complete_information_idx(res_list=in_pocket_res)
        idx_list = sorted( list( out_complete_information_idx & in_complete_information_idx ) )
        out_res,in_res = [],[]
        for idx in idx_list:
            out_res.append(out_pocket_res[idx])
            in_res.append(in_pocket_res[idx])
        return out_res, in_res
    
    
    def in_feature_to_graph(self, complex_graph, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords, R_pocket, T_pocket, aligned=True):
        # 保存向量特征和输入CA的坐标,在保存之前需要做口袋对齐
        if aligned:
            in_c_alpha_coords = in_c_alpha_coords @ R_pocket.T + T_pocket.T
            in_n_coords = in_n_coords @ R_pocket.T + T_pocket.T
            in_c_coords = in_c_coords @ R_pocket.T + T_pocket.T
            in_c_beta_coords = in_c_beta_coords @ R_pocket.T + T_pocket.T
            in_o_coords = in_o_coords @ R_pocket.T + T_pocket.T

        n_rel_pos = in_n_coords - in_c_alpha_coords
        c_rel_pos = in_c_coords- in_c_alpha_coords
        c_beta_rel_pos = in_c_beta_coords - in_c_alpha_coords
        o_rel_pos = in_o_coords - in_c_alpha_coords

        side_chain_vecs = torch.cat([n_rel_pos.unsqueeze(1), c_rel_pos.unsqueeze(1), 
                                     c_beta_rel_pos.unsqueeze(1), o_rel_pos.unsqueeze(1)], axis=1)
        
        complex_graph['receptor'].pos = in_c_alpha_coords
        complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
        return complex_graph
    # 
    def get_same_atoms_pos(self, res_in, res_out):
        # 获取非H原子
        in_atoms = {atom.name:list(atom.get_vector()) for atom in res_in if atom.name in residue_atoms[res_in.resname]}
        out_atoms = {atom.name:list(atom.get_vector()) for atom in res_out if atom.name in residue_atoms[res_out.resname]}
        common_keys = sorted( list( set(in_atoms.keys()) & set(out_atoms.keys())  ) )
        in_atoms_coord = []
        out_atoms_coord = []
        for atom_name in common_keys:
            in_atoms_coord.append(in_atoms[atom_name])
            out_atoms_coord.append(out_atoms[atom_name])
        return torch.from_numpy(np.array(in_atoms_coord)).float(), torch.from_numpy(np.array(out_atoms_coord)).float(), common_keys
            
    
    def atoms_pos_to_graph(self, complex_graph, out_pocket_res, in_pocket_res, R_pocket, T_pocket, align=True):
        # 将输入和输出的原子进行取交集,保存输入和输出的全原子坐标
        # 输入全原子坐标在保存之前需要做口袋对齐
        from .process_mols_pocket import get_pocket_infor
        res_num, max_atom_num = get_pocket_infor(res_list=out_pocket_res)
        res_atoms_mask = torch.zeros([res_num, max_atom_num], dtype=torch.bool)
        res_atoms_pos = torch.zeros([res_num, max_atom_num, 3])
        ref_res_atoms_pos = torch.zeros([res_num, max_atom_num, 3])
        ref_sorted_atom_names = []
        
        for idx, (res_in, res_out) in enumerate(zip(in_pocket_res,out_pocket_res)) :
            res_coord, ref_res_coord, res_sorted_atom_names = self.get_same_atoms_pos(res_in, res_out)
            res_atom_num = len(res_sorted_atom_names)
            if align:
                res_atoms_pos[idx, :res_atom_num] = res_coord @ R_pocket.T + T_pocket.T # 输入进行对齐
            else:
                res_atoms_pos[idx, :res_atom_num] = res_coord 
            ref_res_atoms_pos[idx, :res_atom_num] = ref_res_coord
            res_atoms_mask[idx, :res_atom_num] = True
            ref_sorted_atom_names.append(res_sorted_atom_names)
    
        complex_graph['receptor'].res_atoms_mask = res_atoms_mask # [N_res,]
        complex_graph['receptor'].ref_res_atoms_pos = ref_res_atoms_pos.float() # [N_res, atoms, 3]
        complex_graph['receptor'].res_atoms_pos = res_atoms_pos.float() # [N_res, atoms, 3]
        complex_graph['receptor'].ref_sorted_atom_names = ref_sorted_atom_names 
        return complex_graph
    
    def set_coord_structure(self, pocket_R,pocket_T,structure,full_id_list):
        align_pocket_res = []
        for full_id in full_id_list:
            chain, res_id = full_id[2],full_id[3]
            res = structure[chain][res_id]
            for atom in res:
                old_coord = torch.from_numpy (atom.coord)
                new_coord = old_coord @ pocket_R.T + pocket_T.T
                atom.set_coord( new_coord.numpy().reshape((3,)) )
            align_pocket_res.append(res)
        return structure, align_pocket_res
    
    def save_pocket_out(self, structure, path, pocket_res_list):
        from Bio.PDB import PDBIO
        indices = []
        for res in pocket_res_list:
            indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
        selector = ResidueSelector(indices)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path,select=selector)
    
    def save_pocket_in_aligned(self, structure, path, pocket_res_list, pocket_R, pocket_T, aligned=True):
        from Bio.PDB import PDBIO
        indices = []
        full_id_list = []
        
        for res in pocket_res_list:
            indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
            full_id_list.append(res.full_id)
        align_pocket_res = None
        if aligned:
            structure, align_pocket_res = self.set_coord_structure(pocket_R,pocket_T,structure,full_id_list)
        selector = ResidueSelector(indices)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path,select=selector)
        return  align_pocket_res
    
    def get_mol_from_sdf_pdb_mol2(self,path_list,sanitize=True):
        # path_list路径或者路径列表
        if type(path_list)==str:
            lig_out_path = path_list
            #获取配体
            if not os.path.exists(lig_out_path):
                print('load out ligand is not exist')
                return None
            if '.sdf' in lig_out_path:
                lig_out_mol = Chem.SDMolSupplier(lig_out_path, sanitize=sanitize, removeHs=True)[0]
                return lig_out_mol
            elif '.pdb' in lig_out_path:
                lig_out_mol = Chem.MolFromPDBFile(lig_out_path, sanitize=sanitize, removeHs=True)
                return lig_out_mol
        for path in path_list:
            lig_out_path = path
            #获取配体
            if not os.path.exists(lig_out_path):
                print('load out ligand is not exist')
                continue
            if '.sdf' in lig_out_path:
                lig_out_mol = Chem.SDMolSupplier(lig_out_path, sanitize=sanitize, removeHs=True)[0]
                if lig_out_mol is not None:
                    return lig_out_mol
            elif '.pdb' in lig_out_path:
                lig_out_mol = Chem.MolFromPDBFile(lig_out_path, sanitize=sanitize, removeHs=True)
                if lig_out_mol is not None:
                    return lig_out_mol
            elif '.mol2' in lig_out_path:
                lig_out_mol = Chem.MolFromMol2File(lig_out_path, sanitize=sanitize, removeHs=True)
                if lig_out_mol is not None:
                    return lig_out_mol
        return None
    
    def get_infer_complex(self,cut_pocket_path, cut_lig_path, pdb_in_path, lig_in_path, pdb_out_path, lig_out_path):
        # cut_pocket_path
        # cut_lig_path
        # pdb_in_path 对接的蛋白
        # lig_in_path 对接的配体
        # pdb_out_path 对齐的蛋白
        # lig_out_path 配体晶体
        
        pdbid_in = cut_pocket_path.split('/')[-1][:4]
        pdbid_out = pdb_out_path.split('/')[-1][:4]
        name = pdbid_in+'-'+pdbid_out
        # 判断路径是目录还是文件
        if os.path.isdir(lig_in_path):
            lig_in_path = os.path.join(lig_in_path, random.choice(os.listdir(lig_in_path) )) # 10个构象随机选取一个配体输入
            
        # 将配体路径或者配体路径列表传入(晶体配体)
        lig_out_path_list = lig_out_path
        if os.path.isdir(lig_out_path):
            lig_out_path_list = [os.path.join(lig_out_path, name) for name in os.listdir(lig_out_path) if '.sdf' in name or '.mol2' in name]
        # 将割口袋的配体路径或者路径列表传入
        cut_pocket_lig_path_list = cut_lig_path
        if os.path.isdir(cut_lig_path):
            cut_pocket_lig_path_list = [os.path.join(cut_lig_path, name) for name in os.listdir(cut_lig_path) if '.sdf' in name or '.mol2' in name]
        
        complex_graph = HeteroData()# 定义异构图
        complex_graph['name'] = name
        
        lig_in_mol = self.get_mol_from_sdf_pdb_mol2(path_list=lig_in_path)
        
        if self.dekois:
            name = lig_in_path.split('/')[-1][:-4]
            complex_graph['name'] = name
        
        # 割口袋的配体
        cut_pocket_lig_mol = self.get_mol_from_sdf_pdb_mol2(path_list=cut_pocket_lig_path_list, sanitize=False)
        
        
        if self.dekois:
            complex_graph = get_lig_feature(mol_=lig_in_mol, complex_graph=complex_graph,keep_original=self.keep_original, 
                                            remove_hs=self.remove_hs, sdf_path=lig_in_path, data_type=self.data_type, lig_in_mol=lig_in_mol,dekois=self.dekois
                                            ) # 根据lig_out获取配体特征/由于这里还是测试,因此可以通过lig_out_mol获取特征
        else:
            
            lig_out_mol = self.get_mol_from_sdf_pdb_mol2(path_list=lig_out_path_list)
            if lig_out_mol is None: 
                print('load out ligand is None')
                return (None,None)
            complex_graph = get_lig_feature(mol_=lig_out_mol, complex_graph=complex_graph,keep_original=self.keep_original, 
                                            remove_hs=self.remove_hs, sdf_path=lig_in_path, data_type=self.data_type, lig_in_mol=lig_in_mol,
                                            ) # 根据lig_out获取配体特征/由于这里还是测试,因此可以通过lig_out_mol获取特征
            
        
        if complex_graph is None: # 配体特征获取失败
            return (None,None)
        #对于配体被分为多个部分的连通图,则跳过该类分子
        if not complex_graph.is_connected: 
            print('lig fragments num > 1')
            return (None,None)
        
        #获取蛋白特征
        out_structure = self.get_structure(pdb_out_path)
        in_structure = self.get_structure(cut_pocket_path) # 根据输入割口袋
        
        # 去H2O,去非20类氨基酸,求出最大公共子序列的残基列表
        in_chain_res_list, out_chain_res_list = self.get_chain_res_list(in_structure, out_structure)
        if in_chain_res_list is None:
            return (None,None)
        
        # 根据晶体割口袋,割口袋的几种方式
        method = [0]
        cut_pocket_method = random.choice(method)
        if cut_pocket_method==0: # 根据配体每个原子距离残基6-10A割口袋
            if cut_pocket_lig_mol is None:
                print(cut_pocket_lig_path_list)
            pocket_chain_idx_list = self.cut_pocket(in_chain_res_list, cut_pocket_lig_mol, max_r=10) #[6,8,10]
        elif cut_pocket_method==1:# 根据配体中心位置10-25A内的残基割口袋
            pocket_chain_idx_list = self.cut_pocket_from_pocket_cent(in_chain_res_list, cut_pocket_lig_mol, max_r=random.choice([10,15,25]))
        elif cut_pocket_method==2: # 先在配体上找一段连续片段,然后求出片段中心,然后找到最近的残基,再根据这个残基的alpha-C 10A内的CA割口袋
            pocket_chain_idx_list = self.cut_pocket_from_residues_ca_distance(in_chain_res_list, cut_pocket_lig_mol, max_r=10)
        elif cut_pocket_method==3:# 同上.但用残基原子割
            pocket_chain_idx_list = self.cut_pocket_from_residues_atom_distance(in_chain_res_list, cut_pocket_lig_mol, max_r=10)
        elif cut_pocket_method==4:# 随机找到的关键残基割口袋
            pocket_chain_idx_list = self.cut_pocket_from_pocket_key_res(in_chain_res_list, cut_pocket_lig_mol, max_r=10)
        
        
        
        out_pocket_res = self.from_pocket_idx_to_res_list(chain_res_list=out_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list)
        in_pocket_res = self.from_pocket_idx_to_res_list(chain_res_list=in_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list)
        
        # 对于五个关键原子是否存在进行判断,并且对in和out进行取交集操作
        out_pocket_res, in_pocket_res = self.cut_residues_of_miss_information(out_pocket_res, in_pocket_res)
        
        # 进行口袋对齐
        ca_rmsd, R_pocket, T_pocket = self.align_pocket(out_pocket_res, in_pocket_res) # 模型层面不做对齐,脚本实现二次割口袋和对齐
        
        if self.save_pdb:
            pocket_name = cut_pocket_path.split('/')[5]
            complex_graph['pocket_name'] = pocket_name
            ground_pocket_path = os.path.join('./out_file/dekois_pocket10A/ground_pocket', pocket_name)
            os.makedirs(ground_pocket_path, exist_ok=True)
            uninduced_pocket_path = os.path.join('./out_file/dekois_pocket10A/uninduced_pocket', pocket_name)
            os.makedirs(uninduced_pocket_path, exist_ok=True)
            
            complex_graph['receptor'].R_pocket, complex_graph['receptor'].T_pocket = copy.deepcopy(R_pocket), copy.deepcopy( T_pocket)
            complex_graph['receptor'].out_structure, complex_graph['receptor'].in_structure = copy.deepcopy(out_structure), copy.deepcopy(in_structure)
            complex_graph['receptor'].out_pocket_res  = copy.deepcopy(out_pocket_res)
            # os.makedirs()
            self.save_pocket_out(structure=complex_graph['receptor'].out_structure, 
                                 path=ground_pocket_path + f'/{name}_pocket.pdb', 
                                 pocket_res_list=complex_graph['receptor'].out_pocket_res) # ground口袋
            aligned_pocket_res_list = self.save_pocket_in_aligned(structure=complex_graph['receptor'].in_structure, 
                                                                  path=uninduced_pocket_path + f'/{name}_pocket.pdb', 
                                                                pocket_res_list=in_pocket_res, pocket_R=R_pocket, pocket_T=T_pocket, aligned= not self.save_pdb,
                                        ) # 没有经过诱导的口袋,不进行对齐
            
            complex_graph['receptor'].in_pocket_res = aligned_pocket_res_list if aligned_pocket_res_list is not None else in_pocket_res
        
        
        if ca_rmsd>self.max_align_rmsd or torch.abs (ca_rmsd) <self.min_align_rmsd:
            print(f'aligned pocket rmsd > {self.max_align_rmsd}',ca_rmsd)
            return (None,None)
        
        complex_graph['receptor'].ca_rmsd = ca_rmsd.item()
        
        # 得到口袋残基的标量特征,口袋alphaC的坐标
        out_feature_res, out_c_alpha_coords, out_n_coords, out_c_coords, out_c_beta_coords, out_o_coords = self.extract_pocket_structure(out_pocket_res) 
        in_feature_res, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords = self.extract_pocket_structure(in_pocket_res)
        
        if out_c_alpha_coords.shape[0]<=5:
            print('alpha-C num < 5')
            return (None,None)
        
        # 输出蛋白的信息————输出节点(CA)的坐标, 残基标量特征, 边的关系(根据CA建立), 向量特征, 添加到图数据中
        complex_graph = self.out_feature_to_graph(complex_graph, in_feature_res, in_c_alpha_coords, out_c_alpha_coords, out_n_coords, out_c_coords, out_c_beta_coords, out_o_coords, lm_embeddings=None)
        # 输入蛋白的信息————输入节点(CA)的坐标, 向量特征, 全原子坐标(和全原子的符号做排序对齐),添加到图数据中. 所有的向量特征在传入图数据前都进行对齐
        complex_graph = self.in_feature_to_graph(complex_graph, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords, R_pocket, T_pocket, aligned=not self.save_pdb)
        # 将输入和输出的原子取交集,方便之后求loss, 全原子坐标, 排序后的全原子的符号, 全原子mask, 添加到图数据        
        complex_graph = self.atoms_pos_to_graph(complex_graph, out_pocket_res, in_pocket_res, R_pocket, T_pocket, align= not self.save_pdb)
        
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center# 去中心化,输入口袋
        complex_graph['receptor'].ref_pos -= protein_center # 去中心化,输出口袋
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask # [N_res,]
        complex_graph['receptor'].res_atoms_pos[res_atoms_mask] -= protein_center # esm-fold生成的蛋白坐标以晶体蛋白口袋中心为坐标系  [N_res, atoms, 3]
        complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask] -= protein_center  # 晶体蛋白坐标以蛋白口袋中心为坐标系
        complex_graph['receptor'].crossdock_rmsd = self.get_pocket_rmsd(complex_graph)
        complex_graph['ligand'].pos -= ligand_center # 随机生成的构象进行中心化
        if not self.dekois:
            complex_graph['ligand'].ref_pos -= protein_center # ground_truth的配体坐标以蛋白口袋中心为坐标系
        complex_graph.original_center = protein_center
            
        return complex_graph, lig_out_mol
    
    def get_complex(self, pdb_in_path, lig_in_path, pdb_out_path, lig_out_path):
        
        # 判断路径是目录还是文件
        if os.path.isdir(lig_in_path):
            lig_in_path = os.path.join(lig_in_path, random.choice(os.listdir(lig_in_path) )) # 随机选取一个输入
            
        # 将配体路径或者配体路径列表传入
        lig_out_path_list = lig_out_path
        if os.path.isdir(lig_out_path):
            lig_out_path_list = [os.path.join(lig_out_path, name) for name in os.listdir(lig_out_path) if '.sdf' in name or '.mol2' in name]
        if self.pretrain_method=='pretrain_method1':
            lig_in_path = copy.deepcopy(lig_out_path)
        
        if self.data_type in ['train', 'testA','testB','testC','testD','valid']:
            name = self.get_name(pdb_in_path, pdb_out_path, lig_out_path) # crossdock pdb_in pdb_out lig_out
        elif 'pretrain' in self.data_type:
            name = lig_out_path.split('/')[-1][:4]
        
        name = pdb_out_path.split('/')[-1][:4]
        
        complex_graph = HeteroData()# 定义异构图
        #complex_graph['name'] = name
        
        #获取配体的特征
        
        lig_out_mol = self.get_mol_from_sdf_pdb_mol2(path_list=lig_out_path_list)
            
            
        if lig_out_mol is None: 
            print('load out ligand is None')
            return (None,None)
        complex_graph = get_lig_feature(mol_=lig_out_mol,complex_graph=complex_graph,keep_original=self.keep_original, remove_hs=self.remove_hs, sdf_path=lig_in_path, data_type=self.data_type, pretrain_method=self.pretrain_method) # 根据lig_out获取配体特征
        if complex_graph is None: # 配体特征获取失败
            return (None,None)
        #对于配体被分为多个部分的连通图,则跳过该类分子
        if not complex_graph.is_connected: 
            print('lig fragments num > 1')
            return (None,None)
        
        #获取蛋白特征
        out_structure = self.get_structure(pdb_out_path)
        in_structure = self.get_structure(pdb_in_path)
        
        # 去H2O,去非20类氨基酸,求出最大公共子序列的残基列表
        in_chain_res_list, out_chain_res_list = self.get_chain_res_list(in_structure, out_structure)
        # 根据晶体割口袋
        pocket_chain_idx_list = self.cut_pocket(out_chain_res_list, lig_out_mol, max_r=10)
        out_pocket_res = self.from_pocket_idx_to_res_list(chain_res_list=out_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list)
        in_pocket_res = self.from_pocket_idx_to_res_list(chain_res_list=in_chain_res_list, pocket_chain_idx_list=pocket_chain_idx_list)
        # 对于五个关键原子是否存在进行判断,并且对in和out进行取交集操作
        out_pocket_res, in_pocket_res = self.cut_residues_of_miss_information(out_pocket_res, in_pocket_res)
        
        # 进行口袋对齐   
        ca_rmsd, R_pocket, T_pocket = self.align_pocket(out_pocket_res, in_pocket_res)
        
        if self.save_pdb:
            complex_graph['receptor'].R_pocket, complex_graph['receptor'].T_pocket = R_pocket, T_pocket
            complex_graph['receptor'].out_structure, complex_graph['receptor'].in_structure = out_structure, in_structure
            complex_graph['receptor'].out_pocket_res, complex_graph['receptor'].in_pocket_res = out_pocket_res, in_pocket_res
        
        if ca_rmsd>30:
            print('aligned pocket rmsd > 20',ca_rmsd)
            return (None,None)
        if ca_rmsd>30:
            print('aligned pocket rmsd',ca_rmsd)
            print(pdb_in_path, lig_in_path, pdb_out_path, lig_out_path)
        complex_graph['receptor'].ca_rmsd = ca_rmsd.item()
        
        # 得到口袋残基的标量特征,口袋alphaC的坐标
        out_feature_res, out_c_alpha_coords, out_n_coords, out_c_coords, out_c_beta_coords, out_o_coords = self.extract_pocket_structure(out_pocket_res) 
        in_feature_res, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords = self.extract_pocket_structure(in_pocket_res)
        
        if out_c_alpha_coords.shape[0]<=5:
            print('alpha-C num < 5')
            return (None,None)
        
        # 输出蛋白的信息————输出节点(CA)的坐标, 残基标量特征, 边的关系(根据CA建立), 向量特征, 添加到图数据中
        complex_graph = self.out_feature_to_graph(complex_graph, out_feature_res, out_c_alpha_coords, out_n_coords, out_c_coords, out_c_beta_coords, out_o_coords, lm_embeddings=None)
        # 输入蛋白的信息————输入节点(CA)的坐标, 向量特征, 全原子坐标(和全原子的符号做排序对齐),添加到图数据中. 所有的向量特征在传入图数据前都进行对齐
        complex_graph = self.in_feature_to_graph(complex_graph, in_c_alpha_coords, in_n_coords, in_c_coords, in_c_beta_coords, in_o_coords, R_pocket, T_pocket)
        # 将输入和输出的原子取交集,方便之后求loss, 全原子坐标, 排序后的全原子的符号, 全原子mask, 添加到图数据        
        complex_graph = self.atoms_pos_to_graph(complex_graph, out_pocket_res, in_pocket_res, R_pocket, T_pocket)
        
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center# 去中心化,输入口袋
        complex_graph['receptor'].ref_pos -= protein_center # 去中心化,输出口袋
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask # [N_res,]
        complex_graph['receptor'].res_atoms_pos[res_atoms_mask] -= protein_center # esm-fold生成的蛋白坐标以晶体蛋白口袋中心为坐标系  [N_res, atoms, 3]
        complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask] -= protein_center  # 晶体蛋白坐标以蛋白口袋中心为坐标系
        complex_graph['receptor'].crossdock_rmsd = self.get_pocket_rmsd(complex_graph)
        complex_graph['ligand'].pos -= ligand_center # 随机生成的构象进行中心化
        complex_graph['ligand'].ref_pos -= protein_center # ground_truth的配体坐标以蛋白口袋中心为坐标系
        complex_graph.original_center = protein_center
            
        return complex_graph, lig_out_mol
    
    def get_pocket_rmsd(self, complex_graph):
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask
        res_atoms_pos = complex_graph['receptor'].res_atoms_pos[res_atoms_mask]
        ref_res_atoms_pos = complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask]
        return torch.sqrt(torch.sum((res_atoms_pos-ref_res_atoms_pos)**2,dim=1)).mean().item()
        
    def compute_RT(self, data):
        # 传入的配体坐标如果是一个列表,那么从列表中抽取一个坐标
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
            
        data.tr_score = self.get_lig_translation(data)
        lig_rmsd = self.rmsd_test(x=data['ligand'].pos, y=data['ligand'].ref_pos)
        sota_lig_rmsd = self.rmsd_test(x=self.modify_lig_conformer(x=data['ligand'].pos, tr_update=data.tr_score), y=data['ligand'].ref_pos)
        # print(f'{data.name} lig rmsd/sota_rmsd: ', lig_rmsd.item(), sota_lig_rmsd.item())
        data.res_rot_score, data.res_tr_score = self.get_res_rotation_vector(data)
        
        pocket_rmsd = self.rmsd_test(x=data['receptor'].res_atoms_pos, 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()

        print(f'{data.name} pocket rmsd: ', pocket_rmsd)
        res_rmsd = self.pocket_rmsd(x=data['receptor'].res_atoms_pos, y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        
        sota_pocket_rmsd = self.rmsd_test(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()
        
        aligned_res_rmsd = self.pocket_rmsd(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        deca_list = [(x1-x2)/x1 for x1,x2 in zip(res_rmsd,aligned_res_rmsd) if x1>x2]
        deca = np.average(deca_list)
        rate = len(deca_list)/len(res_rmsd)
        
        data['receptor'].deca = deca
        data['receptor'].rate = rate
        data['receptor'].sota_pocket_rmsd = sota_pocket_rmsd
        print('rate',rate, 'deca rate ', np.average(deca))
        
        # print(f'{data.name} esm-fold rmsd: ',data['receptor'].esm_rmsd)
        
        
        
        return data
        
    def get_lig_translation(self, data):
        assert data['ligand'].ref_pos.shape[0] == data['ligand'].pos.shape[0]
        lig_tr = data['ligand'].ref_pos - data['ligand'].pos
        return lig_tr # [lig_atoms,3]
    
    def get_res_translation(self, data):
        res_tr = data['receptor'].ref_pos - data['receptor'].pos
        return res_tr
    
    def get_res_rotation_vector(self, data):
        
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        ref_atoms_pos = data['receptor'].ref_res_atoms_pos    # ground truth 所有原子坐标
        pos_mask = data['receptor'].res_atoms_mask
        
        rot_mat, tr_vec =self.point_cloud_to_ror_matrix(ref_atoms_pos, atoms_pos, pos_mask=pos_mask) #[res, 3, 3]
        rot_vector = self.R_batch_to_axis_vec(R=rot_mat) #[res, 3]
        return rot_vector, tr_vec
        
    def point_cloud_to_ror_matrix(self, pos, ref, pos_mask, pos_ca=None, ref_ca=None):
        # 两堆点云计算旋转矩阵
        # pos [N,M,3] [res, atoms, 3]
        # ref [N,M,3]
        # pos_mask [N,M]
        # pos_ca, ref_ca [N,3]
        # N : number of examples
        # M : number of atoms
        # R,T maps local reference onto global pos
        if pos_mask is None:
            pos_mask = torch.ones(pos.shape[:2], device=pos.device)
        else:
            if pos_mask.shape[0] != pos.shape[0]:
                raise ValueError("pos_mask should have same number of rows as number of input vectors.")
            if pos_mask.shape[1] != pos.shape[1]:
                raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
            if pos_mask.ndim != 2:
                raise ValueError("pos_mask should be 2 dimensional.")
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        pos_c = pos - pos_mu
        ref_c = ref - ref_mu
    
        # Covariance matrix
        H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c.to(torch.float32))
        U, S, Vh = torch.linalg.svd(H)
        # Decide whether we need to correct rotation matrix to ensure right-handed coord system
        locs = torch.linalg.det(U @ Vh) < 0
        S[locs, -1] = -S[locs, -1]
        U[locs, :, -1] = -U[locs, :, -1]
        # Rotation matrix
        R = torch.einsum('bji,bkj->bik', Vh, U)
        T = pos_mu - torch.einsum('bij,bkj->bki', R, ref_mu)
        return R,T.squeeze(1)
        
    def R_batch_to_axis_vec(self, R):
        """
        旋转矩阵到旋转向量
        [N,3,3] >>> [N,3]
        """
        J = (R - R.permute([0,2,1])) / 2
        t = torch.cat( [ (-J[:, 1, 2]).unsqueeze(1) , (J[:, 0, 2]).unsqueeze(1),  (-J[:, 0, 1]).unsqueeze(1)], dim=1)
        t_norm = torch.norm(t, dim=1, keepdim=True)
        theta = torch.asin(t_norm)
        angle = theta / np.pi * 180
        r = t / t_norm
        v = r * theta
        return v
    
    def rmsd_test(self, x, y, mask=None):
        # x y [res, atoms, 3]
        if mask is not None:
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y
        # x y [atoms, 3]
        
        return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))
        
    def pocket_rmsd(self, x,y,mask):
        rmsd = []
        for i in range(len(mask)):
            rmsd.append( self.rmsd_test(x[i],y[i],mask[i]).item() )
        # max_idx = rmsd.index(max(rmsd))
        # print('res max/mean rmsd ', max(rmsd), self.rmsd_test(x,y,mask))
        return rmsd
    
    def modify_lig_conformer(self, x, tr_update):
        return x + tr_update
        
    def modify_pocket_conformer(self, data, tr_update, rot_update):
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
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
        
    def modify_pocket_conformer2(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = copy.deepcopy( data['receptor'].res_atoms_pos )   # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        max_atom_num = atoms_pos.shape[1]
        
        # 做旋转
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = vec_to_R(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        # atoms_pos[pos_mask] = torch.einsum('bij,bkj->bki',rot_mat[pos_mask].float(), atoms_pos[pos_mask].unsqueeze(1)).squeeze(1) 
        
        # 做平移
        # atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 平移旋转
        atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + tr_update.unsqueeze(1)
        
        return atoms_pos
class EsmFoldDataSet(Dataset):
    def __init__(self, root, transform=None, data_type='train',split_path='',
                limit_complexes=0,start_from_pdbid=None,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False, data_test=True):

        super(EsmFoldDataSet, self).__init__(root, transform)
        self.pdbbind_dir = root # pdb晶体文件路径
        self.esm_pdb_path = '/userdata/xiaoqi/EsmFoldPredict/align_pdb_0529' # esmfold预测的蛋白文件路径
        
        self.generate_ligand_dir = '/userdata/xiaoqi/EsmFoldPredict/ligand10conformers' # sdf文件路径
        self.esm_embeddings_path = esm_embeddings_path # esm embedding路径 所有的碎文件
        self.lm_embedding_paths = self.from_path_to_sorted_dict()
        self.data_type = data_type # coreset数据集划分方式, 'train'
        self.split_path = split_path # coreset数据集划分方式 路径
        self.complex_names_all = self.get_all_pdbid() # 获取所有公共pdb_pid
        # self.lm_embeddings_chains_all = self.esm_embedding_load() # 读取所有embedding
        
        self.max_lig_size = max_lig_size # 配体原子的最大尺寸
        self.start_from_pdbid = start_from_pdbid
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.data_test = data_test
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        
    def len(self):
        return len(self.complex_names_all)

    def get(self, idx):
        complex_graph, lig = self.preprocessing(idx)
        if complex_graph is None:
            return None
        # 评测蛋白之间的rmsd,返回小于5A的esmFold的数据
        if complex_graph['receptor'].esm_rmsd>5:
            return None
        if self.require_ligand:
            complex_graph.mol = copy.deepcopy(lig)
        # 是否计算残基和ground truth之间的的平移旋转量，评测最小的rmsd
        if False:
            complex_graph = self.compute_RT(complex_graph)
        return complex_graph

    
    def load_json(self, data_type='train'):
        with open(self.split_path, 'r') as f:
            dict_data = json.load(f)
        return dict_data[data_type]
    
    def get_all_pdbid(self):
        # coreset的划分方式,读取训练集的所有pdb_id
        complex_names_all = self.load_json(data_type=self.data_type) # 读取所有蛋白 训练集/验证集
        
        sdf_names = set( [name[:4] for name in os.listdir(self.generate_ligand_dir)] )
        # 所有esmfold预测的pdb_id 
        if self.esm_pdb_path is not None:
            esm_pdb_names = set( [file_name[:4] for file_name in os.listdir(self.esm_pdb_path)] )
            # 求个交集
            complex_names_all = set(complex_names_all) & esm_pdb_names
            # 将sdf已经求出的进行训练
            complex_names_all = complex_names_all & sdf_names
            
            # 将失败的数据剔除在外
            all_error = '6v1c 4po7 4u5t 3rum 3svj 3eu7 4nw2 4us4 5tcy 5y1u 5zbz 6hvj 4b0c 3run 4gv8 4qzs 1w70 3bc3 2ao6 4uil 5z95 6rsa'
            error_set = set( all_error.split() )
            complex_names_all = complex_names_all - error_set
            
            # 将所有rmsd小于5A的剔除在外
            pdbid_rmsd5 = self.get_names_to_set('/home/xiaoqi/PPLFDocking/esm/security_pdbid_0529.xlsx')
            complex_names_all = list(set(complex_names_all) & pdbid_rmsd5)
        # 所有数据量
        print(f'Loading {len(complex_names_all)} complexes.')
        return complex_names_all
    
    def get_names_to_set(self,path):
        df = pd.read_excel(path)
        list_data = df.values.tolist()
        pdb_ids = [item[0] for item in list_data if item[1] < 5]
        return set(pdb_ids)
    
    def esm_embedding_load(self):
        complex_names_all = self.complex_names_all
        if self.esm_embeddings_path is not None:
            # id_to_embeddings = torch.load('/mnt/d/data/esm2_3billion_embeddings.pt')
            id_to_embeddings = torch.load(self.esm_embeddings_path)

            chain_embeddings_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                tmp = key.split('_')
                key_name, chain_id = tmp[0], int(tmp[-1])
                if key_name in complex_names_all:
                    chain_embeddings_dictlist[key_name].append((chain_id, embedding))
            new_embeddings_dict = dict()
            for key, embedding in chain_embeddings_dictlist.items():
                sorted_embedding = sorted(embedding, key=lambda x: x[0])
                new_embeddings_dict[key] = [i[1] for i in sorted_embedding]
            chain_embeddings_dictlist = new_embeddings_dict
            
            lm_embeddings_chains_all = []
            esm_not_exit_embedding = []
            for name in complex_names_all:
                try:
                    lm_embeddings_chains_all.append(chain_embeddings_dictlist[name])
                except:
                    esm_not_exit_embedding.append(name)
            for name in esm_not_exit_embedding:
                del complex_names_all[complex_names_all.index(name)]
            
            
            print('esm not exit embedding num is ',len(esm_not_exit_embedding))

        else:
            raise ValueError
        return lm_embeddings_chains_all
    
    def from_path_to_sorted_dict(self):
        paths_dict = defaultdict(list)
        pdbid_chain_names = os.listdir(self.esm_embeddings_path)
        for pdbid_chain in pdbid_chain_names:
            tmp = pdbid_chain.split('_')
            key_name, chain_id = tmp[0], int(tmp[-1])
            paths_dict[key_name].append((pdbid_chain, chain_id))
        new_dict = dict()
        for key,pdbid_chain_list in paths_dict.items():
            sorted_pdbid_chain = sorted(pdbid_chain_list, key=lambda x:x[1])
            new_dict[key] = [os.path.join(self.esm_embeddings_path, x[0]) for x in sorted_pdbid_chain]
        return new_dict
    
    def get_lm_embedding(self, idx):
        pdbid = self.complex_names_all[idx]
        # 按照链的顺序读取一个蛋白的lm_embedding
        lm_embedding = []
        paths = self.lm_embedding_paths[pdbid] # 这是一个存放所有蛋白特征的路径字典,并且已经排好序 {'4aqp':[path1,path2,path3],}
        for path in paths:
            lm_embedding.append(torch.load(path))
        return lm_embedding
        
    def preprocessing(self, idx):
        complex_names_all = self.complex_names_all
        lm_embeddings_chains = self.get_lm_embedding(idx)
        pdb_id_esm_embedding_list = [complex_names_all[idx], lm_embeddings_chains, None, None]
        graph, lig = self.get_complex(par=pdb_id_esm_embedding_list)

        return graph, lig
        
    def assert_tor(self, edge_index, mask_rotate):

        for idx_edge, e in enumerate(edge_index.cpu().numpy()):
            u, v = e[0], e[1]
            # check if need to reverse the edge, v should be connected to the part that gets rotated
            if  not ( (not mask_rotate[idx_edge, u]) and mask_rotate[idx_edge, v]):
                raise ValueError('torsion assert error')
        
    def get_complex(self, par):
        
        
        name, lm_embedding_chains, ligand, ligand_description = par
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)):
            print("Folder not found", name)# 晶体文件夹不存在
            return None,None
        # 模型的输入
        # 配体: rdkit生成,坐标norm化
        # 蛋白: esmFold生成,
        # 模型的label:
        # 配体: 晶体数据,坐标根据蛋白口袋中心norm化
        # 蛋白: 晶体数据,根据配体割口袋,根据口袋中心norm化

        try:
          
            rec_model = parse_receptor(name, self.pdbbind_dir) # PDB文件读取晶体数据
            esm_rec = parse_esm_PDB(name, pdbbind_dir=self.esm_pdb_path) # pdb文件读取esm结构数据
            rec_model, esm_rec = extract_esmProtein_crystalProtein(rec=rec_model, esm_rec=esm_rec, lm_embedding_chains=lm_embedding_chains) # 同时前处理

        except Exception as e:
            print(f'Skipping {name} (bio load) because of the error:')
            print(e)
            return None,None
        
        # ligs_rdkit_gen = read_sdf_to_mol_list(os.path.join(self.generate_ligand_dir, f'{name}_ligand.sdf'), sanitize=True, remove_hs=False)
        ligs = read_all_mols(self.pdbbind_dir, name, remove_hs=False)
        
        if len(ligs)==0:
            print(f'{name} lig is not load')
            return None,None
        
        complex_graphs = []
        failed_indices = []

        # 有的配体可能失败,如果sdf和mol2有一个成功,那么就可结束
        for k, lig in ligs.items():
            
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                break
            complex_graph = HeteroData()# 定义异构图
            complex_graph['name'] = name
            # try:
            # 在图中添加配体信息
            
            useful_key = None
        
            try:
                # print(f'using {name} {k}')
                sdf_path = os.path.join(self.generate_ligand_dir, f'{name}_ligand.sdf')
                get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                            self.num_conformers, remove_hs=self.remove_hs, sdf_path=sdf_path)
                self.assert_tor(complex_graph['ligand', 'ligand'].edge_index.T[complex_graph['ligand'].edge_mask], complex_graph['ligand'].mask_rotate if isinstance(complex_graph['ligand'].mask_rotate, np.ndarray) else complex_graph['ligand'].mask_rotate[0])
                useful_key = k
                
                feature_res, rec_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, lm_embeddings,\
                    rec, close_residues, selector, res_chain_full_id_list = \
                        extract_receptor_pocket_structure(rec_model, lig, lm_embedding_chains=lm_embedding_chains)
                
                if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                    print(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                    break
                # 对graph添加残基ground_truth的标量和向量信息
                get_rec_graph(feature_res, rec_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, complex_graph, 
                                rec=rec, res_list=close_residues, selector=selector, res_chain_full_id_list=res_chain_full_id_list,
                                rec_radius=self.receptor_radius,c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                                atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
            
                add_rec_vector_infor(complex_graph=complex_graph, res_chain_full_id_list=res_chain_full_id_list, 
                                    pdb_rec=esm_rec, ref_sorted_atom_names=complex_graph['receptor'].ref_sorted_atom_names)
                
            except Exception as e:
                print(f'Skipping {name} (process) because of the error:')
                print(e)
                break

            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= protein_center
            complex_graph['receptor'].ref_pos -= protein_center # 蛋白CA
            res_atoms_mask = complex_graph['receptor'].res_atoms_mask # [N_res,]
            complex_graph['receptor'].res_atoms_pos[res_atoms_mask] -= protein_center # esm-fold生成的蛋白坐标以晶体蛋白口袋中心为坐标系  [N_res, atoms, 3]
            complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask] -= protein_center  # 晶体蛋白坐标以蛋白口袋中心为坐标系
            complex_graph['receptor'].esm_rmsd = self.get_pocket_rmsd(complex_graph)
            # print('esmfold and crystal pocket rmsd: ', complex_graph['receptor'].esm_rmsd)
            if (not self.matching) or self.num_conformers == 1:
                # complex_graph['ligand'].pos -= protein_center
                complex_graph['ligand'].pos -= ligand_center # 随机生成的构象进行中心化
                complex_graph['ligand'].ref_pos -= protein_center # ground_truth的配体坐标以蛋白口袋中心为坐标系
            else:
                # 多构象
                for p in complex_graph['ligand'].pos:
                    p -= protein_center
            complex_graph.original_center = protein_center
            complex_graphs.append(complex_graph)
            break
        
        if useful_key is not None and len(complex_graphs)>0:
            ligs = [ligs[useful_key]]
        else:
            ligs = []
        
        if len(complex_graphs)==0:
            return None,None


        return complex_graphs[0], ligs[0]
    
    def get_pocket_rmsd(self, complex_graph):
        res_atoms_mask = complex_graph['receptor'].res_atoms_mask
        res_atoms_pos = complex_graph['receptor'].res_atoms_pos[res_atoms_mask]
        ref_res_atoms_pos = complex_graph['receptor'].ref_res_atoms_pos[res_atoms_mask]
        return torch.sqrt(torch.sum((res_atoms_pos-ref_res_atoms_pos)**2,dim=1)).mean().item()
        
    def compute_RT(self, data):
        # 传入的配体坐标如果是一个列表,那么从列表中抽取一个坐标
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
            
        data.tr_score = self.get_lig_translation(data)
        lig_rmsd = self.rmsd_test(x=data['ligand'].pos, y=data['ligand'].ref_pos)
        sota_lig_rmsd = self.rmsd_test(x=self.modify_lig_conformer(x=data['ligand'].pos, tr_update=data.tr_score), y=data['ligand'].ref_pos)
        # print(f'{data.name} lig rmsd/sota_rmsd: ', lig_rmsd.item(), sota_lig_rmsd.item())
        data.res_rot_score, data.res_tr_score = self.get_res_rotation_vector(data)
        
        pocket_rmsd = self.rmsd_test(x=data['receptor'].res_atoms_pos, 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()

        print(f'{data.name} pocket rmsd: ', pocket_rmsd)
        res_rmsd = self.pocket_rmsd(x=data['receptor'].res_atoms_pos, y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        
        sota_pocket_rmsd = self.rmsd_test(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask).item()
        
        aligned_res_rmsd = self.pocket_rmsd(x=self.modify_pocket_conformer2(data=data, tr_update=data.res_tr_score, rot_update=data.res_rot_score), 
                                    y=data['receptor'].ref_res_atoms_pos, mask=data['receptor'].res_atoms_mask)
        deca_list = [(x1-x2)/x1 for x1,x2 in zip(res_rmsd,aligned_res_rmsd) if x1>x2]
        deca = np.average(deca_list)
        rate = len(deca_list)/len(res_rmsd)
        
        data['receptor'].deca = deca
        data['receptor'].rate = rate
        data['receptor'].sota_pocket_rmsd = sota_pocket_rmsd
        print('rate',rate, 'deca rate ', np.average(deca))
        
        # print(f'{data.name} esm-fold rmsd: ',data['receptor'].esm_rmsd)
        
        
        
        return data
        
    def get_lig_translation(self, data):
        assert data['ligand'].ref_pos.shape[0] == data['ligand'].pos.shape[0]
        lig_tr = data['ligand'].ref_pos - data['ligand'].pos
        return lig_tr # [lig_atoms,3]
    
    def get_res_translation(self, data):
        res_tr = data['receptor'].ref_pos - data['receptor'].pos
        return res_tr
    
    def get_res_rotation_vector(self, data):
        
        # atoms_mask [res, atoms]
        atoms_pos = data['receptor'].res_atoms_pos    # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        ref_atoms_pos = data['receptor'].ref_res_atoms_pos    # ground truth 所有原子坐标
        pos_mask = data['receptor'].res_atoms_mask
        
        rot_mat, tr_vec =self.point_cloud_to_ror_matrix(ref_atoms_pos, atoms_pos, pos_mask=pos_mask) #[res, 3, 3]
        rot_vector = self.R_batch_to_axis_vec(R=rot_mat) #[res, 3]
        return rot_vector, tr_vec
        
    def point_cloud_to_ror_matrix(self, pos, ref, pos_mask, pos_ca=None, ref_ca=None):
        # 两堆点云计算旋转矩阵
        # pos [N,M,3] [res, atoms, 3]
        # ref [N,M,3]
        # pos_mask [N,M]
        # pos_ca, ref_ca [N,3]
        # N : number of examples
        # M : number of atoms
        # R,T maps local reference onto global pos
        if pos_mask is None:
            pos_mask = torch.ones(pos.shape[:2], device=pos.device)
        else:
            if pos_mask.shape[0] != pos.shape[0]:
                raise ValueError("pos_mask should have same number of rows as number of input vectors.")
            if pos_mask.shape[1] != pos.shape[1]:
                raise ValueError("pos_mask should have same number of cols as number of input vector dimensions.")
            if pos_mask.ndim != 2:
                raise ValueError("pos_mask should be 2 dimensional.")
        
        # 使用点云中心为旋转中心
        denom = torch.sum(pos_mask, dim=1, keepdim=True)
        denom[denom == 0] = 1.
        pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
        pos_c = pos - pos_mu
        ref_c = ref - ref_mu
    
        # Covariance matrix
        H = torch.einsum('bji,bjk->bik', ref_c, pos_mask[:, :, None] * pos_c.to(torch.float32))
        U, S, Vh = torch.linalg.svd(H)
        # Decide whether we need to correct rotation matrix to ensure right-handed coord system
        locs = torch.linalg.det(U @ Vh) < 0
        S[locs, -1] = -S[locs, -1]
        U[locs, :, -1] = -U[locs, :, -1]
        # Rotation matrix
        R = torch.einsum('bji,bkj->bik', Vh, U)
        T = pos_mu - torch.einsum('bij,bkj->bki', R, ref_mu)
        return R,T.squeeze(1)
        
    def R_batch_to_axis_vec(self, R):
        """
        旋转矩阵到旋转向量
        [N,3,3] >>> [N,3]
        """
        J = (R - R.permute([0,2,1])) / 2
        t = torch.cat( [ (-J[:, 1, 2]).unsqueeze(1) , (J[:, 0, 2]).unsqueeze(1),  (-J[:, 0, 1]).unsqueeze(1)], dim=1)
        t_norm = torch.norm(t, dim=1, keepdim=True)
        theta = torch.asin(t_norm)
        angle = theta / np.pi * 180
        r = t / t_norm
        v = r * theta
        return v
    
    def rmsd_test(self, x, y, mask=None):
        # x y [res, atoms, 3]
        if mask is not None:
            x_ = x[mask]
            y_ = y[mask]
        else:
            x_ = x
            y_ = y
        # x y [atoms, 3]
        
        return torch.mean(torch.sqrt(torch.sum((x_ - y_)**2, dim=-1)))
        
    def pocket_rmsd(self, x,y,mask):
        rmsd = []
        for i in range(len(mask)):
            rmsd.append( self.rmsd_test(x[i],y[i],mask[i]).item() )
        # max_idx = rmsd.index(max(rmsd))
        # print('res max/mean rmsd ', max(rmsd), self.rmsd_test(x,y,mask))
        return rmsd
    
    def modify_lig_conformer(self, x, tr_update):
        return x + tr_update
        
    def modify_pocket_conformer(self, data, tr_update, rot_update):
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
        atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        return atoms_pos
        
    def modify_pocket_conformer2(self, data, tr_update, rot_update):
        # atoms_mask [res, atoms]
        atoms_pos = copy.deepcopy( data['receptor'].res_atoms_pos )   # 模型的输入:所有原子坐标 atoms_pos [res, atoms, 3] 
        pos_mask = data['receptor'].res_atoms_mask
        
        max_atom_num = atoms_pos.shape[1]
        
        # 做旋转
        # 通过轴向量拿到四元数再得到旋转矩阵
        rot_mat = axis_angle_to_matrix(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = vec_to_R(rot_update) # 轴角到旋转矩阵 [res,3,3]
        # rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        # atoms_pos[pos_mask] = torch.einsum('bij,bkj->bki',rot_mat[pos_mask].float(), atoms_pos[pos_mask].unsqueeze(1)).squeeze(1) 
        
        # 做平移
        # atoms_pos[pos_mask] = atoms_pos[pos_mask] + (tr_update.unsqueeze(1).repeat(1, max_atom_num, 1)[pos_mask]).to(torch.float32)
        
        # 平移旋转
        atoms_pos = torch.einsum('bij,bkj->bki',rot_mat.float(), atoms_pos) + tr_update.unsqueeze(1)
        
        return atoms_pos




def print_statistics(complex_graphs):
    statistics = ([], [], [], [])
    rate = []
    deca = []
    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)
        rate.append(complex_graph['receptor'].rate)
        deca.append(complex_graph['receptor'].deca)
        
    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    print('rate ', np.average(rate), 'deca ', np.average(deca))
    for i in range(4):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


     #TRLabel_Point_Cloud_Center()
def construct_loader(args, data_path=None, data_type='train', continuos=True, pretrain_method='pretrain_method1', save_pdb=False,max_align_rmsd=20,min_align_rmsd=0,cut_r=10):
    transform = None
    common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args.esm_embeddings_path}    

    dataset = CrossDockDataSet(data_type=data_type,pretrain_method=pretrain_method, save_pdb=save_pdb,
                                continuos=continuos, keep_original=True, require_ligand=True,  
                                max_align_rmsd=max_align_rmsd,cut_r=cut_r,data_path=data_path,
                                **common_args)
    
    loader_class = DataListLoader 
    loader = loader_class(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)
    
    return loader


def construct_uni_loader():
    from src.utils.utils import args_parse,get_abs_path
    from src.data.docking_pose_data_helper2 import load_dataset
    from src.data.unicore import Dictionary
    from src.data import get_dataloader
    cfg = args_parse('docking.yml')
    ligand_dict = Dictionary.load(get_abs_path('example_data/molecule/dict.txt'))
    pocket_dict = Dictionary.load(get_abs_path('example_data/pocket/dict_coarse.txt'))
    splits = ['train', 'valid']
    datasets = {split: load_dataset(cfg, split, ligand_dict, pocket_dict, from_list=True, use_H=True) for split in splits}
    dataloaders = get_dataloader(cfg, datasets)
    train_loader, val_loader = [dataloaders[split] for split in splits]
    train_loader.dataset.set_epoch(0)
    val_loader.dataset.set_epoch(0)
    return train_loader, val_loader


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=False)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=False)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            # 利用sdf读mol
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=False)
            # lig = None
            # 如果sdf不能读的话用mol2读
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=False)
            if lig is not None:
                ligs.append(lig)
    return ligs


def read_all_mols(pdbbind_dir, name, remove_hs=False):
    ligs = {}

    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        try:
            if file.endswith(".sdf") and 'rdkit' not in file:
                # 利用sdf读mol
                lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=False)
                if lig is not None:
                    ligs['sdf'] = lig
        except Exception as e:
            pass
            #print(e)
            
        try:
            if os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=False)
                if lig is not None:
                    ligs['mol2'] = lig
        except Exception as e:
            pass
            #print(e)
    return ligs


def collator(data_list):
    return data_list
