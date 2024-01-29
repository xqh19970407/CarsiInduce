import binascii
import glob
import hashlib
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from datasets.process_mols2 import read_molecule, get_rec_graph, generate_conformer, get_calpha_graph,\
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt
from utils import so3, torus


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def __call__(self, data):
        t = np.random.uniform() # 均匀分布
        t_tr, t_rot, t_tor = t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor)

    def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update = None, rot_update=None, torsion_updates=None):
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor) # sigma_min^(t) * sigma_max^(t)
        set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates
        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)

        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        return data

    
class PDBBind(Dataset):
    def __init__(self,  dataset, init_dataset=None, graph_path='', root='', transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False):

        self.dataset = dataset # 蛋白信息和小分子smi从这里来
        self.complex_graphs = None
        if os.path.exists(graph_path):
            with open(graph_path,'rb') as f:
                self.complex_graphs = pickle.load(f)
        super(PDBBind, self).__init__(root, transform)
        self.init_dataset = init_dataset
        self.pdbbind_dir = root
        self.max_lig_size = max_lig_size
        self.split_path = split_path
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
        
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        
       
    def len(self):
        if self.dataset:
            return len(self.dataset)
        elif self.complex_graphs:
            return len(self.complex_graphs)

    def get(self, idx):
        if self.require_ligand:
            pass
            # complex_graph = copy.deepcopy(self.complex_graphs[idx])
            # complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
            # return complex_graph
        else:
            if not self.complex_graphs:
                return copy.deepcopy(self.get_complex_graph(idx))
            else:
                return copy.deepcopy(self.complex_graphs[idx])
                
    def get_complex_graph(self, idx):
        data = self.dataset[idx]
        name, lm_embedding_chains, ligand, ligand_description = data['pocket'], None, None, None
        if self.init_dataset:
            lig = self.init_dataset[idx]['holo_mol'] # 配体坐标mol
        else:
            lig = data['holo_mol']
    
        complex_graph = HeteroData()# 定义异构图
        complex_graph['name'] = name
        
        # 在图中添加配体信息,生成随机构象
        get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                            self.num_conformers, remove_hs=self.remove_hs)
        
        # 在图中添加残基>>>口袋原子相关的信息
        # 返回蛋白相关的信息，坐标，embeding等
        # 对graph添加embedding的信息，坐标信息，远近的信息
        get_calpha_graph(data, complex_graph, cutoff=self.receptor_radius, max_neighbor=self.c_alpha_max_neighbors, 
                         lm_embeddings=None)
        
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        if (not self.matching) or self.num_conformers == 1:
            complex_graph['ligand'].pos -= protein_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= protein_center

        complex_graph.original_center = protein_center
        return complex_graph
    
    # def reset_offset(self):
    #     pass
        
from src.data.unicore.base_wrapper_dataset import BaseWrapperDataset
from functools import lru_cache

class GraphPad(BaseWrapperDataset):
    def __init__(self, dataset, *args, **kwargs):
        super(GraphPad, self).__init__(dataset, *args, **kwargs)
        self.dataset = dataset
    
    def collater(self, samples):
        
        return samples

class ClearDataset(BaseWrapperDataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        with open(f'/mnt/d/diffdock_data/offset_list_{split}_v3','rb') as f:
            self.offset_list = pickle.load(f)

    def __len__(self):
        return len(self.dataset)-len(set(self.offset_list))+1

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx + self.offset_list[idx]]
    

class ClearDataset2(BaseWrapperDataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        with open(f'/home/hxq2/DiffDock/data/protein_ligand_binding_pose_prediction/{split}_bad.txt') as f:
            self.bad_case_name = f.read().splitlines()
        self.offset = 0
        self.split = split
        self.offset_list = []
        

    def __len__(self):
        return len(self.dataset)-len(self.bad_case_name)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        _offset = 0
        while self.dataset[idx+_offset+self.offset]['pocket'] in self.bad_case_name :
            _offset += 1
        self.offset += _offset

        self.offset_list.append(self.offset)
        return self.dataset[idx]


    
def print_statistics(complex_graphs):
    statistics = ([], [], [], [])

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

    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    for i in range(4):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


def construct_loader(args, t_to_sigma):
    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms)

    common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args.esm_embeddings_path}

    train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                            num_conformers=args.num_conformers, **common_args)
    val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)
    

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)

    return train_loader, val_loader


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            # 利用sdf读mol
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            # 如果sdf不能读的话用mol2读
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs

def collator(data_list):
    return data_list