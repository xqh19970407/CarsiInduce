import copy
import os
import warnings

from rdkit import Chem


import numpy as np
import scipy.spatial as spa
import torch
from Bio.PDB import PDBParser
import Bio.PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs, RemoveAllHs
from rdkit.Geometry import Point3D
from scipy import spatial
from scipy.special import softmax
from torch_cluster import radius_graph


import torch.nn.functional as F

from datasets.conformer_matching import get_torsion_angles, optimize_rotatable_bonds
from utils.torsion import get_transformation_mask



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



biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 0)


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)


def rec_residue_featurizer(rec):
    feature_list = []
    for residue in rec.get_residues():
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1



def parse_receptor(pdbid, pdbbind_dir):
    rec = parsePDB(pdbid, pdbbind_dir)
    return rec


def parsePDB(pdbid, pdbbind_dir):
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    return parse_pdb_from_path(rec_path)

def parse_esm_PDB(pdbid, pdbbind_dir):
    rec_path = os.path.join(pdbbind_dir, f'{pdbid}_protein.pdb')
    return parse_pdb_from_path(rec_path)

def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', path)
        rec = structure[0]
    return rec

from Bio.PDB import NeighborSearch, Selection
from rdkit import Chem
import numpy as np




def extract_receptor_structure(rec, lig, lm_embedding_chains=None):
    # 获得配体相关的信息
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions() # 获取配体三维坐标
    center = copy.deepcopy(lig_coords)

    min_distances = []
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    feature_lists = []

    # 蛋白链 和 分子的交织信息。。。当我们换成口袋以后，该怎么交
    for i, chain in enumerate(rec):# 蛋白质对肽链循环，因为只有一条或者两条，其实循环并不多
        
        res_list = Selection.unfold_entities(chain, 'R')# 将肽链展开为残基级别
        # 判断残基个数和lm_embeding是否一致
        
        atoms = Selection.unfold_entities(chain, 'A') # 将肽链展开到原子级别
        ns = NeighborSearch(atoms) # 固定中点和半径，搜索半径范围内的模块(原子，残基，肽链)
        close_residues = []
        dist_threshold = 10
        for a in center:
            close_residues.extend(ns.search(a, dist_threshold, level='R'))
        close_residues = Selection.uniqueify(close_residues) # 去重残基
        pocket_res_idx = []
        for res in close_residues:
            pocket_res_idx.append(res_list.index(res))

        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        feature_list = []
        for res_idx, residue in enumerate(chain):# 肽链对残基(氨基酸循环)
            # 判断残基属于口袋 通过索引找到了lm_embed
            if residue in close_residues:
                if residue.get_resname() == 'HOH':
                    invalid_res_ids.append(residue.get_id())
                    continue

                # 残基名称的特征
                feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])

                residue_coords = []
                c_alpha, n, c = None, None, None
                for atom in residue:# 残基(氨基酸)对原子进行循环
                    if atom.name == 'CA': # 残基alpha碳原子
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N': # 残基N原子
                        n = list(atom.get_vector())
                    if atom.name == 'C': # 残基C原子
                        c = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector())) # 对残基(氨基酸)上的所有原子取出它的坐标添加到一个列表中
                # alphaC C N都不是None
                if c_alpha != None and n != None and c != None:
                    # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                    chain_c_alpha_coords.append(c_alpha) # 当前循环已经把 alpha c原子找到的时候就把它添加到alpha碳原子
                    chain_n_coords.append(n)
                    chain_c_coords.append(c)
                    chain_coords.append(np.array(residue_coords))# 把残基上的所有原子的坐标添加进去
                    count += 1 # 该条肽链上有效残基的个数
                else:# 该条肽链上无效残基的id
                    invalid_res_ids.append(residue.get_id()) # 当我们遍历氨基酸上的所有的原子,没有找到alpha_c, c, N的时候，这个残基的编号就会被标记
        for res_id in invalid_res_ids: 
            chain.detach_child(res_id) # 删除肽链上那些无效的残基
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords) # 计算蛋白口袋肽链所有原子和小分子的距离矩阵
            min_distance = distances.min() # 计算蛋白口袋和小分子距离矩阵的最小值
        else:
            min_distance = np.inf

        min_distances.append(min_distance) # 每条肽链残基 距离配体的最小距离
        lengths.append(count) # 肽链上,口袋残基的个数
        coords.append(chain_coords) # 肽链上，口袋里，所有原子的坐标
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        feature_lists.append(np.array(feature_list))
        if not count == 0: valid_chain_ids.append(chain.get_id()) # 有效肽链的id

    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:# 
        valid_chain_ids.append(np.argmin(min_distances))# 当有效肽链的id是空时，选出那个和分子距离更近的那条肽链
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    valid_features = []

    for i, chain in enumerate(rec): # 蛋白对肽链循环
        
        res_list = Selection.unfold_entities(chain, 'R')# 将肽链展开为残基级别
        atoms = Selection.unfold_entities(chain, 'A') # 将肽链展开到原子级别
        ns = NeighborSearch(atoms) # 固定中点和半径，搜索半径范围内的模块(原子，残基，肽链)
        close_residues = []
        dist_threshold = 10
        for a in center:
            close_residues.extend(ns.search(a, dist_threshold, level='R'))
        close_residues = Selection.uniqueify(close_residues) # 去重残基
        pocket_res_idx = []
        for res in close_residues:
            pocket_res_idx.append(res_list.index(res))
        

        if chain.get_id() in valid_chain_ids:# 对有效肽链进行判断
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError('Encountered valid chain id that was not present in the LM embeddings')
                
                
                if int( lm_embedding_chains[i].shape[0] - len(res_list) )==0 :
                    valid_lm_embeddings.append(lm_embedding_chains[i][pocket_res_idx]) # 添加口袋残基的esm2 embedding
                    print('old seq')
                else:
                    # 将lm_embedding做个循环,然后判断特征长度和肽链长度能对的上,做个顺序轮换
                    for r in range(len(lm_embedding_chains)):
                        if int( lm_embedding_chains[r].shape[0] - len(res_list) )==0:
                            valid_lm_embeddings.append(lm_embedding_chains[r][pocket_res_idx]) # 添加口袋残基的esm2 embedding
                            print('new seq change successful')
                            break
               
            
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_features.append(feature_lists[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    # 将列表中按照肽链分组的坐标进行cat
    if (
        len(valid_c_alpha_coords) == 0 or 
        len(valid_n_coords) == 0 or 
        len(valid_c_coords) == 0 or
        len(valid_features) == 0 or
        len(valid_lm_embeddings) == 0 
    ):
        print('none list')

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3] 
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    feature_res = np.concatenate(valid_features, axis=0) # [n_residues, 1]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    for invalid_id in invalid_chain_ids: # 删除无效的肽链
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    # res 蛋白结构信息(删除了无效肽链), 
    # coords n_res个残基(氨基酸)的原子坐标 [[n_atoms,3]... ]   一万多个原子
    # c_alpha_coords alpha_c原子的坐标 [n_res,3] 
    # n_coords [n_res,3] 
    # c_coords [n_res,3]
    # lm_embeddings [n_res, 1280]
    return feature_res, coords, c_alpha_coords, n_coords, c_coords, lm_embeddings  

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



def detach_H2O_invalid_res(rec, invalid_chain_res_ids=None):
    # 去掉水分子和不完整原子坐标的残基(与推理esm特征的前处理保持一致)
    if invalid_chain_res_ids is not None:
        for i,chain in enumerate(rec):
            for res_id in invalid_chain_res_ids[i]: 
                chain.detach_child(res_id) # 删除肽链上那些无效的残基
            
        return rec, invalid_chain_res_ids
    
    invalid_chain_res_ids = []
    for i,chain in enumerate(rec):
        invalid_res_ids = []
        for idx,residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
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
                if atom.name == 'O':# 残基O原子
                    o = list(atom.get_vector())
            if residue.resname != 'GLY':
                c_beta = copy.deepcopy(o) # 甘氨酸没有beta碳原子,使用O原子充当
            # 无效残基
            if not(c_alpha != None and n != None and c != None and c_beta != None and  o != None):
                invalid_res_ids.append(residue.get_id()) 
    
        for res_id in invalid_res_ids: 
            chain.detach_child(res_id) # 删除肽链上那些无效的残基
        invalid_chain_res_ids.append(invalid_res_ids)
    return rec, invalid_chain_res_ids


def assert_full_id_same(rec1, rec2):
    res_list1 = Selection.unfold_entities(rec1, 'R')
    res_list2 = Selection.unfold_entities(rec2, 'R')
    assert len(res_list1)==len(res_list2)
    
    for res1, res2 in zip(res_list1, res_list2):
        assert res1.resname == res2.resname
           
    for res1, res2 in zip(res_list1, res_list2):
        assert res1.full_id == res2.full_id
        
        
def extract_esmProtein_crystalProtein(rec, esm_rec, lig=None, lm_embedding_chains=None, dist_threshold=10):
    # 残基类型 feature_res, 
    # 所有原子坐标 coords, 
    # CA坐标 c_alpha_coords, 
    # N原子坐标 n_coords, 
    # C原子坐标 c_coords, 
    # beta-C原子坐标 c_beta_coords, 
    # O原子坐标 o_coords, 
    # 残基的esm特征 lm_embeddings, 
    # 对原始蛋白做一些前处理,去除一些原子不存在的残基再进行推理 rec, 
    # 口袋残基列表 close_residues_list, 
    # 口袋残基 selector, 
    # 口袋残基full_id res_chain_full_id_list
    
    rec, invalid_res_ids = detach_H2O_invalid_res(rec)
    esm_rec, _ = detach_H2O_invalid_res(esm_rec, invalid_chain_res_ids=invalid_res_ids)
    # print('rec', len(Selection.unfold_entities(rec, 'R')))
    # print('esm', len(Selection.unfold_entities(esm_rec, 'R')))
    # chain_len_list = [embedding.shape[0] for embedding in lm_embedding_chains]
    # rec_chain_len_list = get_chain_len(rec=rec)
    # esm_chain_len_list = get_chain_len(rec=esm_rec)
    assert_full_id_same(rec, esm_rec)
    return rec,esm_rec
# 对齐99%以上的残基名称
def get_chain_len(rec):
    chain_len = []
    for chain in rec:
        chain_len.append(len(Selection.unfold_entities(chain, 'R')))
    return chain_len
        

def extract_receptor_pocket_structure(rec, lig, lm_embedding_chains=None, dist_threshold=10):
    
    # 获得配体相关的信息
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions() # 获取配体三维坐标
    center = copy.deepcopy(lig_coords)

    min_distances = []
    coords = []
    c_alpha_coords = []
    c_beta_coords = []
    n_coords = []
    c_coords = []
    o_coords = []
    valid_chain_ids = []
    lengths = []
    feature_lists = []
    

    for i, chain in enumerate(rec):# 蛋白质对肽链循环
        # 确认肽链长度
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        chain_o_coords = []
        chain_c_beta_coords = []
        count = 0
        invalid_res_ids = []
        feature_list = []
        for res_idx, residue in enumerate(chain):# 肽链对残基(氨基酸循环)
            # 根据下面的判断删除一些无效的残基,并且将有效残基的信息添加到列表
            # ??????这个判断不应该加,先注释
            # if residue in close_residues:
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue

            # 残基名称转为数字
            feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])

            residue_coords = []
            c_alpha, n, c, c_beta, o = None, None, None, None, None
            if residue.resname != 'GLY':
                for atom in residue:# 残基(氨基酸)对原子进行循环
                    if atom.name == 'CA': # 残基alpha碳原子
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N': # 残基N原子
                        n = list(atom.get_vector())
                    if atom.name == 'C': # 残基C原子
                        c = list(atom.get_vector())
                    if atom.name == 'CB':# 残基beta碳原子
                        c_beta = list(atom.get_vector())
                    if atom.name == 'O':# 残基O原子
                        o = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector())) # 对残基(氨基酸)上的所有原子取出它的坐标添加到一个列表中
            else:
                for atom in residue:# 残基(氨基酸)对原子进行循环
                    if atom.name == 'CA': # 残基alpha碳原子
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N': # 残基N原子
                        n = list(atom.get_vector())
                    if atom.name == 'C': # 残基C原子
                        c = list(atom.get_vector())
                    if atom.name == 'O':# 甘氨酸没有beta碳原子,使用O原子充当
                        c_beta = list(atom.get_vector())
                    if atom.name == 'O':# 残基O原子
                        o = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector())) # 对残基(氨基酸)上的所有原子取出它的坐标添加到一个列表中
            
            # alphaC C N都不是None
            if c_alpha != None and n != None and c != None and c_beta != None and  o != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha) # 当前循环已经把 alpha c原子找到的时候就把它添加到alpha碳原子
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_c_beta_coords.append(c_beta)
                chain_o_coords.append(o)
                chain_coords.append(np.array(residue_coords))# 把残基上的所有原子的坐标添加进去
                count += 1 # 该条肽链上有效残基的个数
            else:# 该条肽链上无效残基的id
                invalid_res_ids.append(residue.get_id()) # 当我们遍历氨基酸上的所有的原子,没有找到alpha_c, c, N的时候，这个残基的编号就会被标记
    
        for res_id in invalid_res_ids: 
            chain.detach_child(res_id) # 删除肽链上那些无效的残基

        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0) # 把肽链上所有的原子坐标concat到一起
            distances = spatial.distance.cdist(lig_coords, all_chain_coords) # 计算蛋白口袋肽链所有原子和小分子的距离矩阵
            min_distance = distances.min() # 计算蛋白口袋和小分子距离矩阵的最小值
        else:
            min_distance = np.inf

        min_distances.append(min_distance) # 每条肽链上的原子 距离配体的最小距离
        lengths.append(count) # 肽链上,残基的个数
        coords.append(chain_coords) # 肽链上,所有原子的坐标
        c_alpha_coords.append(np.array(chain_c_alpha_coords)) # 按照肽链添加alpha C
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        o_coords.append(np.array(chain_o_coords))
        c_beta_coords.append(np.array(chain_c_beta_coords))
        feature_lists.append(np.array(feature_list))
        if not count == 0: valid_chain_ids.append(chain.get_id()) # 有效肽链的id

    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:# 
        valid_chain_ids.append(np.argmin(min_distances))# 当有效肽链的id是空时，选出那个和分子距离更近的那条肽链
    
    # t_2 = time.time()
    # print(t_2-t_1)
    
    valid_coords = []
    valid_c_alpha_coords = []
    valid_c_beta_coords = []
    valid_o_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    valid_features = []
    invalid_res_ids = []
    
    indices = []
    res_chain_full_id_list = []
    close_residues_list = []
    # 根据配体割口袋
    # 上面根据需要的原子坐标是否存在,去除了一些无效的残基,下面对去完残基的蛋白重新循环
    for i, chain in enumerate(rec): # 蛋白对肽链循环
        res_list = Selection.unfold_entities(chain, 'R')# 将肽链展开为残基级别(已经去过口袋上的无效残基)
        atoms = Selection.unfold_entities(chain, 'A') # 将肽链展开到原子级别
        ns = NeighborSearch(atoms) # 固定中点和半径，搜索半径范围内的模块(原子，残基，肽链)
        close_residues = []
        
        for a in center:
            close_residues.extend(ns.search(a, dist_threshold, level='R')) # 割口袋,筛选出残基
        close_residues = Selection.uniqueify(close_residues) # 去重残基
        
        close_residues_list += close_residues
        
        # 通过保存selector的方式割口袋,返回一些对象,存储到data[‘receptor’]的属性中
        pocket_res_idx = []
        for res in close_residues: # 将口袋中的残基编号找到
            if res.resname in residue_atoms.keys(): # 判断氨基酸是否是20种氨基酸
                pocket_res_idx.append(res_list.index(res))
                indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
                res_chain_full_id_list.append((res.full_id[2], res.full_id[3]))
        
        
        # 通过删除残基的方式割口袋
        # for pocket_res in res_list:
        #     if pocket_res not in close_residues:
        #         chain.detach_child(pocket_res.get_id())
        
       
        if chain.get_id() in valid_chain_ids:# 对有效肽链进行判断
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i][pocket_res_idx])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError('protein chain is not in LM embeddings')
                
                valid_lm_embeddings.append(lm_embedding_chains[i][pocket_res_idx]) # 添加口袋残基的esm2 embedding
                valid_c_beta_coords.append(c_beta_coords[i][pocket_res_idx])
                valid_o_coords.append(o_coords[i][pocket_res_idx])
                valid_n_coords.append(n_coords[i][pocket_res_idx])
                valid_c_coords.append(c_coords[i][pocket_res_idx])
                valid_features.append(feature_lists[i][pocket_res_idx])
                valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]
    
    selector = ResidueSelector(indices)

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3] 
    c_beta_coords = np.concatenate(valid_c_beta_coords, axis=0)
    o_coords = np.concatenate(valid_o_coords, axis=0)
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    feature_res = np.concatenate(valid_features, axis=0) # [n_residues, 1]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    
    # 删除无效的肽链 to do
    for invalid_id in invalid_chain_ids: 
        rec.detach_child(invalid_id)

    # res_name = [for res in res]

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert len(c_alpha_coords) == len(c_beta_coords)
    assert len(c_alpha_coords) == len(o_coords)
   
    
    # coords n_res个残基(氨基酸)的原子坐标 [[n_atoms,3]... ]   一万多个原子
    # c_alpha_coords alpha_c原子的坐标 [n_res,3] 
    # n_coords [n_res,3] 
    # c_coords [n_res,3]
    # lm_embeddings [n_res, 1280]
    # bio对象, 口袋残基list, selector, full_id_list
    # rec['A'][full_id[-1]].resname
    # chain,id = (full_id[2],full_id[3])
    return feature_res, coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, lm_embeddings, rec, close_residues_list, selector, res_chain_full_id_list




def extract_pocket_structure(rec, lig, dist_threshold=10):
    # 割口袋
    # 获得配体相关的信息
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions() # 获取配体三维坐标
    center = copy.deepcopy(lig_coords)

    min_distances = []
    coords = []
    c_alpha_coords = []
    c_beta_coords = []
    n_coords = []
    c_coords = []
    o_coords = []
    valid_chain_ids = []
    lengths = []
    feature_lists = []
    
    for i, chain in enumerate(rec):# 蛋白质对肽链循环
        # 确认肽链长度
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        chain_o_coords = []
        chain_c_beta_coords = []
        count = 0
        invalid_res_ids = []
        feature_list = []
        for res_idx, residue in enumerate(chain):# 肽链对残基(氨基酸循环)
            # 根据下面的判断删除一些无效的残基,并且将有效残基的信息添加到列表
            # ??????这个判断不应该加,先注释
            # if residue in close_residues:
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue

            # 残基名称转为数字
            feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname())])

            residue_coords = []
            c_alpha, n, c, c_beta, o = None, None, None, None, None
            if residue.resname != 'GLY':
                for atom in residue:# 残基(氨基酸)对原子进行循环
                    if atom.name == 'CA': # 残基alpha碳原子
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N': # 残基N原子
                        n = list(atom.get_vector())
                    if atom.name == 'C': # 残基C原子
                        c = list(atom.get_vector())
                    if atom.name == 'CB':# 残基beta碳原子
                        c_beta = list(atom.get_vector())
                    if atom.name == 'O':# 残基O原子
                        o = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector())) # 对残基(氨基酸)上的所有原子取出它的坐标添加到一个列表中
            else:
                for atom in residue:# 残基(氨基酸)对原子进行循环
                    if atom.name == 'CA': # 残基alpha碳原子
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N': # 残基N原子
                        n = list(atom.get_vector())
                    if atom.name == 'C': # 残基C原子
                        c = list(atom.get_vector())
                    if atom.name == 'O':# 甘氨酸没有beta碳原子,使用O原子充当
                        c_beta = list(atom.get_vector())
                    if atom.name == 'O':# 残基O原子
                        o = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector())) # 对残基(氨基酸)上的所有原子取出它的坐标添加到一个列表中
            
            # alphaC C N都不是None
            if c_alpha != None and n != None and c != None and c_beta != None and  o != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha) # 当前循环已经把 alpha c原子找到的时候就把它添加到alpha碳原子
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_c_beta_coords.append(c_beta)
                chain_o_coords.append(o)
                chain_coords.append(np.array(residue_coords))# 把残基上的所有原子的坐标添加进去
                count += 1 # 该条肽链上有效残基的个数
            else:# 该条肽链上无效残基的id
                invalid_res_ids.append(residue.get_id()) # 当我们遍历氨基酸上的所有的原子,没有找到alpha_c, c, N的时候，这个残基的编号就会被标记
    
        for res_id in invalid_res_ids: 
            chain.detach_child(res_id) # 删除肽链上那些无效的残基

        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0) # 把肽链上所有的原子坐标concat到一起
            distances = spatial.distance.cdist(lig_coords, all_chain_coords) # 计算蛋白口袋肽链所有原子和小分子的距离矩阵
            min_distance = distances.min() # 计算蛋白口袋和小分子距离矩阵的最小值
        else:
            min_distance = np.inf

        min_distances.append(min_distance) # 每条肽链上的原子 距离配体的最小距离
        lengths.append(count) # 肽链上,残基的个数
        coords.append(chain_coords) # 肽链上,所有原子的坐标
        c_alpha_coords.append(np.array(chain_c_alpha_coords)) # 按照肽链添加alpha C
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        o_coords.append(np.array(chain_o_coords))
        c_beta_coords.append(np.array(chain_c_beta_coords))
        feature_lists.append(np.array(feature_list))
        if not count == 0: valid_chain_ids.append(chain.get_id()) # 有效肽链的id

    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:# 
        valid_chain_ids.append(np.argmin(min_distances))# 当有效肽链的id是空时，选出那个和分子距离更近的那条肽链
    
    # t_2 = time.time()
    # print(t_2-t_1)
    
    valid_coords = []
    valid_c_alpha_coords = []
    valid_c_beta_coords = []
    valid_o_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    valid_features = []
    invalid_res_ids = []
    
    indices = []
    res_chain_full_id_list = []
    close_residues_list = []
    # 根据配体割口袋
    # 上面根据需要的原子坐标是否存在,去除了一些无效的残基,下面对去完残基的蛋白重新循环
    for i, chain in enumerate(rec): # 蛋白对肽链循环
        res_list = Selection.unfold_entities(chain, 'R')# 将肽链展开为残基级别(已经去过口袋上的无效残基)
        atoms = Selection.unfold_entities(chain, 'A') # 将肽链展开到原子级别
        ns = NeighborSearch(atoms) # 固定中点和半径，搜索半径范围内的模块(原子，残基，肽链)
        close_residues = []
        
        for a in center:
            close_residues.extend(ns.search(a, dist_threshold, level='R')) # 割口袋,筛选出残基
        close_residues = Selection.uniqueify(close_residues) # 去重残基
        
        close_residues_list += close_residues
        
        # 通过保存selector的方式割口袋,返回一些对象,存储到data[‘receptor’]的属性中
        pocket_res_idx = []
        for res in close_residues: # 将口袋中的残基编号找到
            if res.resname in residue_atoms.keys(): # 判断氨基酸是否是20种氨基酸
                pocket_res_idx.append(res_list.index(res))
                indices.append(full_id_to_idx(res.get_full_id(), res.get_resname()))
                res_chain_full_id_list.append((res.full_id[2], res.full_id[3]))
        
        
        # 通过删除残基的方式割口袋
        # for pocket_res in res_list:
        #     if pocket_res not in close_residues:
        #         chain.detach_child(pocket_res.get_id())
        
       
        if chain.get_id() in valid_chain_ids:# 对有效肽链进行判断
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i][pocket_res_idx])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError('protein chain is not in LM embeddings')
                
                valid_lm_embeddings.append(lm_embedding_chains[i][pocket_res_idx]) # 添加口袋残基的esm2 embedding
                valid_c_beta_coords.append(c_beta_coords[i][pocket_res_idx])
                valid_o_coords.append(o_coords[i][pocket_res_idx])
                valid_n_coords.append(n_coords[i][pocket_res_idx])
                valid_c_coords.append(c_coords[i][pocket_res_idx])
                valid_features.append(feature_lists[i][pocket_res_idx])
                valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]
    
    selector = ResidueSelector(indices)

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3] 
    c_beta_coords = np.concatenate(valid_c_beta_coords, axis=0)
    o_coords = np.concatenate(valid_o_coords, axis=0)
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    feature_res = np.concatenate(valid_features, axis=0) # [n_residues, 1]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    
    # 删除无效的肽链 to do
    for invalid_id in invalid_chain_ids: 
        rec.detach_child(invalid_id)

    # res_name = [for res in res]

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert len(c_alpha_coords) == len(c_beta_coords)
    assert len(c_alpha_coords) == len(o_coords)
   
    
    # coords n_res个残基(氨基酸)的原子坐标 [[n_atoms,3]... ]   一万多个原子
    # c_alpha_coords alpha_c原子的坐标 [n_res,3] 
    # n_coords [n_res,3] 
    # c_coords [n_res,3]
    # lm_embeddings [n_res, 1280]
    # bio对象, 口袋残基list, selector, full_id_list
    # rec['A'][full_id[-1]].resname
    # chain,id = (full_id[2],full_id[3])
    return feature_res, coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, lm_embeddings, rec, close_residues_list, selector, res_chain_full_id_list






def assert_res_seq(name1, name2):
    for a,b in zip(name1,name2):
        if a!=b:
            return False
    return True


def get_lig_graph(mol, complex_graph):
    
    try:
        atom_feats = lig_atom_featurizer(mol)
    except Exception as e:
        print('get feature faied')
        print(e)
        return None
    
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr
    return complex_graph

def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        #print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)

def get_lig_graph_with_matching(mol_, complex_graph, popsize, maxiter, matching, keep_original, num_conformers, remove_hs, sdf_path=''):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:#True 去H
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True, implicitOnly=True)
            mol_maybe_noh = Chem.RemoveAllHs(mol_maybe_noh, sanitize=True)
        if keep_original:# True 将初始位置添加到图中 # 置信模型需要，以及评价
            complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()
            complex_graph['ligand'].ref_pos = torch.from_numpy(mol_maybe_noh.GetConformer().GetPositions()).float()

        rotable_bonds = get_torsion_angles(mol_maybe_noh)# 获取可旋转键
        #if not rotable_bonds: print("no_rotable_bonds but still using it")
        
        # 生成构象
        for i in range(num_conformers): # num_conformers = 1
            
            
            mol_rdkit = copy.deepcopy(mol_)
            # 下面这行代码用来从分子结构中推断出分子中原子的手性。在有些情况下，mol文件中可能没有完整的手性信息，
            # 这时我们就需要通过其他方法来推断手性。这个函数会将已知的手性信息应用于分子中，例如分子中带有IR/R标签的手性中心、
            # 双键上的Z/E标签等等。其作用是给分子中的原子分配手性标签，从而在后续分析中更准确地反映出分子的拓扑结构和立体构型。
            
            # 通过sdf读取构象
            lig_pos = get_pos_from_sdf(pdbid=sdf_path)
            
            if remove_hs:
                mol_rdkit = RemoveHs(mol_rdkit, sanitize=True, implicitOnly=True)
                mol_rdkit = Chem.RemoveAllHs(mol_rdkit, sanitize=True)
                
            
            if lig_pos is None or (lig_pos.shape[0] !=complex_graph['ligand'].orig_pos.shape[0]):
                #  生成构象
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol_rdkit)
                mol_rdkit.RemoveAllConformers() # 删除ground构象
                mol_rdkit = AllChem.AddHs(mol_rdkit)
                # 生成构象   生成一堆构象
                generate_conformer(mol_rdkit)
                if remove_hs:
                    mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
                mol = copy.deepcopy(mol_maybe_noh) # 带ground truth的构象
                if False:
                    optimize_rotatable_bonds(mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
                    # 获取ground truth 的扭转角,然后给rdkit生成的构象设置二面角
                # 将生成的构象添加到mol中
                mol.AddConformer(mol_rdkit.GetConformer())
                rms_list = []
                if False:
                    # 对分子的所有构象进行对齐以确保它们在空间中的位置和方向相同
                    AllChem.AlignMolConformers(mol, RMSlist=rms_list)
                mol_rdkit.RemoveAllConformers()
                mol_rdkit.AddConformer(mol.GetConformers()[1])# 将生成的随机构象添加到mol_rdkit
                lig_pos = mol_rdkit.GetConformer().GetPositions()

            # 设置构象
            if i == 0:
                lig_coords = torch.from_numpy(lig_pos).float()
                complex_graph['ligand'].pos = lig_coords # 从sdf读取的构象
                get_lig_graph(mol_rdkit, complex_graph)
                    
            else:# 将生成的构象添加到图中，因为num=1,所以这个分支不会进来
                if torch.is_tensor(complex_graph['ligand'].pos):
                    complex_graph['ligand'].pos = [complex_graph['ligand'].pos]
                complex_graph['ligand'].pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())

    else:  # no matching
        complex_graph.rmsd_matching = 0
        if remove_hs: mol_ = RemoveHs(mol_)
        get_lig_graph(mol_, complex_graph)

    edge_mask, mask_rotate = get_transformation_mask(complex_graph)# 可扭转键信息
    complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    complex_graph['ligand'].mask_rotate = mask_rotate

    return

from scipy.spatial.transform import Rotation as R


def gen_conformers(mol, numConfs=10):
    _mol = copy.deepcopy(mol)
    Chem.AssignAtomChiralTagsFromStructure(_mol)
    Chem.AssignStereochemistry(_mol)
    _mol.RemoveAllConformers()
    _mol = Chem.AddHs(_mol)
    params = AllChem.ETKDGv2()
    params.useRandomCoords = True
    # params.useSmallRingTorsions = True
    params.numThreads = 0
    params.numConfs = numConfs
    AllChem.EmbedMultipleConfs(_mol, numConfs=params.numConfs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(_mol, numThreads=0)
    _mol = Chem.RemoveHs(_mol)
    return _mol

def get_lig_feature(mol_, complex_graph, keep_original, remove_hs, sdf_path='', data_type='train',pretrain_method='', lig_in_mol=None, dekois=False):
    if mol_ is None:
        return None
    mol_maybe_noh = copy.deepcopy(mol_)
    if remove_hs:#True 去H
        try:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True, implicitOnly=True)
        except Exception as e:
            print(e)
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=False, implicitOnly=True)
    
    if not dekois and keep_original:# True 将初始位置添加到图中 # 置信模型需要，以及评价
        complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()
        complex_graph['ligand'].ref_pos = torch.from_numpy(mol_maybe_noh.GetConformer().GetPositions()).float()
    lig_in_mol = None
    if sdf_path is not None:
        lig_in_mol = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)[0]
    if lig_in_mol is None:
        print('load ligand is none')
        lig_in_mol = copy.deepcopy(mol_)
    
    lig_in_mol.RemoveAllConformers()
    gen_mol = gen_conformers(lig_in_mol, numConfs=1)
    lig_pos = gen_mol.GetConformer().GetPositions()

    if lig_pos is None or (lig_pos.shape[0] !=complex_graph['ligand'].orig_pos.shape[0]):
        print('lig_pos is None or (lig_pos.shape[0] != orig_pos.shape[0])')
        lig_pos = mol_maybe_noh.GetConformer().GetPositions()

    if mol_.GetNumAtoms() != lig_pos.shape[0]:
        lig_pos = mol_maybe_noh.GetConformer().GetPositions()
        
    lig_coords = torch.from_numpy(lig_pos).float()
    complex_graph['ligand'].pos = lig_coords  # 初始构象
    # if pretrain_method=='pretrain_method1':
    # 设置构象,加入随机旋转
    molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
    random_rotation = torch.from_numpy(R.random().as_matrix()).float()
    # 以配体为中心进行随机旋转
    complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T + molecule_center
    
    
    complex_graph = get_lig_graph(mol_maybe_noh, complex_graph) # 获取配体特征
    if complex_graph is None:
        return None
    from torch_geometric.utils import to_networkx
    import networkx as nx
    G = to_networkx(complex_graph.to_homogeneous(), to_undirected=False).to_undirected()
    complex_graph.is_connected = nx.is_connected(G)
    return complex_graph


def get_infer_lig_feature(mol_, complex_graph, keep_original, remove_hs, sdf_path='', data_type='train',pretrain_method='', lig_in_mol=None):
    
    mol_maybe_noh = copy.deepcopy(mol_)
    if remove_hs:#True 去H
        try:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True, implicitOnly=True)
        except Exception as e:
            print(e)
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=False, implicitOnly=True)
    
    if keep_original:# True 将初始位置添加到图中 # 置信模型需要，以及评价
        complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()
        complex_graph['ligand'].ref_pos = torch.from_numpy(mol_maybe_noh.GetConformer().GetPositions()).float()

    # 通过sdf读取构象
    lig_pos = get_pos_from_sdf(pdbid=sdf_path)
    
    if lig_pos is None or (lig_pos.shape[0] !=complex_graph['ligand'].orig_pos.shape[0]):
        
        if lig_pos is not None:
            if lig_pos.shape[0] !=complex_graph['ligand'].orig_pos.shape[0]:
                print('lig sdf shape is not matching with out')
                if data_type=='test':
                    return None
        lig_pos = mol_maybe_noh.GetConformer().GetPositions()


    lig_coords = torch.from_numpy(lig_pos).float()
    complex_graph['ligand'].pos = lig_coords  # 初始构象
    # if pretrain_method=='pretrain_method1':
    # 设置构象,加入随机旋转
    molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
    random_rotation = torch.from_numpy(R.random().as_matrix()).float()
    # 以配体为中心进行随机旋转
    complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T + molecule_center
    
    
    complex_graph = get_lig_graph(mol_maybe_noh, complex_graph) # 获取配体特征
    if complex_graph is None:
        return None
    from torch_geometric.utils import to_networkx
    import networkx as nx
    G = to_networkx(complex_graph.to_homogeneous(), to_undirected=False).to_undirected()
    complex_graph.is_connected = nx.is_connected(G)
    return complex_graph


def get_pos_from_sdf(pdbid):
    if not os.path.exists(pdbid):
        print('sdf is not exist')
        return None
    # 创建一个空的NumPy数组列表
    conformers_list = []
    try:
        suppl = Chem.SDMolSupplier(pdbid)
        # 遍历每个分子
        for mol in suppl:
            # 检查分子是否有效
            if mol is not None:
                mol = Chem.RemoveAllHs(mol)
                # 获取分子的构象数
                num_conformers = mol.GetNumConformers()
                # 遍历每个构象
                for i in range(num_conformers):
                    # 获取构象的坐标
                    conformer = mol.GetConformer(i)
                    coordinates = conformer.GetPositions()
                    # 将坐标添加到列表中
                    conformers_list.append(coordinates)
    except Exception as e:
        print(e)
        
    if len(conformers_list)==0:
        return None
    
    random_idx = np.random.randint(0,10,size=1)[0]
    if len(conformers_list)==1:random_idx=0
    
    return conformers_list[random_idx]
import time

def get_calpha_graph(feature_res, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, complex_graph, 
                     rec=None, res_list=None, selector=None, res_chain_full_id_list=None, cutoff=20, max_neighbor=None, lm_embeddings=None):
    
    
    # cutoff 受体残基 和 受体残基的临界值为 15A
    # max_neighbor 每个残基的最大临接数为 24
    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    c_beta_rel_pos = c_beta_coords - c_alpha_coords
    o_rel_pos = o_coords - c_alpha_coords
    
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")
    
    # Build the k-NN graph
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords) #氨基酸alpha C的距离矩阵    该函数最
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
    node_feat = torch.tensor(feature_res, dtype=torch.float32)  # [n_res, 1]
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1), 
                        np.expand_dims(c_beta_rel_pos, axis=1), np.expand_dims(o_rel_pos, axis=1)], axis=1))
    

    # 口袋蛋白的bio对象
    # t1 = time.time()
    # complex_graph['receptor'].rec = rec
    # complex_graph['receptor'].orig_rec = copy.deepcopy(rec)
    # t2 = time.time()
    # print(t2-t1)
    
    complex_graph['receptor'].selector = selector
    complex_graph['receptor'].res_chain_full_id_list = res_chain_full_id_list

    # 本来是从口袋bio对象获取残基个数,残基的最大原子个数,我们目前通过残基列表获取
    # 或者通过残基full_id来对所有残基循环进行获取
    
    res_num, max_atom_num = get_pocket_infor(res_list)

    
    assert res_num==num_residues
    res_atoms_mask = torch.zeros([res_num, max_atom_num], dtype=torch.bool)
    res_atoms_pos = torch.zeros([res_num, max_atom_num, 3])
    # res_list 得到所有坐标 seletor存储所有的口袋残基,残基序列列表,蛋白结构对象,残基对象的原子序列
    # 如何设置坐标?如何保存pdb? 通过残基的selector对象的full_id取出残基对象,然后通过循环残基序列对原子坐标设置
    # 将所有原子的坐标排序,再把原子

    ref_sorted_atom_names = []
    for idx, res in enumerate(res_list):
        res_coord, res_sorted_atom_names = res_to_sorted_coord_detachH(res)
        res_atom_num = len(res_sorted_atom_names)
        res_atoms_pos[idx, :res_atom_num] = res_coord
        res_atoms_mask[idx, :res_atom_num] = True
        ref_sorted_atom_names.append(res_sorted_atom_names)
    

    complex_graph['receptor'].res_atoms_mask = res_atoms_mask # [N_res,]
    complex_graph['receptor'].ref_res_atoms_pos = res_atoms_pos.float() # [N_res, atoms, 3]
    complex_graph['receptor'].ref_sorted_atom_names = ref_sorted_atom_names 
    # 氨基酸为粒度， 氨基酸的embdeing和氨基酸种类 [N_res, 1280] [N_res, 1]
    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1).float() if lm_embeddings is not None else node_feat
    # alpha_c的坐标
    complex_graph['receptor'].ref_pos = torch.from_numpy(c_alpha_coords).float()
    # [N_res,4,3]  先计算C原子相对alpha_C原子的位置差, N原子相对alpha_C原子的位置差，然后把它俩concat到一起得到
    complex_graph['receptor'].ref_side_chain_vecs = side_chain_vecs.float()
    # [2，N<255*24] 最大邻接数是24, src_list [0]*24 + [1]*24 + [2]*22 + .....      最大邻接数是24, dst_list [0距离小于15A的下标] + [1距离小于15A的下标].....  
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    return

def res_to_sorted_coord_detachH(res):
    # 就算残基通过各种对齐方式和esm-fold预测的结构做了对齐，依然有很多残基因为肽链id,full_id等不一致导致的无法对齐
    # 即使是晶体数据, 也有部分残基的原子不全
    # 而esm-fold预测的结构并不包含H原子
    # 为了得到对应的点云坐标,首先应该将H过滤掉,再根据晶体残基中实际含有的原子保存点云坐标(训练时)
    # 推理时,可以使用全原子的mask去推理,但其实没必要,因为晶体里没那些原子,推理出来也算不了rmsd,但我们应该支持这件事
    # 因此一套mask用来训练时计算lable,一套mask用来推理
    atom_list = sorted([(atom.name, atom) for atom in res if atom.name in residue_atoms[res.resname]], key=lambda x:x[0])
    atom_names = [name for name,atom in atom_list ]
    atom_coord = torch.tensor([list(atom.get_vector()) for name,atom in atom_list ])
    return atom_coord,atom_names
    
def res_to_sorted_coord(res, sorted_atom_names):
    atom_list = sorted([(atom.name, atom) for atom in res if atom.name in sorted_atom_names], key=lambda x:x[0])
    atom_names = [name for name,atom in atom_list ]
    assert same_names(sorted_atom_names, atom_names)
    atom_coord = torch.tensor([list(atom.get_vector()) for name,atom in atom_list ], dtype=torch.float32)
    return atom_coord

def same_names(name1, name2):
    if len(name1)!=len(name2):
        return False
    for n1,n2 in zip(name1,name2):
        if n1!=n2:
            return False
    return True
def add_rec_vector_infor(complex_graph, res_chain_full_id_list, pdb_rec, ref_sorted_atom_names):
    
    c_alpha_list = []
    n_list = []
    c_list = []
    c_beta_list = []
    o_list = []
    residue_atoms_pos = torch.zeros_like(complex_graph['receptor'].ref_res_atoms_pos)
    res_atoms_mask = complex_graph['receptor'].res_atoms_mask
    for idx, (chain,full_id) in enumerate( res_chain_full_id_list): # 根据full_id取出原始坐标信息
        residue = pdb_rec[chain][full_id]
        
        for atom in residue:# 残基(氨基酸)对原子进行循环
            if atom.name == 'CA': # 残基alpha碳原子
                c_alpha = list(atom.get_vector())
                c_alpha_list.append(c_alpha)
            if atom.name == 'N': # 残基N原子
                n = list(atom.get_vector())
                n_list.append(n)
            if atom.name == 'C': # 残基C原子
                c = list(atom.get_vector())
                c_list.append(c)
            if residue.resname == 'GLY' and atom.name == 'O':# 甘氨酸没有beta碳原子,使用O原子充当
                c_beta = list(atom.get_vector())
                c_beta_list.append(c_beta)
            if atom.name == 'CB':# 残基beta碳原子
                c_beta = list(atom.get_vector())
                c_beta_list.append(c_beta)
            if atom.name == 'O':# 残基O原子
                o = list(atom.get_vector())
                o_list.append(o)
            
        residue_atoms_pos[idx][res_atoms_mask[idx]] = res_to_sorted_coord(res=residue, sorted_atom_names=ref_sorted_atom_names[idx])
    
    
    n_rel_pos = np.array(n_list) - np.array(c_alpha_list)
    c_rel_pos = np.array(c_list)- np.array(c_alpha_list)
    c_beta_rel_pos =np.array(c_beta_list) - np.array(c_alpha_list)
    o_rel_pos = np.array(o_list) - np.array(c_alpha_list)

    side_chain_vecs = torch.from_numpy(
            np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1), 
                            np.expand_dims(c_beta_rel_pos, axis=1), np.expand_dims(o_rel_pos, axis=1)], axis=1))
                    
    complex_graph['receptor'].res_atoms_pos = residue_atoms_pos.float()
    complex_graph['receptor'].pos = torch.from_numpy(np.array(c_alpha_list)).float()
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    return

def get_pocket_infor(res_list,residue_atoms=residue_atoms):
    num_res = len(res_list)
    atom_num = [len(atoms_name) for (res_name,atoms_name) in residue_atoms.items()]
    return num_res, max(atom_num)




def rec_atom_featurizer(rec):
    atom_feats = []
    for i, atom in enumerate(rec.get_atoms()):
        atom_name, element = atom.name, atom.element
        if element == 'CD':
            element = 'C'
        assert not element == ''
        try:
            atomic_num = periodic_table.GetAtomicNumber(element)
        except:
            atomic_num = -1
        atom_feat = [safe_index(allowable_features['possible_amino_acids'], atom.get_parent().get_resname()),
                     safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                     safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                     safe_index(allowable_features['possible_atom_type_3'], atom_name)]
        atom_feats.append(atom_feat)

    return atom_feats


def get_rec_graph(feature_res, rec_coords, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, complex_graph, rec_radius, c_alpha_max_neighbors=None, all_atoms=False, 
                  rec=None, res_list=None, selector=None,  res_chain_full_id_list=None,
                  atom_radius=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    if all_atoms:
        return get_fullrec_graph(feature_res, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph,
                                 c_alpha_cutoff=rec_radius, c_alpha_max_neighbors=c_alpha_max_neighbors,
                                 atom_cutoff=atom_radius, atom_max_neighbors=atom_max_neighbors, remove_hs=remove_hs,lm_embeddings=lm_embeddings)
    else:
        return get_calpha_graph(feature_res, c_alpha_coords, n_coords, c_coords, c_beta_coords, o_coords, complex_graph, 
                                rec=rec, res_list=res_list, selector=selector, res_chain_full_id_list=res_chain_full_id_list,
                                cutoff=rec_radius, max_neighbor=c_alpha_max_neighbors, lm_embeddings=lm_embeddings)


def get_fullrec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, c_alpha_cutoff=20,
                      c_alpha_max_neighbors=None, atom_cutoff=5, atom_max_neighbors=None, remove_hs=False, lm_embeddings=None):
    # builds the receptor graph with both residues and atoms

    n_rel_pos = n_coords - c_alpha_coords
    c_rel_pos = c_coords - c_alpha_coords
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph of residues
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < c_alpha_cutoff)[0])
        dst.remove(i)
        if c_alpha_max_neighbors != None and len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(distances[i, :]))[1: c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(f'The c_alpha_cutoff {c_alpha_cutoff} was too small for one c_alpha such that it had no neighbors. '
                  f'So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert 1 - 1e-2 < weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec)
    mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    side_chain_vecs = torch.from_numpy(
        np.concatenate([np.expand_dims(n_rel_pos, axis=1), np.expand_dims(c_rel_pos, axis=1)], axis=1))

    complex_graph['receptor'].x = torch.cat([node_feat, torch.tensor(lm_embeddings)], axis=1) if lm_embeddings is not None else node_feat
    complex_graph['receptor'].pos = torch.from_numpy(c_alpha_coords).float()
    complex_graph['receptor'].mu_r_norm = mu_r_norm
    complex_graph['receptor'].side_chain_vecs = side_chain_vecs.float()
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))

    src_c_alpha_idx = np.concatenate([np.asarray([i]*len(l)) for i, l in enumerate(rec_coords)])
    atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec)))
    atom_coords = torch.from_numpy(np.concatenate(rec_coords, axis=0)).float()

    if remove_hs:
        not_hs = (atom_feat[:, 1] != 0)
        src_c_alpha_idx = src_c_alpha_idx[not_hs]
        atom_feat = atom_feat[not_hs]
        atom_coords = atom_coords[not_hs]

    atoms_edge_index = radius_graph(atom_coords, atom_cutoff, max_num_neighbors=atom_max_neighbors if atom_max_neighbors else 1000)
    atom_res_edge_index = torch.from_numpy(np.asarray([np.arange(len(atom_feat)), src_c_alpha_idx])).long()

    complex_graph['atom'].x = atom_feat
    complex_graph['atom'].pos = atom_coords
    complex_graph['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
    complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

    return

def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    w.write(mol)
    w.close()

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        # print(e)
        # print("RDKit was unable to read the molecule.")
        return None

    return mol




def read_sdf_to_mol_list(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):

    supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
    mol_list = []
    mol = supplier[0]
    
    for mol in supplier:
        try:
            if sanitize or calc_charges:
                Chem.SanitizeMol(mol)

            if calc_charges:
                # Compute Gasteiger charges on the molecule.
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                except:
                    warnings.warn('Unable to compute charges for the molecule.')

            if remove_hs:
                mol = Chem.RemoveHs(mol, sanitize=sanitize)
        except Exception as e:
            continue
        mol_list.append(mol)

    return mol_list







def read_sdf_or_mol2(sdf_fileName, mol2_fileName):

    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            problem = False
        except Exception as e:
            problem = True

    return mol, problem
