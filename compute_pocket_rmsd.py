
import warnings
from Bio.PDB import PDBParser
from Bio.PDB import Selection

import numpy as np
import torch

import csv


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

def read_txt_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行末尾的换行符并添加到列表中
            lines.append(line.rstrip('\n'))
    return lines

def get_structure(path=''):
    # 该函数读取蛋白质
    warnings.filterwarnings("ignore")
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', path)
    structure = structure[0]
    return structure

def save_csv(path:str, head:list, data_list:list):
    # 创建 CSV 文件并写入数据
    
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头
        writer.writerow(head)
        
        # 写入数据
        writer.writerows(data_list)

    print(f"CSV文件已保存,{len(data_list)}条， 位置在", path)


def get_names_and_coord(path=''):
    structure = get_structure(path=path)
    res_list = []
    all_coord_list_dict = []
    for res in Selection.unfold_entities(structure, 'R'):
        res_list.append(three_to_one[ res.get_resname() ])
        atom_coord = {}
        for atom in res:
            if atom.name in residue_atoms[res.get_resname()]:
                atom_coord[atom.name] = np.array( list(atom.get_vector()) )
        all_coord_list_dict.append(atom_coord)
    return ''.join(res_list), all_coord_list_dict

def get_same_coord(esm_atom_coord,ground_atom_coord):
    esm_coord_list = []
    ground_coord_list = []
    esm_ca_list = []
    ground_ca_list = []
    for res_dict_ground, res_dict_esm in zip (ground_atom_coord, esm_atom_coord):
        for name in res_dict_ground.keys():
            if name in res_dict_esm.keys():
                esm_coord_list.append(res_dict_esm[name])
                ground_coord_list.append(res_dict_ground[name])
                if name=='CA':
                    esm_ca_list.append(res_dict_esm[name])
                    ground_ca_list.append(res_dict_ground[name])
    return np.array(esm_coord_list),np.array(ground_coord_list),np.array(esm_ca_list),np.array(ground_ca_list)


rmsd_list = []
pdbids = read_txt_lines('./example_data/posebusters_esmfold/posebusters.txt')
rmsd_list = []
dis_list = []
ca_rmsd_list = []
ca_dis_list = []


induce_rmsd_list = []
induce_dis_list = []
induce_ca_rmsd_list = []
induce_ca_dis_list = []


head = ['pdbid', 'all atom esm_RMSD',  'all atom induce_RMSD'  ,'all atom esm_MED',  'all atom induce_MED' ,'CA esm_RMSD','CA induce_RMSD', 'CA MED', 'CA induce_MED']
data = []
for idx,pdbid in enumerate (pdbids):
    try:
        esm_pocket = f'./out_file/esmFold_posebusters_esmfold_prepared/esm_pocket/{pdbid}_pocket_10A.pdb'
        ground_pocket = f'./out_file/esmFold_posebusters_esmfold_prepared/ground_pocket/{pdbid}_pocket_10A.pdb'
        induce_pocket = f'./out_file/esmFold_posebusters_esmfold_prepared/induced_pocket/{pdbid}_pocket_10A.pdb'
        
        esm_res_names, esm_atom_coord = get_names_and_coord(path=esm_pocket)
        ground_res_names, ground_atom_coord_ = get_names_and_coord(path=ground_pocket)
        induce_res_names, induce_atom_coord = get_names_and_coord(path=induce_pocket)
        
        assert esm_res_names==ground_res_names
        esm_atom_coord,ground_atom_coord, esm_ca_coord,ground_ca_coord = get_same_coord(esm_atom_coord,ground_atom_coord_)
        rmsd = np.sqrt (np.mean (np.sum ((esm_atom_coord-ground_atom_coord)**2, axis=-1)) )
        dis = np.mean (np.sqrt (np.sum ((esm_atom_coord-ground_atom_coord)**2, axis=-1)) )
        ca_rmsd = np.sqrt (np.mean (np.sum ((esm_ca_coord-ground_ca_coord)**2, axis=-1)) )
        ca_dis = np.mean (np.sqrt (np.sum ((esm_ca_coord-ground_ca_coord)**2, axis=-1)) )
        
        
        assert induce_res_names==ground_res_names
        induce_atom_coord,ground_atom_coord, induce_ca_coord,ground_ca_coord = get_same_coord(ground_atom_coord_, induce_atom_coord)
        induce_rmsd = np.sqrt (np.mean (np.sum ((induce_atom_coord-ground_atom_coord)**2, axis=-1)) )
        induce_dis = np.mean (np.sqrt (np.sum ((induce_atom_coord-ground_atom_coord)**2, axis=-1)) )
        induce_ca_rmsd = np.sqrt (np.mean (np.sum ((induce_ca_coord-ground_ca_coord)**2, axis=-1)) )
        induce_ca_dis = np.mean (np.sqrt (np.sum ((induce_ca_coord-ground_ca_coord)**2, axis=-1)) )
        
        rmsd_list.append(rmsd)
        dis_list.append(dis)
        ca_rmsd_list.append(ca_rmsd)
        ca_dis_list.append(ca_dis)
        
        induce_rmsd_list.append(induce_rmsd)
        induce_dis_list.append(induce_dis)
        induce_ca_rmsd_list.append(induce_ca_rmsd)
        induce_ca_dis_list.append(induce_ca_dis)
        
        data.append ([pdbid, rmsd, induce_rmsd, dis,induce_dis, ca_rmsd,induce_ca_rmsd, ca_dis, induce_ca_dis])
    except Exception as e:
        print(e)
    


data.append (['mean', np.mean(rmsd_list), np.mean(induce_rmsd_list),np.mean(dis_list) ,np.mean(induce_dis_list) , np.mean(ca_rmsd_list),np.mean(induce_ca_rmsd_list), np.mean(ca_dis_list), np.mean(induce_ca_dis_list) ])


save_csv(path='./out_file/posebusters_pocket_induce_rmsd2.csv', head=head, data_list=data)
