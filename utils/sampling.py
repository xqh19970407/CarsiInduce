import numpy as np
import torch
from torch_geometric.loader import DataLoader,DataListLoader
from datasets.pdbbind_pocket import Datalist_to_PDBBind
from utils.diffusion_utils import modify_conformer, set_time, set_listbatch_time, modify_ligand_pocket_conformer
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R
from Bio.PDB import Selection, PDBIO

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, no_rot=False):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            # 随机生成均匀分布,-pi 到 pi的扭转角
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate if isinstance(complex_graph['ligand'].mask_rotate, np.ndarray) else complex_graph['ligand'].mask_rotate[0],
                                                torsion_updates)

    for complex_graph in data_list:
        if not no_rot:
            # randomize position
            molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
            random_rotation = torch.from_numpy(R.random().as_matrix()).float()
            # 以配体为中心进行随机旋转
            complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
            # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update



def apply_euclidean(x, R):
    """
    R [..., 3, 3]
    T [..., 3]
    x [..., Na, 3]
    """
    Rx = torch.einsum('...kl,...ml->...mk', R, x)
    return Rx

def randomize_res_position(data_list, tr_sigma_max, tr_bool=True, rot_bool=True):
    """
    修改了全部口袋原子的坐标
    """
    for data in data_list:
        tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(data['receptor'].pos.shape[0], 3))
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
        
        rot_mat =  torch.from_numpy(np.array( [R.random().as_matrix() for _ in range(res_pos.shape[0])] ) ).float() # [res_num, 3, 3]
        # 向量特征的旋转更新   所有坐标绕CA旋转更新：更新前要减去CA的坐标, 再做旋转, 然后再把CA的坐标加回来
        data['receptor'].side_chain_vecs = apply_euclidean(data['receptor'].side_chain_vecs, rot_mat.float())       # [res_num, 4, 3] 4个向量特征
        data['receptor'].res_atoms_pos[res_atoms_mask] = data['receptor'].res_atoms_pos[res_atoms_mask] - res_pos[res_atoms_mask] # [res_num*atoms, 3]
        rot_mat = rot_mat.unsqueeze(1).repeat(1, max_atom_num, 1, 1) # [res_num, atoms, 3, 3]
        data['receptor'].res_atoms_pos[res_atoms_mask] = apply_euclidean(data['receptor'].res_atoms_pos[res_atoms_mask].unsqueeze(1), rot_mat[res_atoms_mask].float()).squeeze(1) + res_pos[res_atoms_mask]

        data['receptor'].rec = set_res_atom_pos(data, rec=rec, atoms_pos=data['receptor'].res_atoms_pos[res_atoms_mask])
    

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
    atoms = Selection.unfold_entities(rec, 'A')
    assert len(atoms)==len(atoms_pos)
    for idx, atom in enumerate(atoms):
        atom_pos = tuple(atoms_pos[idx].numpy())
        atom.set_coord(atom_pos)
    return rec




def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False, rot_bool=True, tr_bool=True, tor_bool=True):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]
        print(len(data_list))
        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                # 平移分数,旋转分数,扭转分数
                tr_score, rot_score, tor_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            if not model_args.no_torsion:
            # if not True:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None
            
            # Apply noise
            new_data_list.extend(
                
                [   modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
                                          tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if tor_bool else None)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())]
                         )
            # new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
            #                              None)
            #             for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        # 置信模型
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence

import time


def sampling_batch(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False, rot_bool=True, tr_bool=True, tor_bool=True):
    N = len(data_list)

    gpu_time = []
    modify_time = []
    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        # loader = DataLoader(data_list, batch_size=30) # xqup
        dataset = Datalist_to_PDBBind(data_list=data_list)
        b = len(dataset)
        loader = DataListLoader(dataset=dataset, batch_size=b, num_workers=0, shuffle=False)

        new_data_list = []
        
        for complex_graph_batch in loader:
            t1 = time.time()
            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_listbatch_time(complex_graph_batch, t_tr, t_rot, t_tor, model_args.all_atoms, device)
            
            with torch.no_grad():
                # 平移分数,旋转分数,扭转分数
                tr_score, rot_score, tor_score, res_tr_score, res_rot_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            if not model_args.no_torsion:
            # if not True:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                
            else:
                tor_perturb = None
            t2 = time.time()
            batch_torsion_num = [complex_graph['ligand'].edge_mask.sum().item() for i, complex_graph in enumerate(complex_graph_batch)]
            start_idx = [0] + [sum(batch_torsion_num[:idx]) for idx in range(1,len(batch_torsion_num))]
            end_ix = [sum(batch_torsion_num[:idx]) for idx in range(1,len(batch_torsion_num)+1)]
            # Apply noise
            new_data_list.extend(
                
                [modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
                                          tor_perturb[start_idx[i]:end_ix[i]] if tor_bool else None)
                         for i, complex_graph in enumerate(complex_graph_batch)]
                         )
            t3 = time.time()
            gpu_time.append(t2-t1)
            modify_time.append(t3-t2)
            # new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
            #                              None)
            #             for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list
        
        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)
    # print('gpu time ',np.sum(gpu_time))
    # print('modify time ',np.sum(modify_time))
    with torch.no_grad():
        # 置信模型
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence



def sampling_flexible_batch(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False, rot_bool=True, tr_bool=True, tor_bool=True):
    N = len(data_list)

    gpu_time = []
    modify_time = []
    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        # loader = DataLoader(data_list, batch_size=30) # xqup
        dataset = Datalist_to_PDBBind(data_list=data_list)
        b = len(dataset)
        loader = DataListLoader(dataset=dataset, batch_size=b, num_workers=0, shuffle=False)

        new_data_list = []
        
        for complex_graph_batch in loader:
            t1 = time.time()
            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_listbatch_time(complex_graph_batch, t_tr, t_rot, t_tor, model_args.all_atoms, device)
            
            with torch.no_grad():
                # 平移分数,旋转分数,扭转分数
                tr_score, rot_score, tor_score, res_tr_score, res_rot_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

                res_num = res_tr_score.shape[0]
                res_tr_z = torch.normal(mean=0, std=1, size=(res_num, 3))
                res_tr_perturb = (tr_g ** 2 * dt_tr * res_tr_score.cpu() + tr_g * np.sqrt(dt_tr) * res_tr_z).cpu()
                
                res_rot_z = torch.normal(mean=0, std=1, size=(res_num, 3))
                res_rot_perturb = (res_rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * res_rot_z).cpu()
                
            if not model_args.no_torsion:
            # if not True:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                
            else:
                tor_perturb = None
            t2 = time.time()
            batch_torsion_num = [complex_graph['ligand'].edge_mask.sum().item() for i, complex_graph in enumerate(complex_graph_batch)]
            start_idx = [0] + [sum(batch_torsion_num[:idx]) for idx in range(1,len(batch_torsion_num))]
            end_ix = [sum(batch_torsion_num[:idx]) for idx in range(1,len(batch_torsion_num)+1)]
            
            batch_res_num = [complex_graph['receptor'].pos.shape[0] for i, complex_graph in enumerate(complex_graph_batch)]
            res_start_idx = [0] + [sum(batch_res_num[:idx]) for idx in range(1,len(batch_res_num))]
            res_end_idx = [sum(batch_res_num[:idx]) for idx in range(1,len(batch_res_num)+1)]
            # Apply noise
            new_data_list.extend(
                [modify_ligand_pocket_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
                                          tor_perturb[start_idx[i]:end_ix[i]] if tor_bool else None,
                                          res_tr_update=res_tr_perturb[res_start_idx[i]:res_end_idx[i]],
                                          res_rot_update=res_rot_perturb[res_start_idx[i]:res_end_idx[i]],
                                          )
                         for i, complex_graph in enumerate(complex_graph_batch)]
                         )
            t3 = time.time()
            gpu_time.append(t2-t1)
            modify_time.append(t3-t2)
            # new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1] * float(tr_bool), rot_perturb[i:i + 1].squeeze(0) * float(rot_bool),
            #                              None)
            #             for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list
        
        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)
    confidence = None

    return data_list, confidence