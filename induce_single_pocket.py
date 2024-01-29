import copy
import math
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
from datasets.pdbbind_pocket import construct_loader
from utils.parsing import parse_train_args
from utils.utils import  get_model
import numpy as np
import random
from datasets.pdbbind_pocket import full_id_to_idx,ResidueSelector
import csv

def set_seed(seed=0):
    from torch.backends import cudnn
    cudnn.benchmark = False            # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    
    random.seed(seed)
    np.random.seed(seed)




def infer_batch_to_rmsd_list(model, data, device, save_predict=False):
    # 返回rmsd的列表
    
    with torch.no_grad():
        tr_pred, res_tr_pred, res_rot_pred, pocket_atoms_pos, ligand_atoms_pos = model(data)
    ca_rmsd = [d['receptor'].ca_rmsd for d in data]
    # ligand_AtomNum_list = [d['ligand'].ref_pos.shape[0] for d in data] # 每个配体的原子个数
    # ligand_ground_pos = torch.cat([d['ligand'].ref_pos for d in data], dim=0) # [bsz*atoms, 3] 
    pocket_ResNum_list = [d['receptor'].ref_res_atoms_pos.shape[0] for d in data] # 每个蛋白口袋的残基数量
    pocket_AtomNum_list = [torch.sum( d['receptor'].res_atoms_mask).item() for d in data]
    pocket_ground_pos = torch.cat([d['receptor'].ref_res_atoms_pos for d in data], dim=0) # [bsz*res, 14, 3] 
    pocket_mask = torch.cat([d['receptor'].res_atoms_mask for d in data], dim=0) # [bsz*res, 14] 
    init_pocket_rmsd_list = [d['receptor'].crossdock_rmsd for d in data] # 每个蛋白口袋的残基数量
    pdbids = [d.name for d in data]
    # 配体原子坐标rmsd loss
    # lig_rmsd = get_rmsd_list(ligand_atoms_pos, ligand_ground_pos, Num_list=ligand_AtomNum_list)
    # 残基原子坐标rmsd loss
    pocket_rmsd = get_rmsd_list(pocket_atoms_pos, pocket_ground_pos, mask=pocket_mask, Num_list=pocket_ResNum_list) 
    
    return pdbids,pocket_rmsd,init_pocket_rmsd_list,None,pocket_ResNum_list,pocket_AtomNum_list,ca_rmsd

def sum_first_n_elements(lst):
    result_list = []  # 创建一个空列表用于保存结果
    for n in range(1, len(lst) + 1):
        sum_result = sum(lst[:n])  # 对列表的前n个元素求和
        result_list.append(sum_result)  # 将求和结果添加到结果列表中
    return result_list

def get_rmsd_list(pos, ground_pos, mask=None, Num_list=None):
    rmsd_list = []
    
    start_idx = [0] + sum_first_n_elements(Num_list[:-1])
    end_idx = sum_first_n_elements(Num_list)
    
    # 对所有的配体和蛋白进行循环计算rmsd
    for i, (idx_s, idx_e) in enumerate( zip(start_idx, end_idx) ):
        if mask is None:
            rmsd =torch.sqrt ( torch.mean( torch.sum((pos[idx_s:idx_e]-ground_pos[idx_s:idx_e].to(pos.device))**2, dim=-1) ) ).to('cpu').item()
        else:
            mask_pocket = mask[idx_s:idx_e]
            from utils.geometry import rigid_transform_Kabsch_3D_torch
            in_coord = pos[idx_s:idx_e][mask_pocket].cpu()
            out_coord = ground_pos[idx_s:idx_e][mask_pocket].cpu()
            R, t = rigid_transform_Kabsch_3D_torch(in_coord.T, out_coord.T)
            aligned_in_coord = in_coord @ R.T + t.T
            
            rmsd =torch.sqrt (torch.mean ( torch.sum((aligned_in_coord-out_coord)**2, dim=-1) ) ).to('cpu').item()
        rmsd_list.append(rmsd)
    return rmsd_list
        

def save_csv(path:str, head:list, data_list:list):
    # 创建 CSV 文件并写入数据
    
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头
        writer.writerow(head)
        
        # 写入数据
        writer.writerows(data_list)

    print(f"CSV文件已保存,{len(data_list)}条， 位置在", path)


def inference(model, test_loader,device=None):
    print("Starting testing...")
    lig_rmsd_list,pocket_rmsd_list,init_pocket_rmsd_list = [],[],[]
    pocket_ResNum_lists,pocket_AtomNum_lists = [],[]
    ligand_AtomNum_lists=[]
    ca_rmsd_list = []
    all_pdbid = []
    for data in tqdm(test_loader, total=len(test_loader)):
        pdbids, pocket_rmsd, init_pocket_rmsd,ligand_AtomNum_list,pocket_ResNum_list,pocket_AtomNum_list,ca_rmsd = infer_batch_to_rmsd_list(model, data, device, save_predict=True)
        all_pdbid += pdbids 
        ca_rmsd_list += ca_rmsd
        pocket_rmsd_list+=pocket_rmsd
        init_pocket_rmsd_list+=init_pocket_rmsd
        pocket_ResNum_lists += pocket_ResNum_list
        pocket_AtomNum_lists += pocket_AtomNum_list
        idx_max,idx_min = np.argmax(init_pocket_rmsd_list),np.argmin(init_pocket_rmsd_list)
        
        print('test dataset size are: ', len(pocket_rmsd_list))
        print('--------------------------pocket---------------------------------')
        print('init ca rmsd', np.average(ca_rmsd_list))
        print('test pocket average init rmsd: ', np.average(init_pocket_rmsd_list))
        print('test pocket average rmsd: ', np.average(pocket_rmsd_list))
        print(f'max rmsd docking before/after {init_pocket_rmsd_list[idx_max]} / {pocket_rmsd_list[idx_max]}')
        print(f'min rmsd docking before/after {init_pocket_rmsd_list[idx_min]} / {pocket_rmsd_list[idx_min]}')
        print(f'average/max pocket res num are {np.average(pocket_ResNum_lists)}/{np.max(pocket_ResNum_lists)}')
        print(f'average/max pocket atoms num are {np.average(pocket_AtomNum_lists)}/{np.max(pocket_AtomNum_lists)}')
        print('test pocket rmsd<2A ', np.average( np.array(pocket_rmsd_list)<2 ) )
       
        
        head = ['pdbid', 'esmFold pocket RMSD', 'induced pocket RMSD']
        DATA = [[all_pdbid[i], init_pocket_rmsd_list[i], pocket_rmsd_list[i]] for i in range(len(pocket_rmsd_list))]
        save_csv(path='./out_file/esmFold_posebusters_esmfold_prepared/results.csv', data_list=DATA, head=head)
    
    
    
    
def main_function():
    args = parse_train_args()
    
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.batch_size = 1
    args.num_dataloader_workers = 0
    os.makedirs('./out_file', exist_ok=True)
    ground_protein = ''
    esmfold_protein = ''
    ground_ligand = ''
    data_path = [[esmfold_protein, ground_ligand, ground_protein, ground_ligand]]
    test_loader = construct_loader(args, data_path=data_path, data_type='single_pocket',pretrain_method=None, save_pdb=True, max_align_rmsd=20,cut_r=10,min_align_rmsd=0.0)  
    model = get_model(args, device=device, no_parallel=False, inference=True) 
    # 保存诱导前和诱导后，并且对齐的口袋
    load_moldel_path = './ckpt' 
    if load_moldel_path:
        dict = torch.load(f'{load_moldel_path}/step59000_model.pt', map_location=torch.device('cpu'))
        pretrained_dict = dict['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        pre_trained_num = sum([v.numel() for k,v in pretrained_dict.items()])
        print('load pretrain num: ',pre_trained_num)
        
    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    args.device = device
    inference(model, test_loader,device=device)


if __name__ == '__main__':
    # python inference_esmfold.py 
    set_seed()
    main_function()