
from argparse import ArgumentParser,FileType

def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--log_dir', type=str, default='/userdata/xiaoqi/workdir', help='Folder in which to save model and logs')
    parser.add_argument('--datatype', type=str, default='apodata_esmfold',help='')
    parser.add_argument('--restart_dir', type=str, help='')
    parser.add_argument('--cache_path', type=str, default='/userdata/xiaoqi/cache', help='Folder from where to load/restore cached dataset') # 文件保存路径
    parser.add_argument('--data_dir', type=str, default='/userdata/share/a100_1/data3/xiaoqi_data/diffdock_data/PDBBind_processed/', help='Folder containing original structures')
    parser.add_argument('--split_path', type=str, default='/userdata/xiaoqi/unimol_dataset.json', help='Path of file defining the split')
    
    parser.add_argument('--test_sigma_intervals', action='store_true', default=False, help='Whether to log loss per noise interval')
    parser.add_argument('--val_inference_freq', type=int, default=50, help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--train_inference_freq', type=int, default=None, help='Frequency of epochs for which to run expensive inference on train data')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps for inference on val')
    parser.add_argument('--num_inference_complexes', type=int, default=100, help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--inference_earlystop_metric', type=str, default='valinf_rmsds_lt2', help='This is the metric that is addionally used when val_inference_freq is not None')
    parser.add_argument('--inference_earlystop_goal', type=str, default='max', help='Whether to maximize or minimize metric')
    parser.add_argument('--wandb', action='store_true', default=True, help='')
    parser.add_argument('--project', type=str, default='CarsiDock2.0', help='')
    parser.add_argument('--run_name', type=str, default='pretrain_R50E36_esm2_finetune', help='')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=20, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')

    # Training arguments
    parser.add_argument('--gpu', type=int, default=0, help='input gpu id')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=30, help='Patience of the LR scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--restart_lr', type=float, default=None, help='If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.')
    parser.add_argument('--w_decay', type=float, default=0.0, help='Weight decay added to loss')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for preprocessing')
    parser.add_argument('--use_ema', action='store_true', default=False, help='Whether or not to use ema for the model weights')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='decay rate for the exponential moving average model parameters ')

    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='If positive, the number of training and validation complexes is capped')
    parser.add_argument('--all_atoms', action='store_true', default=False, help='Whether to use the all atoms model')
    parser.add_argument('--receptor_radius', type=float, default=50, help='Cutoff on distances for receptor edges') #50 15
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=36, help='Maximum number of neighbors for each residue')#36 24
    parser.add_argument('--atom_radius', type=float, default=5, help='Cutoff on distances for atom connections')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='Maximum number of atom neighbours for receptor')
    parser.add_argument('--matching_popsize', type=int, default=20, help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', action='store_true', default=True, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')
    parser.add_argument('--esm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')
    # "/userdata/xiaoqi/EsmFoldPredict/esm_lm_embeddings"
    # Diffusion
    parser.add_argument('--tr_weight', type=float, default=0.33, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=0.33, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=0.33, help='Weight of torsional loss')
    parser.add_argument('--rot_sigma_min', type=float, default=0.03, help='Minimum sigma for rotational component')
    parser.add_argument('--rot_sigma_max', type=float, default=1.55, help='Maximum sigma for rotational component')
    parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--tr_sigma_max', type=float, default=3, help='Maximum sigma for translational component')
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument('--no_torsion', action='store_true', default=False, help='If set only rigid matching')

    # Model
    parser.add_argument('--num_conv_layers', type=int, default=6, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=48, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=10, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=64, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=64, help='Embeddings size for the cross distance')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--sigma_embed_dim', type=int, default=0, help='Size of the embedding of the diffusion time')
    parser.add_argument('--embedding_scale', type=int, default=1000, help='Parameter of the diffusion time embedding')

    args = parser.parse_args()
    return args
