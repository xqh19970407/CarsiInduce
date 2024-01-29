from models.score_model import TensorProductScoreModel
from src.modeling.modeling_hf_unimol import UnimolConfig,UniMolModel
import torch
from torch_geometric.data import Batch
from torch import nn
from src.data.unicore import Dictionary
from src.utils.utils import get_abs_path


pocket_dict = Dictionary.load(get_abs_path('example_data/pocket/dict_coarse.txt'))
pocket_config = UnimolConfig(hidden_size=1280, num_hidden_layers=6, vocab_size=len(pocket_dict) + 1)


class UniMolEncode(nn.Module):
    def __init__(self, uniconfig=pocket_config) -> None:
        super().__init__()
        self.unimol = UniMolModel(config=uniconfig)
        # self.unimol.load_state_dict( torch.load('./unimol_pretrain/pocket_pre_220816.pt')['model'], strict=False)
        
    def forward(self, batch):
        
        pocket_outputs = self.unimol(src_tokens=batch['net_input']['pocket_src_tokens'], 
                                     src_distance=batch['net_input']['pocket_src_distance'],
                                     src_edge_type=batch['net_input']['pocket_src_edge_type'])
        lm_embeddings_batch = pocket_outputs.last_hidden_state
        
        pocket_tokens_batch = batch['net_input']['pocket_src_tokens']
        # 取出原子tokens 0123是padding cls eos 等
        pocket_tokens_batch = pocket_tokens_batch.ne(0) & pocket_tokens_batch.ne(1) & pocket_tokens_batch.ne(2) & pocket_tokens_batch.ne(3)
        # 对每个样本的图lm_embeding进行更新
        for i in range(len(batch['net_input']["pocket_lig_graph"])):
            pocket_tokens_idx = pocket_tokens_batch[i]
            pocket_tokens = (batch['net_input']['pocket_src_tokens'][i][pocket_tokens_idx]).view([-1, 1])
            lm_embeddings = lm_embeddings_batch[i][pocket_tokens_idx]
            assert len(lm_embeddings)==len(pocket_tokens)
            # 氨基酸为粒度， 氨基酸的embdeing和氨基酸种类 [N_res, 1280] [N_res, 1]
            batch['net_input']["pocket_lig_graph"][i]['receptor'].x = torch.cat([pocket_tokens.to(lm_embeddings.device), lm_embeddings], axis=1)
        # 将Unimol编码好的图列表返回
        return batch['net_input']["pocket_lig_graph"]

class UniScoreModel(nn.Module):
    def __init__(self, scoreconfig) -> None:
        super().__init__()
        self.unimol_encode = UniMolEncode()
        self.score_model = TensorProductScoreModel(**scoreconfig)
        
    def forward(self, batch):
        graph_batch_list = self.unimol_encode(batch)
        graph_batch = Batch.from_data_list(graph_batch_list)
        return self.score_model(graph_batch)