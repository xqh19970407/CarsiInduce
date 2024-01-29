import torch

class UnimolEmbeddingToScoreModel(torch.nn.Module):
    def __init__(self, uniconfig: UnimolConfig, scoreconfig) -> None:
        super().__init__()
        self.unimol = UniMolModel(config=uniconfig)
        self.score_model = TensorProductScoreModel(**scoreconfig)
        
    def forward(self,data):
        lm_embedding = self.unimol()
        padding_mask = data['net_input']['src_tokens'].ne(0) # bsz, seq_len
        lm_embedding_for_diff = lm_embedding[padding_mask] # total_len, 512
        data['recptor'].x = torch.cat([lm_embedding_for_diff, data['recptor'].x], dim=-1)
        