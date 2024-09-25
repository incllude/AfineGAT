import torch.nn.functional as F
from models import GAT
import lightning as l
import torch.nn as nn
import torch


class StockRankingModel(l.LightningModule):
    def __init__(self, sectors, input_size, hidden_size, gat_out_size, num_categories, T):
        super(StockRankingModel, self).__init__()
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.T = T
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                             nn.Tanh())
        self.gat_layers = nn.ModuleDict({
            sector: GAT(hidden_size, gat_out_size) for _ in range(num_categories)
            for sector in sectors
        })
        self.fc = nn.Linear(hidden_size + gat_out_size, 1)

    def sliding_attention(self, gru_embedding):

        sliding_windows = []
        batch_size, seq_len, hidden_size = gru_embedding.shape
#         gru_embedding = torch.cat((torch.zeros(batch_size, self.T - 1, hidden_size), gru_embedding), dim=1)
        scores = torch.vmap(self.attention_layer)(gru_embedding)
        
        for i in range(seq_len - self.W + 1):
            window = gru_embedding[:, i : i + self.T, :]
            window_scores = scores[:, i : i + self.T, :]
            weights = F.softmax(window_scores, dim=1)
            attentioned = torch.sum(weights * window, dim=1)
            sliding_windows.append(attentioned)
        
        sliding_context = torch.stack(sliding_windows, dim=1)
        return sliding_context
    
    def forward(self, x, category):
        
        category = x.pop("category")
        nodes_batched = torch.stack(list(x.values()), dim=0)
        gru_embedding = self.gru(nodes_batched)
        
        temporal_embedding = self.sliding_attention(gru_embedding)
        temporal_embedding = temporal_embedding.permute(1, 0, 2)
        adjacency_matrix = torch.ones(len(x), len(x)) - torch.eye(len(x))
        graph_embedding = self.gat_layers[category](temporal_embedding, adjacency_matrix)
        
        concated_embedding = torch.cat((gru_embedding, graph_embedding), dim=-1)
        score = self.fc(concated_embedding)
        return score
    
    def training_step(self, batch):
        pass
    
    def configure_optimizers(self):
        pass
    