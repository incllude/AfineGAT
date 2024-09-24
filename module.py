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

    def sliding_attention(self, gru_output):

        batch_size, seq_len, hidden_size = gru_output.shape
        sliding_windows = []
        gru_output = torch.cat((torch.zeros(batch_size, self.T - 1, hidden_size), gru_output), dim=1)
        attention_scores = self.attention_layer(gru_output)
        
        for i in range(seq_len - self.W + 1):
            window = gru_output[:, i:i + self.T, :]
            window_attention_scores = attention_scores[:, i:i + self.T, :]
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * window, dim=1)
            sliding_windows.append(context_vector)
        
        sliding_context = torch.stack(sliding_windows, dim=1)
        return sliding_context
    
    def forward(self, x, category):
        
        category = x.pop("category")        
        gru_output = [self.gru(x[ticker])[0] for ticker in x]
        adjacency_matrix = torch.ones(len(x), len(x)) - torch.eye(len(x))
        
        temporal_embedding = torch.vmap(self.sliding_attention)(gru_output)
        graph_embedding = self.gat_layers[category](temporal_embedding, adjacency_matrix)
        
        score = self.fc(combined_embedding)
        return score
    
    def training_step(self, batch, batch_idx):
        
        x, edge_index, category, pairwise_labels = batch
        predictions = self(x, edge_index, category)
        
        pos_scores = predictions[pairwise_labels == 1]
        neg_scores = predictions[pairwise_labels == -1]
        loss = F.relu(1 - (pos_scores - neg_scores)).mean()
        
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return None
    