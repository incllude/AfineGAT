from torchmetrics import RetrievalNormalizedDCG
from scipy.stats import rankdata
import torch.nn.functional as F
from models import GAT
import lightning as l
import torch.nn as nn
import numpy as np
import torch


class StockRankingModel(l.LightningModule):
    def __init__(self, structure, input_size, hidden_size, gat_out_size, T, val_cut):
        super(StockRankingModel, self).__init__()
        self.save_hyperparameters()
        
        self.structure = structure
        self.gat_out_size = gat_out_size
        self.hidden_size = hidden_size
        self.val_cut = val_cut
        self.T = T
        
        self.embedding_extractor = nn.Sequential(nn.Linear(input_size, 3 * hidden_size, bias=False),
                                                 nn.ReLU(),
                                                 nn.LazyBatchNorm1d(),
                                                 nn.Linear(3 * hidden_size, 2 * hidden_size, bias=False),
                                                 nn.ReLU(),
                                                 nn.Linear(2 * hidden_size, hidden_size, bias=False),
                                                 nn.ReLU())
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size, 1),
                                             nn.Tanh())
        self.gat_sectors_layers = nn.ModuleDict({
            sector: GAT(hidden_size, gat_out_size) for sector in structure
        })
        self.gat_world_layer = GAT(gat_out_size, gat_out_size)
        self.fusion = nn.Sequential(nn.Linear(hidden_size + gat_out_size * 2, hidden_size), nn.ReLU())
        self.rank = nn.Linear(hidden_size, 1)
        self.return_avg = nn.Linear(hidden_size, 1)
        self.return_prob = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        
        self.ndcg = RetrievalNormalizedDCG()
        self.metrics = {
            "Train Loss": [],
            "NDCG": [],
            "MAE of positive return prob": [],
            "MSE of avg return": []
        }

    def sliding_attention(self, gru_embedding):

        sliding_windows = []
        batch_size, seq_len, hidden_size = gru_embedding.shape
        gru_embedding = torch.cat((torch.zeros(batch_size, self.T - 1, hidden_size).to(gru_embedding.device), gru_embedding), dim=-2)
        scores = torch.vmap(self.attention_layer)(gru_embedding)
        
        for i in range(seq_len):
            window = gru_embedding[:, i : i + self.T, :]
            window_scores = scores[:, i : i + self.T, :]
            weights = F.softmax(window_scores, dim=1)
            attentioned = torch.sum(weights * window, dim=1)
            sliding_windows.append(attentioned)
        
        sliding_context = torch.stack(sliding_windows, dim=1)
        return sliding_context
    
    def forward(self, x, structure):
#         self.data.features[:, i : i + self.seq_len],\
#         self.data.ranking_probs[i : i + self.seq_len],\
#         self.data.average_return[:, i : i + self.seq_len],\
#         self.data.pos_return_prob[:, i : i + self.seq_len]
        
        m = len(structure)
        V, T, H = x.shape
#         features = self.embedding_extractor(x.view(V * T, H)).reshape(V, T, -1)
        gru_embedding = self.gru(x)[0]
        temporal_embedding = self.sliding_attention(gru_embedding) # V, T, H
        temporal_embedding_t = temporal_embedding.permute(1, 0, 2) # T, V, H
        
        sector_embedding = torch.empty(x.size(0), x.size(1), self.gat_out_size, device=self.device) # V, T, H
        world_embedding = torch.empty(x.size(0), x.size(1), self.gat_out_size, device=self.device) # V, T, H
        world_sector_embedding = torch.empty(x.size(1), m, self.gat_out_size, device=self.device) # T, V, H
        for i, (sector, idxs) in enumerate(structure.items()):
            n = len(idxs)
            adjacency_matrix = torch.ones(n, n) - torch.eye(n)
            temporal_embedding_in_sector = temporal_embedding_t[:, idxs, :]
            graph_embedding = self.gat_sectors_layers[sector](temporal_embedding_in_sector, adjacency_matrix) # T, V, H
            sector_embedding[idxs] = graph_embedding.permute(1, 0, 2)
            world_sector_embedding[:, i, :] = graph_embedding.max(dim=-2).values
        
        adjacency_matrix = torch.ones(m, m) - torch.eye(m)
        world_graph_embedding = self.gat_world_layer(world_sector_embedding, adjacency_matrix) # T, V, H
        world_graph_embedding = world_graph_embedding.permute(1, 0, 2) # V, T, H
        for i, (world_graph_embedding_for_sector, idxs) in enumerate(zip(world_graph_embedding, structure.values())):
            world_embedding[idxs] = world_graph_embedding_for_sector.repeat(len(idxs), 1, 1)
        
        concated_embedding = torch.cat((temporal_embedding, sector_embedding, world_embedding), dim=-1)
        final_embedding = self.fusion(concated_embedding)
        logits = self.rank(final_embedding)
        return_avg = self.return_avg(final_embedding)
        return_prob = self.return_prob(final_embedding)
        
        return logits, return_avg, return_prob
    
    def training_step(self, batch):
        
        loss = 0.
        x, rp, ar, prp = batch
        x, rp, ar, prp = x.to(self.device), rp.to(self.device), ar.to(self.device), prp.to(self.device)
        
        logits, return_avg, return_prob = self.forward(x, self.structure) # V, T, 1
        probs = F.log_softmax(logits.squeeze(-1).T, dim=-1)
        loss += F.kl_div(probs, rp, reduction="batchmean")
        loss += 0.02 * F.mse_loss(return_avg.squeeze(-1), ar)
        loss += F.binary_cross_entropy(return_prob.squeeze(-1), prp)
        
        self.metrics["Train Loss"].append(loss.item())
        return loss

    def validation_step(self, batch):
        
        x, rp, ar, prp = batch
        x, rp, ar, prp = x.to(self.device), rp.cpu(), ar.cpu(), prp.cpu()
        rp, ar, prp = rp[self.val_cut:], ar[:, self.val_cut:], prp[:, self.val_cut:]
        
        logits, return_avg, return_prob = self.forward(x, self.structure)
        logits = logits.squeeze(-1).T[self.val_cut:].cpu()
        return_avg = return_avg.squeeze(-1)[:, self.val_cut:].cpu()
        return_prob = return_prob.squeeze(-1)[:, self.val_cut:].cpu()
        
        target = torch.from_numpy(np.vstack([rankdata(q, method="ordinal") for q in rp]))
        idxs = torch.arange(logits.size(0)).unsqueeze(-1).repeat(1, logits.size(-1)).to(torch.long)
        
        ndcg = self.ndcg(logits, target, indexes=idxs)
        prob_mae = F.l1_loss(return_prob, prp)
        return_mse = F.mse_loss(return_avg, ar)
        self.metrics["NDCG"].append(ndcg.item())
        self.metrics["MAE of positive return prob"].append(prob_mae.item())
        self.metrics["MSE of avg return"].append(return_mse.item())
        
    def on_train_epoch_end(self):
        
        for metric in self.metrics:
            self.log(metric, np.mean(self.metrics[metric]))
            self.metrics[metric].clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    