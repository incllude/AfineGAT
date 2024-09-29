from torchmetrics import RetrievalNormalizedDCG, MeanAbsoluteError
from models import GAT, StockMixer
from scipy.stats import rankdata
import torch.nn.functional as F
import lightning as l
import torch.nn as nn
import numpy as np
import torch


class StockRankingModel(l.LightningModule):
    def __init__(self, structure, input_size, hidden_size, gat_out_size, total_steps):
        super(StockRankingModel, self).__init__()
        self.save_hyperparameters()
        
        self.total_steps = total_steps
        self.structure = structure
        self.gat_out_size = gat_out_size
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.day_attention_layer = nn.Sequential(nn.Linear(hidden_size, 1))
        self.world_attention_layer = nn.Sequential(nn.Linear(gat_out_size, 1))
        self.sector_attention_layer = nn.Sequential(nn.Linear(gat_out_size, 1))
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
            "MAE of avg return": []
        }

    def sliding_attention(self, gru_embedding, layer):

        scores = torch.vmap(layer)(gru_embedding)
        weights = F.softmax(scores, dim=-2)
        attentioned = torch.sum(weights * gru_embedding, dim=-2)
    
        return attentioned
    
    def forward(self, x, structure):
#         self.data.features[:, i : i + self.seq_len],\
#         self.data.ranking_probs[i : i + self.seq_len],\
#         self.data.average_return[:, i : i + self.seq_len],\
#         self.data.pos_return_prob[:, i : i + self.seq_len]
        
        m = len(structure)
        V, T, D, H = x.shape
        
        gru_embedding = self.gru(x.reshape(V * T, D, H))[0]
        temporal_embedding = self.sliding_attention(gru_embedding, self.day_attention_layer).view(V, T, -1) # V, T, H
        temporal_embedding_t = temporal_embedding.permute(1, 0, 2) # T, V, H
        week_embedding = temporal_embedding[:, -1, :]
        
        sector_embedding = torch.empty(x.size(0), x.size(1), self.gat_out_size, device=self.device) # V, T, H
        world_embedding = torch.empty(x.size(0), self.gat_out_size, device=self.device) # V, T, H
        world_sector_embedding = torch.empty(1, m, self.gat_out_size, device=self.device) # T, V, H
        for i, (sector, idxs) in enumerate(structure.items()):
            n = len(idxs)
            adjacency_matrix = torch.ones(n, n) - torch.eye(n)
            temporal_embedding_in_sector = temporal_embedding_t[:, idxs, :]
            graph_embedding = self.gat_sectors_layers[sector](temporal_embedding_in_sector, adjacency_matrix) # T, V, H
            sector_embedding[idxs] = graph_embedding.permute(1, 0, 2)

        sector_embedding = self.sliding_attention(sector_embedding, self.sector_attention_layer)
        for i, (sector, idxs) in enumerate(structure.items()):
            world_sector_embedding[0, i, :] = sector_embedding[idxs].max(dim=0).values
        
        adjacency_matrix = torch.ones(m, m) - torch.eye(m)
        world_graph_embedding = self.gat_world_layer(world_sector_embedding, adjacency_matrix) # T, V, H
        world_graph_embedding = world_graph_embedding.squeeze(0) # V, T, H
        for i, (world_graph_embedding_for_sector, idxs) in enumerate(zip(world_graph_embedding, structure.values())):
            world_embedding[idxs] = world_graph_embedding_for_sector.repeat(len(idxs), 1)
        
        concated_embedding = torch.cat((week_embedding, sector_embedding, world_embedding), dim=-1)
        final_embedding = self.fusion(concated_embedding)
        logits = self.rank(final_embedding)
        return_avg = self.return_avg(final_embedding)
        return_prob = self.return_prob(final_embedding)
        
        return logits, return_avg, return_prob
    
    def training_step(self, batch):
        
        loss = 0.
        xs, rps, ars, prps = batch
        xs, rps, ars, prps = xs.to(self.device), rps.to(self.device), ars.to(self.device), prps.to(self.device)
        
        for x, rp, ar, prp in zip(xs, rps, ars, prps):
            logits, return_avg, return_prob = self.forward(x, self.structure) # V, 1
            probs = F.log_softmax(logits.squeeze(-1).T, dim=-1)
            loss += F.kl_div(probs, rp, reduction="batchmean")
            loss += F.l1_loss(return_avg.squeeze(-1) * 10, ar * 100)
            loss += F.binary_cross_entropy(return_prob.squeeze(-1), prp)
        
        loss /= xs.size(0)
        self.metrics["Train Loss"].append(loss.item())
        return loss

    def validation_step(self, batch):
        
        x, rp, ar, prp = batch
        x, rp, ar, prp = x.to(self.device), rp.cpu(), ar.cpu(), prp.cpu()
        
        logits, return_avg, return_prob = self.forward(x, self.structure)
        logits = logits.squeeze(-1).T.cpu()
        return_avg = return_avg.squeeze(-1).cpu() / 10
        return_prob = return_prob.squeeze(-1).cpu()
        
        target = torch.from_numpy(np.vstack([rankdata(q, method="ordinal") for q in rp])).squeeze(-1)
        idxs = torch.arange(logits.size(0)).to(torch.long)
        
        ndcg = self.ndcg(logits, target, indexes=idxs)
        prob_mae = F.l1_loss(return_prob, prp)
        return_mse = F.l1_loss(return_avg, ar)
        self.metrics["NDCG"].append(ndcg.item())
        self.metrics["MAE of positive return prob"].append(prob_mae.item())
        self.metrics["MAE of avg return"].append(return_mse.item())
        
    def on_train_epoch_end(self):
        
        for metric in self.metrics:
            self.log(metric, np.mean(self.metrics[metric]))
            self.metrics[metric].clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.99, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=self.total_steps, pct_start=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


class StockMixerModel(l.LightningModule):
    def __init__(self, time_steps, input_size, total_steps, max_lr):
        super(StockRankingModel, self).__init__()
        self.save_hyperparameters()
        
        self.max_lr = max_lr
        self.total_steps = total_steps
        
        self.mixer = StockMixer(time_steps, input_size, scale=2)
        
        self.metrics_values = {
            "Train Loss": [],
            "MAE of Return": []
        }
        self.metrics = {
            "MAE of Return": MeanAbsoluteError()
        }
    
    def forward(self, x):
        # V, T, H = x.shape
        
        logits = self.mixer(x)
        return logits
    
    def training_step(self, batch):
        
        loss = 0.
        xs, trs = batch
        
        for x, tr in zip(xs, trs):
            logits = self.forward(x) # V, 1
            loss += F.mse_loss(logits, tr)
        
        loss /= xs.size(0)
        self.metrics_values["Train Loss"].append(loss.item())
        return loss

    def validation_step(self, batch):
        
        x, tr = batch
        tr = tr.cpu()
        
        logits = self.forward(x)
        predictions = logits.flatten().cpu()
        
        for metric_name, metric in self.metrics.items():
            self.metrics_values[metric_name].append(metric(predictions, tr).item())
        
    def on_train_epoch_end(self):
        
        for metric in self.metrics_values:
            self.log(metric, np.mean(self.metrics_values[metric]))
            self.metrics_values[metric].clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.total_steps, pct_start=0.05, div_factor=100, final_div_factor=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    