import torch.nn.functional as F
import lightning as l
import torch.nn as nn
import torch


class GAT(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 negative_slope=0.2):
        super().__init__()
        
        self.W = nn.Linear(input_size, output_size, bias=False)
        self.a = nn.Linear(2 * output_size, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.nonlinear = nn.ReLU()
        
    def forward(self, x, adjacency_matrix):
        B, V, H = x.shape
        
        x_r = x.reshape(-1, H)
        p = self.W(x_r)
        p = p.reshape(B, V, -1)

        p_repeated_in_chunks = p.repeat_interleave(V, dim=1)
        p_repeated_alternating = p.repeat(1, V, 1)

        all_combinations_matrix = torch.cat((p_repeated_in_chunks, p_repeated_alternating), dim=-1)
        scores = self.leaky_relu(self.a(all_combinations_matrix))
        scores = scores.view(B, V, V)
        weights = F.softmax(scores, dim=-1)
        weights_masked = weights * adjacency_matrix.view(1, V, V)
        weights_masked *= weights.sum(dim=-1, keepdim=True) / weights_masked.sum(dim=-1, keepdim=True)
        weights_masked = weights_masked.view(B, V, V, 1)
        
        p_repeated = p_repeated_alternating.view(B, V, V, -1)
        attentioned = (p_repeated * weights_masked).sum(dim=-2)
        attentioned = self.nonlinear(attentioned)
        
        return attentioned
