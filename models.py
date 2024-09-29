import torch.nn.functional as F
import lightning as l
import torch.nn as nn
import torch


class GAT(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 negative_slope=0.2):
        super(MixerBlock, self).__init__()
        
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
    

class MixerBlock(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(MixerBlock, self).__init__()

        self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_size, input_size),
                                   nn.Dropout(dropout))

    def forward(self, inputs):        
        return self.layer(inputs)


class TriU(nn.Module):
    
    def __init__(self, time_step):
        super(TriU, self).__init__()
        
        self.time_step = time_step
        self.triU = nn.ModuleList([
            nn.Linear(i + 1, 1)
            for i in range(time_step)
        ])

    def forward(self, inputs):
        
        x = torch.empty(*inputs.shape[:-1], 0)
        for i, triU in enumerate(self.triU):
            x = torch.cat((x, triU(inputs[..., :i + 1])), dim=-1)
        
        return x
    
    
class Mixer2dTriU(nn.Module):
    
    def __init__(self, time_steps, input_size):
        super(Mixer2dTriU, self).__init__()
        
        self.layer_norm_1 = nn.LayerNorm((time_steps, input_size))
        self.layer_norm_2 = nn.LayerNorm((time_steps, input_size))
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(input_size, input_size)

    def forward(self, inputs):
        
        x = self.layer_norm_1(inputs)
        x = self.timeMixer(x.mT).mT
        x = self.layer_norm_2(x + inputs)
        y = self.channelMixer(x)
        
        return x + y
    
    
class MultTime2dMixer(nn.Module):
    
    def __init__(self, time_step, channel, scale_dim):
        super(MultTime2dMixer, self).__init__()
        
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        
        x = self.mix_layer(inputs)
        y = self.scale_mix_layer(y)
        
        return torch.cat((inputs, x, y), dim=1)
    
    
class StockMixer(nn.Module):
    
    def __init__(self, time_steps, input_size, scale):
        super(StockMixer, self).__init__()
        
        scale_dim = time_steps // scale
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=scale, stride=scale)
        self.mixer = MultTime2dMixer(time_steps, input_size, scale_dim=scale_dim)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.channel_fc = nn.Linear(input_size, 1)

    def forward(self, inputs):
        
        x = self.conv(inputs.mT).mT
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)
        y = self.time_fc(y)
        
        return y
