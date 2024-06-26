import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat

class SinusoidalPosEmb(nn.Module):
    # ?
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb


class LearnablePositionalEncoding(nn.Module):
    # ?
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        
        return self.dropout(x)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module"""
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    

class AdaLayerNorm(nn.Module):
    # ?
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.emb(timestep)
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        
        return x


class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            nn.GELU(),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 n_heads,
                 attn_pdrop=0.1, 
                 resid_pdrop=0.1,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_heads = n_heads

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 condition_embd,
                 n_heads,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_heads = n_heads

    def forward(self, x, encoder_output):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd,
                 n_heads,
                 attn_pdrop,
                 resid_pdrop,
                 mlp_hidden_times,
                 ):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                nn.GELU(),
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep):
        a, att = self.attn(self.ln1(x, timestep))
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_embd,
        n_heads,
        attn_pdrop,
        resid_pdrop,
        mlp_hidden_times,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(n_embd=n_embd,
                           n_heads=n_heads,
                           attn_pdrop=attn_pdrop,
                           resid_pdrop=resid_pdrop,
                           mlp_hidden_times=mlp_hidden_times,
                           ) 
              for _ in range(n_layer)])

    def forward(self, input, t):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd,
                 n_heads,
                 attn_pdrop,
                 resid_pdrop,
                 mlp_hidden_times,
                 condition_dim,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            nn.GELU(),
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep):
        a, att = self.attn1(self.ln1(x, timestep))
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd,
        n_heads,
        n_layer,
        attn_pdrop,
        resid_pdrop,
        mlp_hidden_times,
        condition_dim    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(
          *[DecoderBlock(
              n_feat=n_feat,
              n_channel=n_channel,
              n_embd=n_embd,
              n_heads=n_heads,
              attn_pdrop=attn_pdrop,
              resid_pdrop=resid_pdrop,
              mlp_hidden_times=mlp_hidden_times,
              condition_dim=condition_dim) 
            for _ in range(n_layer)])
      
    def forward(self, x, t, enc):
        b, c, _ = x.shape
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc,
        n_layer_dec,
        n_embd,
        n_heads,
        attn_pdrop,
        resid_pdrop,
        mlp_hidden_times,
        max_len,
    ):
        super().__init__()
        self.embedding = Conv_MLP(n_feat, n_embd, pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, pdrop=resid_pdrop)
        
        self.combine_s = nn.Conv1d(in_channels=n_embd, 
                                   out_channels=n_feat, 
                                   kernel_size=1, 
                                   stride=1, 
                                   padding_mode='circular', 
                                   bias=False)
        
        self.combine_m = nn.Conv1d(in_channels=n_layer_dec, 
                                   out_channels=1, 
                                   kernel_size=1, 
                                   stride=1,
                                   padding_mode='circular', 
                                   bias=False)

        self.encoder = Encoder(n_layer=n_layer_enc, 
                               n_embd=n_embd, 
                               n_heads=n_heads, 
                               attn_pdrop=attn_pdrop, 
                               resid_pdrop=resid_pdrop, 
                               mlp_hidden_times=mlp_hidden_times, 
                               )
        
        self.pos_enc = LearnablePositionalEncoding(n_embd, 
                                                   dropout=resid_pdrop, 
                                                   max_len=max_len)

        self.decoder = Decoder(n_channel, 
                               n_feat, 
                               n_embd, 
                               n_heads, 
                               n_layer_dec, 
                               attn_pdrop, 
                               resid_pdrop, 
                               mlp_hidden_times,
                               condition_dim=n_embd)
        
        self.pos_dec = LearnablePositionalEncoding(n_embd, 
                                                   dropout=resid_pdrop, 
                                                   max_len=max_len)

    def forward(self, input, t):
        embedding = self.embedding(input)
        inp_enc = self.pos_enc(embedding)
        enc_cond = self.encoder(inp_enc, t)

        inp_dec = self.pos_dec(embedding)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond)

        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend

        return trend, season_error

