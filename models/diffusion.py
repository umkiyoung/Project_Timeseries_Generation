import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from models.transformer import Transformer
from utils.utils import default, extract, identity


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            **kwargs
    ):
        super().__init__()
        
        # model
        self.model = Transformer(n_feat=feature_size, 
                                 n_channel=seq_length, 
                                 n_layer_enc=n_layer_enc, 
                                 n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, 
                                 attn_pdrop=attn_pd, 
                                 resid_pdrop=resid_pd, 
                                 mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, 
                                 n_embd=d_model, 
                                 conv_params=[kernel_size, padding_size], 
                                 **kwargs)
        self.timesteps = int(timesteps)
        self.loss_type = loss_type
        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        # To enhance computing performance
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # diffusion
        betas = self.cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Used in Forward Process
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

        # sampling
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        return torch.clip(betas, 0, 0.999)
    
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def output(self, x, t):
        trend, season = self.model(x, t)
        model_output = trend + season
        return model_output

    def q_sample(self, x_0, t, noise):
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t

    def _train_loss(self, x_0, t, target=None):
        noise = torch.randn_like(x_0)
        
        if target is None:
            target = x_0
        
        x = self.q_sample(x_0=x_0, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t)
        train_loss = self.loss_fn(model_out, target, reduction='none')
        
        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            f_loss1 = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
            f_loss2 = self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none') # ?
            fourier_loss = f_loss1 + f_loss2
            train_loss +=  self.ff_weight * fourier_loss
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        
        return train_loss.mean()

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        return self._train_loss(x_0=x, t=t, **kwargs)
    
    # sampling
    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        
        return sample_fn((batch_size, seq_length, feature_size))
    
    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_0, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_0
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_0 * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
    
    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.timesteps)),
                      desc='sampling loop time step', total=self.timesteps):
            img, _ = self.p_sample(img, t)
        return img

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, t, clip_x_start=False):
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_0 = self.output(x, t)
        x_0 = maybe_clip(x_0)
        pred_noise = self.predict_noise_from_start(x, t, x_0)
        return pred_noise, x_0

    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_0 = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_0

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_0 = self.model_predictions(x, t)
        if clip_denoised:
            x_0.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_0=x_0, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_0

    def q_posterior(self, x_0, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    