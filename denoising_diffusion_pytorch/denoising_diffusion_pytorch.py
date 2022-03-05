"""@xvdp minor modifications for local running
Trainer:
    added self._cont() creating a '_continue_training'  file in results: delete file to terminate training
    added more verbose logging for time
    added 'milestone' arg: to load milestone
    removed 'channels' arg: grab from diffusion model
    removed inference on first step when loading milestones

Dataset:
    added robust loading (skip if bad file, ensure channels are correct)
    added channels arg

GaussianDiffusion:
    removed 'channels' arg: grab from unet
 
"""
import os
import os.path as osp
import time
import random

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

import vidi # for video out
from .util import numpy_grid, strf

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# pylint: disable=no-member

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
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

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
    return a[t].view(x_shape[0], *[1]*(len(x_shape)-1))

    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False, **kwargs):
  
    if "scale" in kwargs and kwargs["scale"] > 1 or "tile" in kwargs and kwargs["tile"] > 1:
        return _noise_like(shape, device, repeat=False, **kwargs)
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    out = repeat_noise() if repeat else noise()

    return out

def _noise_like(shape, device, repeat=False, scale=1, interpolation="nearest", tile=1):
    """
    repeat, same noise for every batch sample
    scale   upsample noise by factors:
        TEST: upsampled deterministic noise returns garbage outside of the trained manifold
    tile    tile noise
        TEST: tiled deterministic noise returns tiles differnetn but on the same palette range than nontiled

    """
    if scale > 1:
        shape = [*shape[:2], shape[2]//scale, shape[3]//scale]
    if tile > 1:
        div = 2**(tile-1)
        shape = [*shape[:2], shape[2]//div, shape[3]//div]

    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    out = repeat_noise() if repeat else noise()

    if scale > 1:
        out = torch.nn.functional.interpolate(out, scale_factor=scale, mode=interpolation)
    for _ in range(tile - 1):
        out = torch.cat((out, out), dim=2)
        out = torch.cat((out, out), dim=3)
    return out

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

##
# original beta schedules
#
def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def original_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class GaussianDiffusion(nn.Module):
    """ removed arg channels, read from denoise_fn.channels
    """
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = denoise_fn.channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.out = None
        self.outstep=10

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        noise=self.denoise_fn(x, t)
        self.appendout(noise, f"1.-> x1 = U(x) : unet μ{noise.mean().item():.3f}, σ{noise.std().item():.3f}") # 1
        x_recon = self.predict_start_from_noise(x, t=t,noise=noise)
        self.appendout(x_recon, f"2.-> x1 = x1√(1/Πα) - x0/√(1/Πα - 1)) : predict start μ{x_recon.mean().item():.3f}, σ{x_recon.std().item():.3f}") # 2
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        self.appendout(model_mean, f"3.-> x1 c_1 + x0 c_2 : model_mean μ{model_mean.mean().item():.3f}, σ{model_mean.std().item():.3f}") # 2
        return model_mean, posterior_variance, posterior_log_variance

    def _predict_start_from_noise(self, x, t, noise):
        """ same as above, fore readability
        """
        _a = self.sqrt_recip_alphas_cumprod
        _b = self.sqrt_recipm1_alphas_cumprod
        shape = (len(x), *[1] * (x.dim() - 1)) # (b, ...ones)

        return _a[t].view(shape)*x - _b[t].view(shape)*noise

    def _q_posterior(self, x_start, x, t):
        _c1 = self.posterior_mean_coef1
        _c2 = self.posterior_mean_coef2
        shape = (len(x), *[1] * (x.dim() - 1)) # (b, ...ones)

        post_mean = _c1[t].view(shape)*x_start + _c2[t].view(shape)*x
        self.appendout(post_mean)

        post_var = self.posterior_variance[t].view(shape)
        post_log_var_clipped = self.posterior_log_variance_clipped[t].view(shape)
 
        return post_mean, post_var, post_log_var_clipped

    @torch.no_grad()
    def _p_mean_variance(self, x, t, clip_denoised: bool):
        # run UNet once forward
        noise=self.denoise_fn(x, t)
        self.appendout(noise, "-> U(x)")
        x_recon = self._predict_start_from_noise(x, t=t, noise=noise)
        self.appendout(x_recon, "-> model mean")

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        return self._q_posterior(x_start=x_recon, x=x, t=t)


    def appendout(self, x, msg=None):
        if self.out is not None and (not self.step or self.step == self.num_timesteps-1 or not self.step%(self.num_timesteps//self.outstep)):
            self.out.append(x.cpu().clone().detach())
            if msg is not None:
                print(msg)

    @torch.no_grad()
    def sample_steps(self, batch_size=1, seed=None, step=None):
        self.outstep = step or self.outstep
        self.out = []

        if seed is not None:
            torch.manual_seed(seed)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(shape)

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, scale=1, interpolation="nearest", tile=1):
 
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)

        noise = noise_like(x.shape, device, repeat_noise, #ORIGINAL ARGS
                          scale=scale, interpolation=interpolation, tile=tile) # TEST ARGS

        self.appendout(noise, f"4.-> noise_like(x) μ{noise.mean().item():.3f}, σ{noise.std().item():.3f}")
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def _p_sample(self, x, t, clip_denoised=True):
        """
        Args
            x               tensor (b, h, h, w) markov chain step, initialized as torch.randn(shape) at x_0
            t               tensor (b)  blend factor, on fwd step initialized as torch.full((b,), i), 1000>i>=0
            # clip_denoised   bool (True) clamp -1,1 # disabled
            repeat_noise    bool (False) same noise for every batch sample
        """
        b, h, h, w = x.shape
        model_mean, _, model_log_variance = self._p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        self.appendout(model_mean, "-> model mean")

        noise = torch.randn(x.shape, device=x.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, step=1, scale=1, interpolation="nearest", tile=1):
    
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps, step)),
                      desc='sampling loop time step', total=self.num_timesteps):
            self.step = i
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                scale=scale, interpolation=interpolation, tile=tile)
            self.appendout(img, f"out = mean + noise*e^(logvar/2)  μ{img.mean().item():.3f}, σ{img.std().item():.3f}")
        return img

    @torch.no_grad()
    def _p_sample_loop(self, shape, step=1):
    
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps, step)),
                      desc='sampling loop time step', total=self.num_timesteps):
            self.step = i
            img = self._p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
            self.appendout(img)

        if self.out is not None:
            self.out = torch.cat(self.out)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, height=None, width=None, seed=None, step=1):
        """ added args
                'seed'              reproducibility testing
                'width' 'height'    test different sizes
                'step'              test skipping markov steps # useless results
        """
        height = height or self.image_size
        width = width or self.image_size
        if seed is not None:
            torch.manual_seed(seed)
        shape = (batch_size, self.channels, height, width)
        return self.p_sample_loop(shape, step=step)


    ###
    # sampling experiments
    #
    # input image instead of rnd
    # 1. video out
    @torch.no_grad()
    def p_sample_loop_vid(self, shape, name="sample.mp4", play=False):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        _img = numpy_grid(img)
        with vidi.FFcap(name, size=_img.shape[:2], fps=25) as F:
            F.add_frame(_img)

            for i in tqdm(reversed(range(0, self.num_timesteps)),
                        desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
                F.add_frame(numpy_grid(img))
        if play:
            vidi.ffplay(name)
        return name

    @torch.no_grad()
    def sample_vid(self, batch_size=16, name="sample.mp4", play=False):
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop_vid(shape, name=name, play=play)

    @torch.no_grad()
    def sample_from(self, img, timesteps=None, scale=1, interpolation="nearest", tile=1):
        return self.p_sample_loop_from(img, timesteps=timesteps, scale=scale, interpolation=interpolation, tile=tile)

    @torch.no_grad()
    def p_sample_loop_from(self, img, timesteps=None, scale=1, interpolation="nearest", tile=1):
        device = self.betas.device
        img = img.to(device=device)
        timesteps = timesteps or self.num_timesteps

        b = img.shape[0]
        for i in tqdm(reversed(range(0, timesteps)),
                      desc='sampling loop time step', total=timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), scale=scale,
                                interpolation=interpolation, tile=tile)
        return img

    # unused
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # q_sample, per batch sample: x*sqrt(alpha) + noise*(1-sqrt(alpha))
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _q_sample(self, x, t, noise=None):
        """ same as above - for readability
        """
        noise = noise or torch.randn_like(x)
        shape = (len(x), *[1] * (x.dim() - 1)) # (b, ...ones)
        _a = self.sqrt_alphas_cumprod
        _b = self.sqrt_one_minus_alphas_cumprod

        return _a[t].view(shape)*x + _b[t].view(shape)*noise

    # def _q_sample_lerp(self, x_start, t, noise=None):
    #     """ to test: lerp over self.alphas_cumprod[t]
    #     """
    #     noise = noise or torch.randn_like(x_start)
    #     shape = (len(x_start), *[1] * (x_start.dim() - 1))
    #     return torch.lerp(x_start, noise, self.alphas_cumprod[t].view(shape))

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        # training noise
        noise = default(noise, lambda: torch.randn_like(x_start))

        # per batch sample: x*sqrt(alpha) + noise*(1-sqrt(alpha))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    
        # Unet
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)

        # what if we minimize the distance to image given no- this should also learn something
        elif self.loss_type == '-l1':
            loss = (x_start - x_recon).abs().mean()

        # ELBO minimization is KLD, not L1 or L2... what does this mean.

        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # per batchrandom noise blend 1/num_timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    """ added channels arg
    """
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png'], channels=3, resize_images=True):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.channels = channels

        # @xvdp this set of transforms doesn make much sense
        # Resize & Crop should be different sizes for randomness
        if resize_images:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index=None):
        index = index if index is not None else random.randint(0, len(self)-1)
        path = self.paths[index]
        try: # fault tolerant image loading
            img = Image.open(path)
            if self.channels == 1:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
        except:
            return self.__getitem__()
        return self.transform(img)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        train_batch_size = 48,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        resize_images = True, # if false Dataset loads images as they are
        num_workers=4, # default num_workers should be ~cores//3
        milestone = None
    ):
        """
        removed image_size & channels (get from diffusion_model)
        added:  milestone [None] loads milestone if found

        Notes:
            * fp16 on single TitanRTX is 2x slower
        """
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset(folder, self.image_size, channels=diffusion_model.channels, resize_images=resize_images)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        if milestone == 'last':
            inf = sorted([f.path for f in os.scandir(self.results_folder)
                            if f.name[-3:] ==".pt"],
                            key=lambda x: int(x.split(".")[0].split("-")[1]))
            milestone = inf[-1] if inf else None

        if milestone is not None:
            print(f"loading milestone {milestone}")
            self.load(milestone)
        else:
            print("Resetting parameters")
            self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone, verbose=False):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        """
        """
        if not osp.isfile(milestone):
            milestone = osp.join(self.results_folder, f'model-{milestone}.pt')
        assert osp.isfile(milestone), f"could not load milestone {milestone}"

        data = torch.load(milestone)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

  
    def train(self):
        time_start = time.time()
        backwards = partial(loss_backwards, self.fp16)
        zero_step = self.step

        self._cont(True, name="_stop")  # when deleted stop immediately
        self._cont(True, name="_softstop") # when deleted wait until next milestone
        _msg = ""

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)

                total_time = time.time() - time_start
                step = self.step + 1
                iter_time = total_time / (step - zero_step)
                remain_time = (self.train_num_steps - step) * iter_time
                print(f'{[self.step]} {loss.item():.3f}  time /it {iter_time:.3f} elapsed {strf(total_time)} remain {strf(remain_time)} data {tuple(data.shape)}')

                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if not self._cont(name="_stop") or (self.step != zero_step and self.step % self.save_and_sample_every == 0):
                milestone = self.step # // self.save_and_sample_every
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                image_name = osp.join(self.results_folder, f'sample-{milestone}.png')
                utils.save_image(all_images, image_name, nrow = 6)
                self.save(milestone, verbose=True)

                if not self._cont(name="_stop") or not self._cont(name="_softstop"):
                    _msg = f"interupted training, at step {self.step}"
                    break

            self.step += 1

        print(f'training completed {_msg}')

    def _cont(self, init=False, name="_continue_training"):
        """
        Create place holder file. Delete file to stop training.
        """
        _cont = osp.join(self.results_folder, name)
        if init and not osp.isfile(_cont):
            with open(_cont, "w") as _fi:
                pass
        return osp.isfile(_cont)
