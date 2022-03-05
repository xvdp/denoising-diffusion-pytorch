"""@xvdp
Set of tests to understand DDPM workings

"""
import math
import sys
import os.path as osp
import random
from pprint import pprint
import torch
from koreto import ObjDict

from denoising_diffusion_pytorch.util import to_image, open_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


CONFIG="config_multitudes.yaml"
CONFIG="config_ffhq.yaml"
# CONFIG="config_ffhq128.yaml"

# pylint: disable=no-member
def train(config=CONFIG, **kwargs):
    """ simple trainer from yaml
    """
    trainer = load_trainer(config, **kwargs)
    trainer.train()

def load_config(config=CONFIG, **kwargs):
    """ Load default parameters from lucirains depot
    updated with whatever is in config yaml
    """
    # default Train parameters
    model_params = {'dim': 64, 'dim_mults': (1, 2, 4, 8), 'channels': 3}
    diffusion_params = {'image_size': 128, 'timesteps': 1000, 'loss_type': 'l1', 'betas': None}
    train_params = {'ema_decay': 0.995, 'train_batch_size': 32, 'train_lr': 2e-5,
                    'train_num_steps':100000, 'gradient_accumulate_every': 2, 'fp16': True,
                    'step_start_ema':2000, 'update_ema_every': 10, 'save_and_sample_every':1000,
                    'results_folder': './results'}

    if config is not None:
        assert osp.isfile(config), f"missing config file {config}"
        print(f"Loading configuration {config}")
        obj = ObjDict()
        obj.from_yaml(config)
        if hasattr(obj, 'model'):
            model_params.update(obj.model)
        if hasattr(obj, 'diffusion'):
            diffusion_params.update(obj.diffusion)
        if hasattr(obj, 'train'):
            train_params.update(obj.train)
        if hasattr(obj, 'data'):
            data = obj.data


    model_params.update(**{k:v for k,v in kwargs.items() if k in model_params})
    diffusion_params.update(**{k:v for k,v in kwargs.items() if k in diffusion_params})
    train_params.update(**{k:v for k,v in kwargs.items() if k in train_params})

    pprint({"data": data})
    pprint({"model": model_params})
    pprint({"diffusion": diffusion_params})
    pprint({"train":train_params})
    return model_params, diffusion_params, train_params, data

def load_trainer(config=CONFIG,  **kwargs):
    """
    """
    model_params, diffusion_params, train_params, data = load_config(config, **kwargs)
    model = Unet(**model_params).cuda()
    diffusion = GaussianDiffusion(model, **diffusion_params).cuda()
    return Trainer(diffusion, data, **train_params)
###
#
# Show single step training
#
def single_train_step(fix_noise=None, train_batch_size=5, **kwargs):
    """
    collects images of
        1. input
        2. noise
        3. noise + input * blendfactors
        4. reconstructed noise
        5. diff: abs (noise, recon)
    """
    kwargs['train_batch_size'] = train_batch_size
    self = load_trainer(**kwargs)
    x_start = next(self.dl)
    b, *_ = x_start.shape

    _fix = ""
    if fix_noise is not None and 1 > fix_noise >= 0:
        t = torch.ones((b,)).long()*int(self.model.num_timesteps* fix_noise)
        _fix = f"α{fix_noise}"
    else:
        t = torch.randint(0, self.model.num_timesteps, (b,)).long()

    noise = torch.randn_like(x_start)
    x_noisy = self.model.q_sample(x_start=x_start.cuda(), t=t.cuda(), noise=noise.cuda())
    x_recon = self.model.denoise_fn(x_noisy, t.cuda()).cpu().clone().detach()
    x_loss = (noise - x_recon).abs()
    out = torch.cat((x_start, noise, x_noisy.cpu().clone().detach(), x_recon, x_loss))

    try:
        from augment.transforms import Show, NormToRange
        S = Show()
        N = NormToRange()

        S(N(out), ncols=train_batch_size,
        title=f"Train Step {self.step}: 1.x, 2.noise, 3.blend(x, noise)[{_fix}] 4.Unet: recon, 5. abs(noise - recon) {x_loss.mean():.3f} ",
        path=osp.join(self.results_folder, f"OneStepTrain_{_fix}_{self.step}.png"), width=10)
    except:
        pass
    return out

def sample2(config=CONFIG, batch_size=16, ema=True, seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        self = load_trainer(config, **kwargs)
        model = self.ema_model if ema else self.model

        # sample
        shape = (batch_size, self.model.channels, self.image_size, self.image_size)
        device = model.betas.device

        x = torch.randn(shape, device=device) # at noise x_T

        # p_sample_loop
        mk_steps = model.num_timesteps

        for i in range(0, mk_steps)[::-1]:

            t = torch.full((batch_size,), i, device=device, dtype=torch.long) # [i]*batch_size
            _a = view_like((1. - model.betas[t]), x)
            _pa = view_like(model.alphas_cumprod[t], x)
            _lv = model.posterior_log_variance_clipped[t]
            _Unet = model.denoise_fn

            _z = torch.rand_like(x)
            _z = mul((0.5 * _lv).exp()*torch.clamp(t, 0,1).to(dtype=x.dtype), _z)

            x = (x - (1-_a)*_Unet(x, t)/torch.sqrt(1-_pa))/torch.sqrt(_a) + _z

        return x


def trace_sample(config=CONFIG, batch_size=1, seed=None, ema=True, steps=4,  **kwargs):
    """ trace inference
    
    """

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():

        self = load_trainer(config, **kwargs)
        model = self.ema_model if ema else self.model

        # sample
        shape = (batch_size, self.model.channels, self.image_size, self.image_size)
        device = model.betas.device

        x = torch.randn(shape, device=device) # at noise x_T

        # p_sample_loop
        mk_steps = model.num_timesteps
        skips = int(mk_steps//steps)
        out = []

        ncols = None

        for i in range(0, mk_steps)[::-1]:
            step = []
            t = torch.full((batch_size,), i, device=device, dtype=torch.long) # [i]*batch_size

            # p_sample()
            # p_mean_variance()
            # model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)

            u_noise = model.denoise_fn(x, t) # unet(input noise 'x', markov step 't')

            if not i%skips or i== mk_steps-1 or not i:
                step.append(u_noise.cpu().clone().detach())
            # predict_start_from_noise()
            #x_recon = self._predict_start_from_noise(x, t=t, noise=noise)
            _a = model.sqrt_recip_alphas_cumprod
            _b = model.sqrt_recipm1_alphas_cumprod
            _shape = (len(x), *[1] * (x.dim() - 1)) # (b, ...ones)

            # reconstruction: blend by reciprocals of alphas
            x_recon =  _a[t].view(_shape)*x - _b[t].view(_shape)*u_noise
            x_recon.clamp_(-1., 1.)
            if not i%skips or i== mk_steps-1 or not i:
                step.append(x_recon.cpu().clone().detach())

            _c1 = model.posterior_mean_coef1
            _c2 = model.posterior_mean_coef2
            _lv = model.posterior_log_variance_clipped

            post_mean = _c1[t].view(_shape)*x_recon + _c2[t].view(_shape)*x
            if not i%skips or i== mk_steps-1 or not i:
                step.append(post_mean.cpu().clone().detach())


            post_log_var = _lv[t].view(_shape)
            # sample noise AGAIN!?
            noise = torch.randn(x.shape, device=x.device)
            # mask = torch.clamp(t, 0,1).to(dtype=post_mean.dtype).view(_shape)
            mask = (1 - (t == 0).float()).reshape(_shape)


            # at the last step
            # t == 0: x_t1 = post_mean
            x =  post_mean + mask * (0.5 * post_log_var).exp() * noise
            if not i%skips or i== mk_steps-1 or not i:
                step.append(x.cpu().clone().detach())
            
            ncols = ncols or len(step)
            if not i%skips or i== mk_steps-1 or not i:
                print(f"step [{i}]")
                print(f"\treciprocals x1*{_a[t][0].item():.3f} - x0*{_b[t][0].item():.3f}")
                print(f"\tcoeffs x1*{_c1[t][0].item():.3f} + x0*{_c2[t][0].item():.3f}")
                print(f"\te^(logvar/2) x1 + noise*{ (0.5 * post_log_var).exp().item():.3f}")
                out.append(torch.cat(step))

        out = torch.cat(out)
    try:
        from augment.transforms import Show, NormToRange
        S = Show()
        N = NormToRange()

        S(N(out), ncols=ncols, title=f"ux = U(x)   rx = reciprocals(ux, x)    mx = coeffs(rx, x)    x = mx + noise e^logvar/2",
        
        path=osp.join(self.results_folder, f"SampleSteps_{steps}_{self.step}.png"), width=10)
    except:
        pass
    return out

        # image = model.sample(batch_size=batch_size, height=height, width=width, seed=seed, step=step)

def sample_trace(config=CONFIG, batch_size=1, step=5, ema=True, seed=None, save=False, suffix="", show=False, **kwargs):

    self = load_trainer(config, **kwargs)
    model = self.ema_model if ema else self.model

    image = model.sample_steps(batch_size=batch_size, seed=seed, step=step)

    out = torch.cat(model.out)
    title = "1. x1=Unet(x0)    2.x1=x1√(1/Πα)-x0/√(1/Πα -1)   "
    title += "3. x1=x1β√([1],Πα[:-1])/(1-Πα)+x0(1-([1],Πα[:-1]))√α/(1-Πα)   4.noise   5.x1=x+noise(e^(σσ/2))"

    try:
        from augment.transforms import Show
        S = Show()
        S((out+1)/2, ncols=5, title=title, path=osp.join(self.results_folder, f"SampleSteps_{step}_{self.step}.png"),
          width=15, adjust={"top":0.99})
    except:
        pass
    return out, image
def mul(x: torch.Tensor, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """ mutliply over dimension
    x   tensor, ndim=1
    y   tensor
    dim int [0] dimension to expand
    """
    assert x.ndim == 1, f"expected single dim tensor, found {x.shape}"
    assert 0 <= dim < y.ndim
    shape = [1] * y.ndim
    shape[dim] = len(x)
    return x.view(shape) * y

def view_like(x: torch.Tensor, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    x   tensor, ndim=1
    y   tensor
    dim int [0] dimension to expand
    """
    assert x.ndim == 1, f"expected single dim tensor, found {x.shape}"
    assert 0 <= dim < y.ndim
    shape = [1] * y.ndim
    shape[dim] = len(x)
    return x.view(shape)

###
#
# Sampling Tests
#
# Video
def sample_vid(config=CONFIG, batch_size=16, ema=True, play=True, **kwargs):
    """
    TEST: generate video from markov chain
    loads 'milestone' from 'results_folder' spec'd in 'config_....yaml'
    renders video of sampling loop
    """
    trainer = load_trainer(config, **kwargs)
    model = trainer.ema_model if ema else trainer.model

    name = osp.join(trainer.results_folder,
                    f"{('ema' if ema else 'raw')}_sample-{trainer.step}.mp4")
    name = model.sample_vid(batch_size=batch_size, name=name, play=play)

    print(f"saved video to {name}")

# General Sampler
def sample(config=CONFIG, batch_size=16, height=None,
           width=None, seed=None, step=1, ema=True, save=False, suffix="", show=False, **kwargs):
    """
    TEST: determinism by applying random seed
    >>> sample(seed=0, show=True, save=True)
    >>> sample(seed=0, show=True, save=True, suffix="_take2")
    RESULT: image generation on same milestone is deterministic

    >>> milestones = ['last', 20000, 15000, 37, 25, 1]
    >>> for m in milestones:
    >>>     sample(seed=0, show=False, save=True, milestone=m)

    test width, height: ok, a little weird
    >>> sample(seed=0, show=True, save=True, width=512, height=256, batch_size=8)
    # ema_sample-27951_256x512_seed0.png

    test step: no good
    >>> sample(seed=0, show=True, save=True, batch_size=16, step=2)
    """
    trainer = load_trainer(config, **kwargs)
    model = trainer.ema_model if ema else trainer.model

    image = model.sample(batch_size=batch_size, height=height, width=width, seed=seed, step=step)

    if save or show:
        if save:
            name = osp.join(trainer.results_folder,
                            f"{('ema' if ema else 'raw')}_sample-{trainer.step}")
            if width or height:
                width = width or model.image_size
                height = height or model.image_size
                name = f"{name}_{height}x{width}"
            if seed is not None:
                name = f"{name}_seed{seed}"
            if step > 1:
                name = f"{name}_step{step}"
            save = f"{name}{suffix}.png"

    return to_image(image, save, show)

# Sample from other images
def sample_from(image, config=CONFIG, seed=None,
             save=False, suffix="", show=True, **kwargs):
    """
    """
    image_size = image.shape[2]
    assert 2**(int(math.log2(image_size))) == image_size and  image.shape[3] == image_size, f"square power of 2 image reqd, found {image.shape}"

    if seed is not None:
        torch.manual_seed(seed)
    trainer = load_trainer(config, image_size=image_size, **kwargs)
    model = trainer.ema_model# if ema else trainer.model

    image = model.sample_from(image.cuda())

    if save:
        if save:
            name = osp.join(trainer.results_folder,
                            f"from_img_")
            if seed is not None:
                name = f"{name}_seed{seed}"

            save = f"{name}{suffix}_sample-{trainer.step}.png"
    return to_image(image, save, show)


# Upsampling
def sampleup(config=CONFIG, seed=None, scale=2,
             batch_size=1, interpolation="bicubic",
             save=False, suffix="", show=True, **kwargs):
    """ TEST:   upsample random source image, then sample
    RESULT:     rnd -> img has no correlation to rnd -> upsample -> img
    """
    if seed is not None:
        torch.manual_seed(seed)
    trainer = load_trainer(config, **kwargs)
    model = trainer.ema_model# if ema else trainer.model

    shape = (batch_size, model.channels, model.image_size, model.image_size)
    img = torch.randn(shape, device='cuda')
    if scale != 1:
        img = torch.nn.functional.interpolate(img, scale_factor=scale, mode=interpolation)
    print(f"img size {tuple(img.shape)}")
    image = model.sample_from(img, scale=scale, interpolation=interpolation)

    if save:
        if save:
            name = osp.join(trainer.results_folder,
                            f"upres_{scale}")
            if seed is not None:
                name = f"{name}_seed{seed}"
            name = f"{name}_{interpolation}"

            save = f"{name}{suffix}_sample-{trainer.step}.png"
    return to_image(image, save, show)

def sampleup_loop(config=CONFIG, seed=None,
                  scales=[1, 2, 4], batch_size=1, interpolation="bicubic",
                    save=True, suffix="", **kwargs):
    """ TEST:   upsample random source image, then sample
    RESULT:     rnd -> img has no correlation to rnd -> upsample -> img
    """
    seed = seed if seed is not None else random.randint(0,1000)
    out = []
    for scale in scales:
        print(f"sampling with scale x{scale}")
        out.append(sampleup(config=config, seed=seed, scale=scale,
                            batch_size=batch_size, interpolation=interpolation, save=save,
                            suffix=suffix, show=False, **kwargs))
    return out

# Tile Sampling
def sample_tile_loop(config=CONFIG, seed=None, tiles=[1,2,3],
                batch_size=1, save=True, suffix="", **kwargs):
    """ TEST:   tile source image
    RESULT:     rnd -> img has no correlation to rnd -> upsample -> img
    """
    seed = seed if seed is not None else random.randint(0,1000)
    out = []
    for tile in tiles:
        print(f"sampling with scale x{tile}")
        out.append(sample_tile(config=config, seed=seed, tile=tile,
                            batch_size=batch_size, save=save, suffix=suffix, show=False, **kwargs))
    return out

def sample_tile(config=CONFIG, seed=None, tile=2,
                batch_size=1, save=True, suffix="", show=True, **kwargs):

    if seed is not None:
        torch.manual_seed(seed)
    trainer = load_trainer(config, **kwargs)
    model = trainer.ema_model# if ema else trainer.model

    shape = (batch_size, model.channels, model.image_size, model.image_size)
    img = torch.randn(shape)

    for _ in range(tile - 1):
        img = torch.cat((img, img), dim=2)
        img = torch.cat((img, img), dim=3)
    print(f"img size {tuple(img.shape)}")

    img = img.cuda()
    image = model.sample_from(img, scale=1, tile=tile)
    if save:
        if save:
            name = osp.join(trainer.results_folder,
                            f"tiles_{tile}")
            if seed is not None:
                name = f"{name}_seed{seed}"
            save = f"{name}{suffix}_sample-{trainer.step}.png"
    return to_image(image, save, show)

if __name__ == "__main__":
    """
    python train.py myconfigfile.yaml
    """
    if len(sys.argv) > 1:
        CONFIG = sys.argv[1]
    train(config=CONFIG)
