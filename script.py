"""

"""
import sys
import os.path as osp
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
from koreto import ObjDict

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import numpy_grid

def load_config(config="config_multitudes.yaml", data="/home/z/data/Self/Multitudes"):

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
        pprint(obj)

    return model_params, diffusion_params, train_params, data

def load_trainer(config="config_multitudes.yaml", milestone=None):
    model_params, diffusion_params, train_params, data = load_config(config)
    if milestone is not None:
        train_params['milestone'] = milestone
    model = Unet(**model_params).cuda()
    diffusion = GaussianDiffusion(model, **diffusion_params).cuda()
    return Trainer(diffusion, data, **train_params)

def train(config="config_multitudes.yaml", milestone=None):
    trainer = load_trainer(config, milestone)
    trainer.train()

def sample_vid(config="config_multitudes.yaml", milestone=None, batch_size=16, ema=True, play=True):
    """ loads 'milestone' from 'results_folder' spec'd in 'config_....yaml'
    renders video of sampling loop
    """
    trainer = load_trainer(config, milestone)
    model = trainer.ema_model if ema else trainer.model

    name = osp.join(trainer.results_folder,
                    f"{('ema' if ema else 'raw')}_sample-{trainer.step}.mp4")
    name = model.sample_vid(batch_size=batch_size, name=name, play=play)
    
    print(f"saved video to {name}")

def sample(config="config_multitudes.yaml", milestone=None, batch_size=16, height=None,
           width=None, seed=None, step=1, ema=True, save=False, suffix="", show=False):
    """
    test determinism by applying random seed
    >>> sample(seed=0, show=True, save=True)
    >>> sample(seed=0, show=True, save=True, suffix="_take2")
    # image generation on same milestone is deterministic

    >>> milestones = ['last', 20000, 15000, 37, 25, 1]
    >>> for m in milestones:
    >>>     sample(seed=0, show=False, save=True, milestone=m)

    test width, height: ok, a little weird
    >>> sample(seed=0, show=True, save=True, width=512, height=256, batch_size=8)
    # ema_sample-27951_256x512_seed0.png

    test step: no good
    >>> sample(seed=0, show=True, save=True, batch_size=16, step=2)
    """
    trainer = load_trainer(config, milestone)
    model = trainer.ema_model if ema else trainer.model

    image = numpy_grid(model.sample(batch_size=batch_size, height=height, width=width, seed=seed, step=step))

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
        name = f"{name}{suffix}.png"
        im = Image.fromarray(image)
        im.save(name)

    if show:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    return image

if __name__ == "__main__":
    """
    python train.py myconfigfile.yaml
    """
    CONFIG = "config_multitudes.yaml" if len(sys.argv) < 2 else sys.argv[1]
    train(config=CONFIG)
