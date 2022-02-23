"""

"""
import sys
import os.path as osp
from pprint import pprint
from koreto import ObjDict

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

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

def run(config="config_multitudes.yaml"):
    model_params, diffusion_params, train_params, data = load_config(config)
    model = Unet(**model_params).cuda()
    diffusion = GaussianDiffusion(model, **diffusion_params).cuda()
    trainer = Trainer(diffusion, data, **train_params)
    trainer.train()

if __name__ == "__main__":
    """
    python train.py myconfigfile.yaml
    """
    config = "config_multitudes.yaml" if len(sys.argv) < 2 else sys.argv[1]
    run(config=config)
