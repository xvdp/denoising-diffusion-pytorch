""" @ saving utils
"""
import math
import torch
from torchvision import utils
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


def numpy_grid(x, pad=0, nrow=None, uint8=True):
    """ thin wrap to make_grid to return frames ready to save to file
    args
        pad     (int [0])   same as utils.make_grid(padding)
        nrow    (int [None]) # defaults to horizonally biased rectangle closest to square
        uint8   (bool [True]) convert to img in range 0-255 uint8
    """
    x = x.clone().detach().cpu()
    nrow = nrow or int(math.sqrt(x.shape[0]))
    x = ((utils.make_grid(x, nrow=nrow, padding=pad).permute(1,2,0) - x.min())/(x.max()-x.min())).numpy()
    if uint8:
        x = (x*255).astype("uint8")
    return x


def to_image(image, save=True, show=True, pad=1):
    """ util tensor to image, show, save
    """
    image = numpy_grid(image, pad=pad)
    if save:
        im = Image.fromarray(image)
        im.save(save)
        print(f"saved image {save}")
    if show:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    return image

def strf (x):
    """ format time output
    strf = lambda x: f"{int(x//86400)}D{int((x//3600)%24):02d}:{int((x//60)%60):02d}:{int(x%60):02d}s"
    """
    days = int(x//86400)
    hours = int((x//3600)%24)
    minutes = int((x//60)%60)
    seconds = int(x%60)
    out = f"{minutes:02d}:{seconds:02d}"
    if hours or days:
        out = f"{hours:02d}:{out}"
        if days:
            out = f"{days}_{out}"
    return out

# pylint: disable=no-member
def open_image(path, channels=3, image_size=128):
    """ open img with same transforms as ddpm dataset
    """
    if isinstance(path, (list, tuple)):
        return torch.cat([open_image(p, channels=channels, image_size=image_size)
                          for p in path])
    img = Image.open(path)
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
    return transform(img)[None,...]
