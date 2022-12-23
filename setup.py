from setuptools import setup, find_packages

setup(
  name = 'denoising-diffusion-pytorch',
  packages = find_packages(),
  version = '0.7.1.1',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'xvdpahlen@gmail.com based on lucidrains@gmail.com',
  url = 'https://github.com/xvdp/denoising-diffusion-pytorch',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  # install_requires=[
  #   'einops',
  #   'numpy',
  #   'pillow',
  #   'torch',
  #   'torchvision',
  #   'tqdm',
  #   "koreto @ git+https://github.com/xvdp/koreto.git",
  #   "vidi @ git+https://github.com/xvdp/vidi.git",
  # ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
