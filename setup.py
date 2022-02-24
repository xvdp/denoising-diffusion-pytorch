from setuptools import setup, find_packages

def get_required():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
  name = 'denoising-diffusion-pytorch',
  packages = find_packages(),
  version = '0.7.1',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/denoising-diffusion-pytorch',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  install_requires=[
    'einops',
    'numpy',
    'pillow',
    'torch',
    'torchvision',
    'tqdm',
    "koreto @ git+https://github.com/xvdp/koreto.git"
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

"""
 # https://github.com/pypa/pip/issues/3939
install_requires = [
      'somerepo @ https://api.github.com/repos/User/somerepo/tarball/v0.1.1',
],
'koreto @ https://api.github.com/repos/xvdp/koreto' #/tarball/v0.0.7'
"""