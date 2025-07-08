from setuptools import setup, find_packages


setup(name='PFCS',
      version='1.0.0',
      description='Part-based Feature Complementary Denoising for Unsupervised Person Re-identification',
      author='Tian qing, Wang bin',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Person Re-identification'
      ])
