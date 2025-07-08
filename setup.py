from setuptools import setup, find_packages


setup(name='PFCS',
      version='1.0.0',
      description='Part-based Feature Complementary Denoising for Unsupervised Person Re-identification',
      author='Wang bin',
      author_email='wangbin@nuist.edu.cn',
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
