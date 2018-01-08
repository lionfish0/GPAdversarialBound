from distutils.core import setup
setup(
  name = 'GPAdversarialBound',
  packages = ['GPAdversarialBound'],
  version = '1.01',
  description = 'A module for computing bounds the scale of change a perturbation can cause to GP predictions ',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/GPAdversarialBound.git',
  download_url = 'https://github.com/lionfish0/GPAdversarialBound.git',
  keywords = ['Gaussian Processes','Adversarial Examples','bounds','regression','classification'],
  classifiers = [],
  install_requires=['numpy','hypercuboid_integrator'],
)
