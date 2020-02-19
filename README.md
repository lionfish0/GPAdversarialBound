# GPAdversarialBound
A module for computing bounds the scale of change a perturbation can cause to GP predictions

## Installation
      conda update scipy
      sudo apt-get install python-dev
      sudo apt-get install python3-dev
      sudo apt-get install libblas-dev libatlas-base-dev
      pip install git+https://github.com/SheffieldML/GPy
      pip install git+https://github.com/lionfish0/hypercuboid_integrator.git
      pip install git+https://github.com/lionfish0/boundmixofgaussians.git
      pip install git+https://github.com/lionfish0/GPAdversarialDatasets.git
      git clone https://github.com/lionfish0/GPAdversarialBound.git
      cd GPAdversarialBound
      pip install mnist
      pip install -e .
      
## Usage

See <a href="https://github.com/lionfish0/GPAdversarialBound/blob/master/jupyter/Full%20Sparse%20Version%20Demo.ipynb">jupyter notebook</a> demo.

## Commandline tool

To run Adversarial Bound experiments you will need to install modules (listed below) and then the easiest way to run the actual bound algorithm is either by looking at the jupyter notebook example in the GPAdversarialBound folder, or by running adversarial_experiment.py command line tool.

For example:

      python adversarial_experiment.py --ntrain 100 --dataset mnist --lsmin 10.0 --lsmax 10.0 --lssteps 1 --v 1 --nstep 5 --depth 1 --scaling 10 --split 01 --keepthreshold 15 --gridres 20 --griddim 2 --sparse 4 --saveto new_mnist_sparse_new2.pkl

runs the algorithm on 100 training points from mnist, with just a length scale of 10 and variance of 1. The nsteps is the number of slices. The depth is the number of dimensions to search at once (typically one). The scaling, split and keepthreshold are MNIST specific: scaling the images, what pairs to compare [0 and 1 in this case] and what level of pixel value a pixel must reach in the training data to be included.

It outputs a pickle file.

If you open this file, and look in

import pickle
p = pickle.load(open('new_mnist_sparse_new2.pkl','rb'))
p['gp'][0]['results'][0][0]

this has an upper bound of how much each dimension (sorted in order) can influence the latent function value.

p['gp'][0]['abCI']
this is the confidence interval between the 5th and 95th percentile training data.
