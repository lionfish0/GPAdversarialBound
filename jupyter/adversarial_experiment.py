from boundmixofgaussians import findpeak, compute_sum, compute_grad
import numpy as np
from GPAdversarialBound import getallchanges, zeromean_gaussian, getbound, AdversBound, compute_bounds
from GPAdversarialBound.logistic import get_logistic_result
from GPAdversarialDatasets import getMNISTexample, getbankexample, getcreditexample, getspamexample,getsynthexample
import argparse
import sys
import pickle

parser = argparse.ArgumentParser(description='Adversarial Bounds Computation.')
parser.add_argument('--ntrain', dest='ntrain', type=int, help='number of training points. Default=100', default=100)
parser.add_argument('--ntest', dest='ntest', type=int, help='number of test points. Default=200', default=200)
parser.add_argument('--dataset',dest='dataset',type=str, help='type of dataset: "mnist", "bank", "credit", "spam", "synth"', required=True)

#mnist related:
parser.add_argument('--scaling', dest='scaling', type=int, help='scaling factor for MNIST (number of times smaller)',default=None)
parser.add_argument('--split', dest='splittype', type=str, help='splittype for MNIST ("fiveormore" or digit pair, e.g. 01)', default=None)
parser.add_argument('--keepthreshold', dest='keepthreshold', type=int, help='pixel value keep threshold for MNIST',default=None)

parser.add_argument('--depth', dest='depth', type=int, help='Number of pixels to change',required=True)
parser.add_argument('--sparse', dest='sparse', type=int, help='Number of inducing inputs (leave this option out if you don\'t want to use the sparse approximation)',default=None)
parser.add_argument('--lsmin', dest='lsmin', type=float, help='Kernel lengthscale (min)',required=True)
parser.add_argument('--lsmax', dest='lsmax', type=float, help='Kernel lengthscale (max)',required=True)
parser.add_argument('--lssteps', dest='lssteps', type=int, help='Number of lengthscale steps. Default=10.',default=10)

parser.add_argument('--v', dest='v', type=float, help='Kernel variance',required=True)
parser.add_argument('--sigma', dest='sigma', type=float, help='Sigma (Gaussian noise variance, should be small as we\'re not using a Gaussian noise model!). Default=0.000001',default=0.000001)
parser.add_argument('--nstep', dest='nstep', type=int, help='Number of steps to slice each dimension',required=True)
parser.add_argument('--gridres', dest='gridres', type=int, help='Grid resolution. Default=50',default=50)
parser.add_argument('--griddim', dest='griddim', type=int, help='Grid dimensionality. Default=2',default=2)
parser.add_argument('--enhancestep', dest='enhancenstep', type=int, help='On enhancement, the number of steps', default=1)
parser.add_argument('--enhancecount', dest='enhancecount', type=int, help='On enhancement, the number of iterations. Default=0',default=0)
parser.add_argument('--saveto', dest='saveto', type=str, help='Where to save the data (e.g. data.pkl)',default=None)


args = parser.parse_args()

if args.saveto is None:
    print("WARNING: DATA NOT BEING SAVED!!!")
print(vars(args))
X = None
if args.dataset=='mnist':
    if (args.scaling is None) or (args.splittype is None) or (args.keepthreshold is None):
        print("Invalid/missing MNIST configuration parameters: scaling, split, keepthreshold")
        sys.exit(1)
    fullX,Y = getMNISTexample(scalingfactor=args.scaling,Ntraining=args.ntrain+args.ntest,splittype=args.splittype) #4
    keep = np.max(fullX,0)>args.keepthreshold
    X = fullX[:,keep]
if args.dataset=='bank':
    X,Y = getbankexample()
if args.dataset=='spam':
    X,Y = getspamexample()
if args.dataset=='credit':
    X,Y = getcreditexample()
if X is None:
    print("Invalid dataset choice '%s'." % args.dataset)
    sys.exit(1)
    
for x in X.T:
    low=np.sort(x)[10]
    high=np.sort(x)[-10] 
    x[x<low]=low
    x[x>high]=high
X=X-np.min(X,0)
X=X/(np.max(X,0)+1)
X = X*1.0
Y = Y*1.0
Y[Y==0]=-1
Xtest = X[args.ntrain:(args.ntrain+args.ntest),:]
Ytest = Y[args.ntrain:(args.ntrain+args.ntest),:]
Xtrain = X[0:args.ntrain,:]
Ytrain = Y[0:args.ntrain,:]




allresults = []
for ls in np.exp(np.linspace(np.log(args.lsmin),np.log(args.lsmax),args.lssteps)): #np.linspace(args.lsmin, args.lsmax, args.lssteps):
    print("Length Scale: %0.5f" % ls)
    results, m, sparsem, accuracy, abCI = compute_bounds(Xtrain,Ytrain,Xtest,Ytest,args.depth,args.sparse, ls, args.v, args.sigma, args.nstep,args.gridres, args.griddim,(args.enhancenstep, args.enhancecount))
    allresults.append({'config':vars(args),'ls':ls,'results':results,'m':m,'sparsem':sparsem,'accuracy':accuracy,'abCI':abCI})

log_results = get_logistic_result(Xtrain,Ytrain,Xtest,Ytest)   

if args.saveto is not None:
    results_object = {'gp':allresults,'logistic':log_results}
    pickle.dump(results_object,open(args.saveto,'wb'))