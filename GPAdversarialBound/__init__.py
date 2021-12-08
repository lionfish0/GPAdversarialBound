import numpy as np
from boundmixofgaussians import zeromean_gaussian_1d, zeromean_gaussian, findbound, PCA, findpeak, compute_sum, compute_grad
from scipy.optimize import minimize
import GPy
from itertools import combinations
from time import time
import itertools
import bisect

def getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v):
    """
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v)
   
    Given a hypercube, specified by hypercube_start and hypercube_end
    what are the bounds on the greatest changes to the peak of each gaussian, in
    the sum of weighted Gaussians specified by EQcentres and EQweights. For example
    if we have a square from [0,0] to [1,1], and just one EQcentres at [0.5,0.5] of
    weight 2, if the gaussian equals 1 at [0,0.5] & [1,0.5], then the endchange will equal 1
    midchange will equal 0 (as over the whole square the mean hasn't changed), and
    0 at startchange (this is an upper bound, and so although it could be negative, it
    can't be positive). innerchange will equal 1 (going from 0 to 0.5).
    
    d = dimension we're looking at changing over, and ls = lengthscale.
    
    ls, v = lengthscale and variance of kernel
    
    Importantly we want several results. It returns:
    
     endchange = the amount the function can change from a point on the starting plane
        of the hypercube, to any point in the hypercube (along the d axis)
    
     midchange = the change over the whole width of the hypercube (along the d axis)
    
     startchange = the amount the function can change from any point in the hypercube to
        a point on the endplane (along the d axis)
    
     innerchange = the amount the function can change /within/ the hypercube (along the d axis)
     """
    
    #get the highest and lowest values of an EQ between the start and end, along dimension d
    startvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_start[d],ls,v )
    endvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_end[d],ls, v)

    #if the peak is inside the cube we keep the peak weight, otherwise it's not of interest
    midvals = EQweights.copy()*v
    midvals[(EQcentres[:,d]<hypercube_start[d]) | (EQcentres[:,d]>hypercube_end[d])] = np.nan

    #starting cube: we're interested in the biggest increase possible from any location to the end of the hypercube
    #this is bound by the change from the values at the start to the values at the end
    startchange = np.nanmax(np.array([endvals - startvals,endvals-midvals]),0)
    startchange[startchange<0] = 0

    #ending cube: we're interested in the biggest increase possible from the start to anywhere inside
    endchange = np.nanmax(np.array([midvals - startvals,endvals - startvals]),0)
    endchange[endchange<0] = 0
    
    innerchange = np.nanmax(np.array([midvals - startvals,endvals-midvals,endvals-startvals]),0)
    innerchange[innerchange<0] = 0
    #middle cube: we're interested in the total change from start to end
    midchange = endvals - startvals #negative values can be left in this
    #print(startvals,midvals,endvals)
    return startchange, midchange, endchange, innerchange
    
    
def getchangesTwoWeights(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v):
    """
    startchange, midchange, endchange, innerchange = getchangesTwoWeights(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v)
   
    Given a hypercube, specified by hypercube_start and hypercube_end
    what are the bounds on the greatest changes to the peak of each gaussian, in
    the sum of weighted Gaussians specified by EQcentres and EQweights.
    
    EQweights in this case is a tuple of two items: an upper weight and a lower,
    the change reported here is the LARGEST increase such that one picks the weight
    that maximises this increase.
    
    For example
    if we have a square from [0,0] to [1,1], and just one EQcentres at [0.5,0.5] of
    weights [2,1], if the upper gaussian equals 1 at [0,0.5] & [1,0.5] and the lower
    gaussian equals 0.5 at [0,0.5] & [1.0.5], then the startchange will equal 0.5
    midchange will equal 0.5, and endchange 1.5. innerchange will equal 1.5
    (going from location 0 to 0.5).
    
    d = dimension we're looking at changing over, and ls = lengthscale.
    
    ls, v = lengthscale and variance of kernel
    
    Importantly we want several results. It returns:
    
     endchange = the amount the function can change from a point on the starting plane
        of the hypercube, to any point in the hypercube (along the d axis)
    
     midchange = the change over the whole width of the hypercube (along the d axis)
    
     startchange = the amount the function can change from any point in the hypercube to
        a point on the endplane (along the d axis)
    
     innerchange = the amount the function can change /within/ the hypercube (along the d axis)
     """
    
    #get the highest and lowest values of an EQ between the start and end, along dimension d
    #print("EQweights",EQweights)
    startvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_start[d],ls,v )
    endvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_end[d],ls, v)

    #if the peak is inside the cube we keep the peak weight, otherwise it's not of interest
    midvals = EQweights.copy()*v
    
    startvals = np.sort(startvals,axis=0)
    endvals = np.sort(endvals,axis=0)
    midvals = np.sort(midvals,axis=0)
    
    #print("startvals\n",startvals,"\nmidvals\n",midvals,"\nendvals\n",endvals)
    #print("!!")
    #print(EQcentres, EQweights)
    #print(".>>")
    #print("startvals:",startvals,"midvals:",midvals,"endvals:",endvals)    
    midvals[:,(EQcentres[:,d]<hypercube_start[d]) | (EQcentres[:,d]>hypercube_end[d])] = np.nan
    #print(midvals)
    #print("S",startvals,"\nM",midvals,"\nE",endvals)
    #print(startvals.shape,midvals.shape,endvals.shape)
    #starting cube: we're interested in the biggest increase possible from any location to the end of the hypercube
    #this is bound by the change from the values at the start to the values at the end
    startchange = np.nanmax(np.array([endvals[0,:] - startvals[0,:],endvals[0,:]-midvals[0,:]]),0)
    startchange[startchange<0] = 0

    #ending cube: we're interested in the biggest increase possible from the start to anywhere inside
    endchange = np.nanmax(np.array([midvals[1,:] - startvals[0,:],endvals[1,:] - startvals[0,:],startvals[1,:] - startvals[0,:]]),0)
    endchange[endchange<0] = 0
    #print(np.array([midvals[1,:] - startvals[0,:],endvals[1,:]-midvals[0,:],endvals[1,:]-startvals[0,:]]))
    innerchange = np.nanmax(np.array([midvals[1,:] - startvals[0,:],endvals[1,:]-midvals[0,:],endvals[1,:]-startvals[0,:],endvals[1,:]-endvals[0,:],startvals[1,:]-startvals[0,:],midvals[1,:]-midvals[0,:]]),0)
    innerchange[innerchange<0] = 0
    #middle cube: we're interested in the total change from start to end
    midchange = endvals[0,:] - startvals[0,:] #negative values can be left in this
    #print(startvals,midvals,endvals)
    return startchange, midchange, endchange, innerchange

def getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls,v,logistic_transform=False): #removed gridres from parameters
    """
    Basically computes the startchanges, midchanges, endchanges, innerchanges for
    all the hypercubes. We also compute wholecubechanges and wholecubecount
    as we use the former to select the hypercube to split. wholecubechanges
    is roughly the amount that the function can change in the cube - but unlike
    midchanges or innerchanges, it takes into account how close to the cube
    the EQcentres are. Summary it combines the start, mid, end and inner changes.
    
    startchanges, midchanges, endchanges, innerchanges, wholecubechanges, wholecubecount = 
         getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls)
    
    EQcentres = the training locations
    EQweights = the weights of the Gaussians in the mixtures of gaussians (i.e. the alpha vector = k^-1 y)
    hypercube_starts,hypercube_ends = hypercube corners
    d = dimension we're moving alon.
    ls = lengthscale
    
    returns:
    startchanges, midchanges, endchanges, innerchanges = lists of hypercubes,
       describe the maximum increase in the GP mean if this were a start
       cube, a middle cube, an end cube or the only cube.
    wholecubechanges, wholecubecount = metrics useful for selecting which cube to split
    """
    startchanges =[]
    midchanges = []
    endchanges = []
    innerchanges = []
    wholecubechanges = []
    wholecubecount = []
    for hypercube_start, hypercube_end in zip(hypercube_starts,hypercube_ends):
        assert hypercube_start.shape[0]==EQcentres.shape[1], "The number of columns in EQcentres should be equal to the dimensions in the hypercube."
        assert hypercube_end.shape[0]==EQcentres.shape[1], "The number of columns in EQcentres should be equal to the dimensions in the hypercube."
        startchange, midchange, endchange, innerchange = getchangesTwoWeights(EQcentres,EQweights,hypercube_start,hypercube_end,d,ls,v)
        
        #New code for handling conversion to classification
        if logistic_transform:
            assert False, "needs testing"
            ###logisticgradbound = getlogisticgradientbound(EQcentres,EQweights,hypercube_start,hypercube_end,ls,v,gridres=gridres)
            #print("LOGISTIC TRANSFORM BOUND: %0.4f" % logisticgradbound)
            startchange *= logisticgradbound
            midchange *= logisticgradbound
            endchange *= logisticgradbound
            innerchange *= logisticgradbound
        
        startchanges.append(startchange)
        midchanges.append(midchange)
        endchanges.append(endchange)
        innerchanges.append(innerchange)
        
        
        #remove axis we differentiated along
        EQpeak = np.delete(EQcentres,d,1)
        EQpeak_incube = EQpeak.copy()
        s = np.delete(hypercube_start,d)
        e = np.delete(hypercube_end,d)
        
        
        #This bit of code is for computing a heuristic about the effect of this
        #cube - so we can decide which cube to slice.
        
        #where is the nearest point to a maximum?
        part = (EQpeak - (s+e)/2)
        
        for i in range(len(s)):
            EQpeak_incube[(innerchange>0) & (EQpeak_incube[:,i]<s[i]),i] = s[i]
            EQpeak_incube[(innerchange>0) & (EQpeak_incube[:,i]>e[i]),i] = e[i]       
            EQpeak_incube[(innerchange<0) & (part[:,i]>0),i] = s[i] #& (EQpeak_incube[:,i]>s[i])
            EQpeak_incube[(innerchange<0) & (part[:,i]<=0),i] = e[i] #& (EQpeak_incube[:,i]<e[i])        
        cubebound = np.sum(innerchange*zeromean_gaussian(EQpeak_incube-EQpeak,ls=ls,v=v))

    
        wholecubecount.append(np.all((EQcentres>=hypercube_start) & (EQcentres<hypercube_end),1))
        wholecubechanges.append(cubebound)

    wholecubecount = np.sum(np.array(wholecubecount),1)                
    #wholecubecount = np.array(wholecubecount)            
    wholecubechanges = np.array(wholecubechanges)
    startchanges = np.array(startchanges)
    midchanges = np.array(midchanges)
    endchanges = np.array(endchanges)
    innerchanges = np.array(innerchanges)
    return startchanges, midchanges, endchanges, innerchanges, wholecubechanges, wholecubecount

#here we go through the combinations of starts and ends in this simple line of hypercubes
def getbound(EQcentres,hypercube_start,hypercube_end,d,ls,v,change,gridres=10,fulldim=False,forceignorenegatives=False,dimthreshold=3):
    """
    
    EQcentres = training point locations
    hypercube_start,hypercube_end = corners of the hypercube
    d = dimension over which we flatten the search
    ls,v = lengthscale and variance of kernel
    change = the training 'y' value for each training point
    gridres = the resolution of the search grid (default 10)
    fulldim= Over 3d the algorithm falls back to using a low-dimensional linear manifold to reduce the
     search grid volume. Set to True to over-ride this behaviour. Default=False
     
    Returns a bound on the peak of the potential increase within the hypercube specified by the
     hypercube_start and hypercube_end parameters, along dimension d.
    """
    EQcentres_not_d = np.delete(EQcentres,d,1)
    hc_start_not_d = np.delete(hypercube_start,d)
    hc_end_not_d = np.delete(hypercube_end,d)
    if np.all(hc_start_not_d==hc_end_not_d): return 0
    bound = findbound(EQcentres_not_d,change,ls=ls,d=EQcentres_not_d.shape[1],gridres=gridres,gridstart=hc_start_not_d,gridend=hc_end_not_d,fulldim=fulldim,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)

    return bound

class AdversarialBoundException(Exception):
    pass

class NoAvailableDimensionsToSearch(AdversarialBoundException):
    pass

class AdversBound:
    def __init__(self):
        self.hires_debug = []
        self.count_compute = 0
    
    def duplicate(self,ab,boxstart,boxend,nsteps):
        """
        Copy the configuration etc from one adversarial bound object into another.
        """
        self.maxlist = ab.maxlist
        self.EQcentres = ab.EQcentres.copy()
        self.EQweights = ab.EQweights.copy()
        self.ls = ab.ls
        self.sigma = ab.sigma
        self.dimthreshold = ab.dimthreshold
        self.v = ab.v
        self.nsteps = nsteps
        self.dims = ab.dims
        self.gridres = ab.gridres
        self.initialise_hypercube_system(boxstart,boxend,nsteps)
        
    def configure(self,EQcentres,EQweights,ls,sigma,v,boxstart,boxend,nsteps,gridres,k,dimthreshold,maxlist=100):
        """
        nsteps = number of hypercubes along each axis.
        gridres = resolution of searchgrid on inner call to boundmixofgaussians.
        """
        self.maxlist = maxlist
        self.EQcentres = EQcentres
        self.EQweights = EQweights
        self.ls = ls
        self.sigma = sigma
        self.dimthreshold=dimthreshold
        self.v = v
        self.nsteps = nsteps
        self.dims = EQcentres.shape[1]
        self.gridres = gridres
        self.k = k
        self.initialise_hypercube_system(boxstart,boxend,nsteps)
        
    def initialise_hypercube_system(self,boxstart,boxend,nsteps):
        """
        Sets up the hypercubes, computes the amount of change within each hypercube for each type of movement.
        """
        self.startpositions = tuple([np.linspace(s,e,nstep+1)[0:-1] for s,e,nstep in zip(boxstart,boxend,nsteps)])
        self.endpositions = tuple([np.linspace(s,e,nstep+1)[1:] for s,e,nstep in zip(boxstart,boxend,nsteps)])
        self.hypercube_starts = np.array(list(itertools.product(*self.startpositions)))
        self.hypercube_ends = np.array(list(itertools.product(*self.endpositions)))
        
        self.boxstart = boxstart
        self.boxend = boxend
        #print(K @ self.alpha)
        
        assert self.dims == self.EQcentres.shape[1]
        
        #startchanges = if this were a cube at the start of the path...
        #midchanges = if this were a cube in the middle of the path...
        #endchanges = if this were a cube at the end of the path...
        #innerchanges = if the path is completely within this cube...
        #...what is the largest increase (aligned with each training point)
        #in each dimension?
        
        
        #TODO We've switched from Nones to 0s. Hope this is ok...
        self.startchanges = [[] for _ in range(self.dims)]
        self.midchanges = [[] for _ in range(self.dims)]
        self.endchanges = [[] for _ in range(self.dims)]
        self.innerchanges = [[] for _ in range(self.dims)]
        self.negstartchanges = [[] for _ in range(self.dims)]
        self.negmidchanges = [[] for _ in range(self.dims)]
        self.negendchanges = [[] for _ in range(self.dims)]
        self.neginnerchanges = [[] for _ in range(self.dims)]
     
        #for a unit kernel... (it is scaled later)
        
        
        #Build list of lengthscales and weights
        if type(self.k)==GPy.kern.RBF:
            self.apprx_ls = np.array([1.0]) #the lengthscales of the list of approximating kernels
            self.apprx_ws = np.array([[1.0],[1.0]])     
            
               
        if type(self.k)==GPy.kern.Exponential:
            self.apprx_ls = np.array([ 0.213704,   0.875284,   2.189295,   4.62008 ,9.212761, 113.691967, 140.42632 , 148.05403 ])  
            apprx_ws = []
            apprx_ws.append(np.array([0.06829387, 0.14179143, 0.26536302, 0.11843095, 0.22876908, 0.18288816, 0.00226348]))
            apprx_ws.append(np.array([ 0.06529387,  0.13729143,  0.25786302,  0.11843095,  0.22876908, 0.18288816, -0.00083652]))
            self.apprx_ls = np.array([ 0.038469  ,  0.16337993,  0.42350206,  0.91900753,  0.91912203,1.84561768, 17.07884635])

            self.apprx_ws = np.array(apprx_ws)  
        
        #Loop over these...
        for l,w in zip(apprx_ls,apprx_ws.T):        
            #C)make startchanges etc sum over these...
            for d in range(self.dims):
                
                newWs = np.array([w[0]*self.EQweights,w[1]*self.EQweights])
                sc,mc,ec,ic,_,_ = getallchanges(self.EQcentres,newWs,self.hypercube_starts,self.hypercube_ends,d,l*self.ls,self.v)
                #print(ic)
                self.startchanges[d].append(sc)
                self.midchanges[d].append(mc)
                self.endchanges[d].append(ec)
                self.innerchanges[d].append(ic)
                 
                newWs = np.array([-w[0]*self.EQweights,-w[1]*self.EQweights])
                sc,mc,ec,ic, _, _ = getallchanges(self.EQcentres,newWs,self.hypercube_starts,self.hypercube_ends,d,l*self.ls,self.v)
                self.negstartchanges[d].append(sc)
                self.negmidchanges[d].append(mc)
                self.negendchanges[d].append(ec)
                self.neginnerchanges[d].append(ic)

    def compute(self,depth,steps = None, availdims=None,availsteps = None, hires=1):
        """
        Compute upper bound on the change 'depth' perturbations can cause to the prediction.
        
        depth = number of dimensions to modify
        steps = the steps that describe the starting location, e.g. [[]]
        availdims = the dimensions one can modify at each depth, e.g. [[8],[0]] means that one can
                    only move in the 8th dimension first, then along the 0th dimension.
                    This can either just be a straightforward list of dimensions [0,1,2,3]
                    or can be a list of lists, which specifies the dimensions that are available to take
                    at each step [[0,1],[2,3]]
        availsteps = if availsteps is not specified then we just use the object's nsteps list
                     to get the number of steps that we should split this dimension (d) into.
                     If it is specified, then each iteration of the recursion can have a
                     different number of steps for each dimension.
                     E.g. availsteps = [[[3,1],[1,3]],[[3],[3]]]
                     this means in the first recursive step the two dimensions can each have 3 steps to test
                     in the second iteration each dimension only gets one step to test
       
        
        abhires.findtop([9, 0, 0, 0, 0, 0, 0, 0, 9],2,[[8],[0]],[[[],[],[],[],[],[],[],[],[0,1,2,3,4]],[[0,1,2,3,4],[],[],[],[],[],[],[],[]]])
        """
        
        
        if steps is None:
            steps = tuple([np.arange(nstep) for nstep in self.nsteps])
        if availdims is None:
            availdims = list(range(self.dims))
        biggestb = 0
        sequences = []
        sequence_bounds = []
        
        for c in itertools.product(*steps):
            seqs, seqbs = self.findtop(list(c),depth,availdims,availsteps)
            for seqb, seq in zip(seqbs,seqs):
                
                seq.insert(0,[c])
                idx = bisect.bisect(sequence_bounds,seqb)
                sequences.insert(idx,seq)
                sequence_bounds.insert(idx,seqb) 
            if len(sequence_bounds)>self.maxlist:
                sequence_bounds = sequence_bounds[-self.maxlist:]
                sequences = sequences[-self.maxlist:]
        if hires>1: #this basically isn't used any more.
            print("USING HIRES: THIS IS DEPRECATED")
            hires_bounds = []
            for seqs in sequences:
                hires_seq_bounds, highres_seqs = self.compute_high_res_bound(depth,seqs,scaling=hires)
                
                hires_bounds.append(hires_seq_bounds[-1])
                
            hires_bounds = np.array(hires_bounds)
        else:
            hires_bounds = None
        #print("done")
        return sequence_bounds, sequences, hires_bounds

    def get_index_of_cell(self,cell):
        index = 0
        for ns,sc in zip(self.nsteps,cell):
            index = (index*ns)+sc
        return index
        
    def compute_indexes_of_line(self,startcell,endcell,d):
        """
        Compute the indices 
        """
        #confirm the coordinates only differ by the dimension we are purported to
        #be moving in.
        
        assert np.all(endcell<self.nsteps), "endcell outside of range of possible numbers of steps"
        assert np.all(startcell<self.nsteps), "startcell outside of range of possible numbers of steps"
        assert np.all(np.array(startcell)>=0), "startcell outside of range of possible numbers of steps"
        assert np.all(np.array(endcell)>=0), "endcell outside of range of possible numbers of steps"
        
        for i,x in enumerate(np.array(startcell)-np.array(endcell)):
            if i!=d:
                assert x==0

        startindex = self.get_index_of_cell(startcell)
        endindex = self.get_index_of_cell(endcell)
        if endindex>startindex:
            res = np.arange(startindex,endindex+1,int(np.prod(self.nsteps[d+1:])))
        else:
            res = np.arange(startindex,endindex-1,-int(np.prod(self.nsteps[d+1:])))
        
        assert np.all(res<np.prod(self.nsteps)), "Cell outside valid range of indices"
        assert np.all(res>=0)
        return res
    
    
    def compute_bound(self,idx,d,positive,include_start=True,include_end=True):
        """
        idx = list of indicies of cells in the path
        d = dimension we're moving along (along path)
        positive = whether to use the changes computed for the normal training
          values of 'y' or their negatives. This flip allows us to consider
          the paths in the opposite direction
        include_start/include_end = whether to incorporate the start cell and end cell, or just compute up to the boundary
          
          TODO: As we consider both directions like this we don't need steps being in both directions cover this?"""
        self.count_compute+=1
        if positive:
            innerchange = self.innerchanges
            startchange = self.startchanges
            midchange = self.midchanges
            endchange = self.endchanges
        else:
            innerchange = self.neginnerchanges
            startchange = self.negstartchanges
            midchange = self.negmidchanges
            endchange = self.negendchanges

        #pass list of indices
        #if only one cell exists, we just compute the innerchanges, otherwise we need to add up all the influences
        idx = np.sort(idx)
        #print("Hypercube corners")
        #print(self.hypercube_starts[idx[0]],self.hypercube_ends[idx[-1]])
        if len(idx)==1:
            if include_start and include_end:
                changes = innerchange[d][idx[0],:]
                #b = getbound(self.EQcentres,self.hypercube_starts[idx[0]],self.hypercube_ends[idx[0]],d,self.ls,self.v,,gridres=self.gridres,dimthreshold=self.dimthreshold)
            else:
                if not include_start:
                    changes = endchange[d][idx[0],:]
                if not include_end:
                    changes = startchange[d][idx[0],:]
                if not include_start and not include_end:
                    changes = midchange[d][idx[0],:]
        else: #if len(idx)>1:
            if include_start and include_end:
                changes = startchange[d][idx[0],:] + np.sum(midchange[d][idx[1:-1],:],0) + endchange[d][idx[-1],:]
            if not include_start:
                changes = np.sum(midchange[d][idx[0:-1],:],0) + endchange[d][idx[-1],:]
            if not include_end:
                changes = startchange[d][idx[0],:] + np.sum(midchange[d][idx[1:],:],0)
            if not include_start and not include_end:
                changes = np.sum(midchange[d][idx,:],0)
        b = getbound(self.EQcentres,self.hypercube_starts[idx[0]],self.hypercube_ends[idx[-1]],d,self.ls,self.v,changes,gridres=self.gridres,dimthreshold=self.dimthreshold)
        print("::::")
        print(b)
        return b

    def findtop(self,c,depth,availdims,availsteps = None):
        """
        Finds the largest sum of bounds from cell c, with depth number of steps
        availdims specifies the directions we are still able to go in.
        
        availdims = this can either just be a straightforward list of dimensions [0,1,2,3]
                    or can be a list of lists, which specifies the dimensions that are available to take
                    at each step [[0,1],[2,3]]
        availsteps = if availsteps is not specified then we just use the object's nsteps list
                     to get the number of steps that we should split this dimension (d) into.
                     (note that this allows us to have a different number of steps still for each
                     dimension as this is a list).
                     If it is specified, then each iteration of the recursion can have a
                     different number of steps for each dimension.
                     E.g. availsteps = [[[3,1],[1,3]],[[3],[3]]]
                     this means in the first recursive step the two dimensions can each have 3 steps to test
                     in the second iteration each dimension only gets one step to test.
                     
        returns:
                    
        """
        #print("   "*(2-depth),'findtop(',"c=",c,",depth=",depth,",availdims=",availdims,",availsteps=",availsteps,')')
        sequences = []
        sequence_bounds = []
        
        if len(availdims)<1:
            raise NoAvailableDimensionsToSearch
                    
        #we can pass a list of available dims to use at each recursive step
        if isinstance(availdims[0],list):
            availdim = availdims[0]
        else:
            availdim = availdims

        if len(availdim)<1:
            raise NoAvailableDimensionsToSearch
        
        
        for i,d in enumerate(availdim): #loop over the available dimensions we can take
            
            #if we're given a list of lists, then the next recursive step should be given the remaining lists
            #otherwise we've just been given a list of dimensions. We only need to search the (i+1)th dimension
            #and onwards, as we don't need to test both 0->1->2 and 0->2->1, etc.
            
            if isinstance(availdims[0],list): 
                newavaildims = availdims[1:]
            else:
                newavaildims = availdims[i+1:]
                #if there are fewer available dimensions to travel down than depth of search
                #we should stop, as we can't fulfill this.                
                if len(newavaildims)<depth-1: 
                    break
                    #print(len(newavaildims),depth-1)            
            
            #if availsteps is not specified then we just use the object's nsteps list to get the number of steps
            #that we should split this dimension (d) into. If it is specified, then each iteration of the recursion
            #can have a different number of steps for each dimension.
            if availsteps is None:
                steps = np.arange(self.nsteps[d])
                newavailsteps = None
            else:
                steps = availsteps[0][d]
                newavailsteps = availsteps[1:]
            
            #For each step (in dimension d)
            for x in steps:
                #create new cell location at destination
                newc = c.copy()
                newc[d]=x
                #so now e.g. c = [1,2,3], newc = [1,5,3], d = 1
                pathindicies = self.compute_indexes_of_line(c,newc,d)
                #print("   "*(2-depth),pathindicies)
                if x==c[d]: #we need to check both positive and negative directions inside same hypercube
                    stepbound = max(self.compute_bound(pathindicies,d,True),self.compute_bound(pathindicies,d,False))
                else:
                    stepbound = self.compute_bound(pathindicies,d,x>=c[d])
                if depth>1:
                    seqs, seq_bounds = self.findtop(newc,depth-1,newavaildims,newavailsteps)
                else:
                    seqs = [[]]
                    seq_bounds = [0]

                for j in range(len(seq_bounds)):
                    #print(".",end="")
                    seq_bounds[j] = seq_bounds[j]+stepbound
                    seqs[j].insert(0,[newc,stepbound,d])
                    
                    idx = bisect.bisect(sequence_bounds,seq_bounds[j])
                    sequences.insert(idx,seqs[j])
                    sequence_bounds.insert(idx,seq_bounds[j])
                #keeps list a reasonable size
                if len(sequence_bounds)>self.maxlist:
                    sequence_bounds = sequence_bounds[-self.maxlist:]
                    sequences = sequences[-self.maxlist:]
                    
        return sequences, sequence_bounds

    def compute_high_res_bound(self,depth,sequence,scaling=4):
        starts = []
        ends = []
        changingdims = []
        changingsteps = []


        #we want to find the bounds of this path, typically this will
        #be in a low-dimensional manifold as depth<<dims
        #
        #for each depth stage of the sequence, we look up the hypercube, and 
        #also take a note of the dimension it travels in and the number of steps
        #after this we have four variables:
        # changingdims = list of dimensions that are travelled down, in order
        # changingsteps = list of the step index we reach along each dimension
        # boxstart = starting coordinate of hypercube we're pathing over..
        # boxend = ending coordinate of hypercube
        for seq in sequence:
            #we get the index of the starting cell
            index = self.compute_indexes_of_line(list(seq[0]),list(seq[0]),0)
            if len(seq)>2:
                changingdims.append(seq[2])
                changingsteps.append(seq[0][seq[2]])
            starts.append(self.hypercube_starts[index][0])
            ends.append(self.hypercube_ends[index][0])
        boxstart = np.min(np.array(starts),0)
        boxend = np.max(np.array(ends),0)

        #for now we expand those dimensions that we are searching over the whole
        #of the original domain - this is for computational simplicity, as it means
        #we can be assured that there are a whole number of hires cuboids inside
        #each original hypercuboid, etc.
        for d in changingdims:
            boxstart[d]=0
            boxend[d]=self.boxend[d]

        abhires = AdversBound()

        nsteps = [1]*self.dims
        for d in changingdims: nsteps[d] = self.nsteps[d]*scaling
        abhires.duplicate(self,boxstart,boxend,nsteps)
        abhires.gridres=self.gridres
        abhires.dimthreshold=self.dimthreshold

        startsteps = [[0]]*self.dims
        startcell = np.array(sequence[0][0])
        for d in changingdims: startsteps[d] = list(np.arange(startcell[d]*scaling,(startcell[d]*scaling)+scaling))

        availsteps = []
        for d,s in zip(changingdims,changingsteps):
            part_availsteps = [[]]*self.dims
            part_availsteps[d]=list(np.arange(s*scaling,s*scaling+scaling))
            availsteps.append(part_availsteps)
        availdims = [[d] for d in changingdims]
        
        
        hires_sequence_bounds, hires_sequences, _ = abhires.compute(depth,startsteps,availdims,availsteps)
        self.hires_debug.append({'abhires':abhires,'depth':depth,'startsteps':startsteps,'availdims':availdims,'availsteps':availsteps,'hires_sequence_bounds':hires_sequence_bounds, 'hires_sequences':hires_sequences})
        return hires_sequence_bounds, hires_sequences
    
    def compute_empirical_lowerbound(self,depth,N=100000):
        assert False, "NEEDS TO DEPEND ON KERNEL"
        maxchange = 0
        for it in range(50):
            ps = np.random.rand(int(N/50),self.dims)
            newps = ps.copy()
            for d in range(depth):
                newps[:,np.random.randint(dims)] = np.random.rand(int(N/50))
            for p,newp in zip(ps,newps):
                r = self.EQweights @ zeromean_gaussian(self.EQcentres-p,self.ls,self.v)
                newr = self.EQweights @ zeromean_gaussian(self.EQcentres-newp,self.ls,self.v)
                maxchange = max(maxchange,np.abs(newr-r))
        return maxchange
    
    def compute_CI(self,CI=0.95):
        #assert False, "NEEDS TO DEPEND ON KERNEL"
        outputs = []
        for x in self.EQcentres:
            r = self.EQweights @ zeromean_gaussian(self.EQcentres-x,self.ls,self.v)
            outputs.append(r)
        outputs=np.sort(np.array(outputs))
        top = outputs[int(len(outputs)*CI)]
        bottom = outputs[int(len(outputs)*(1-CI))]
        return bottom,top




def compute_bounds(Xtrain,Ytrain,Xtest,Ytest,depth, sparse,ls,v,sigma,nstep_per_dim,gridres=50,dimthreshold=2,enhance=None,k=None):
    """
    Xtrain,Ytrain = training data (classification)
    
    sparse = None if not sparse otherwise a number for number of inducing points
    ls,v = lengthscale and variance of kernel
    
    sigma = Gaussian noise (should really be zero for standard GP Classification)
    nstep_per_dim = how much to divide up each dimension
    gridres, dimthreshold = parameters for bound approximation.
    enhance = tuple of (nstep_per_dim, num_iterations) [set to None for no enhance]
    
    Returns:
     results,
        this contains a list of each of the dimension combinations that can be taken,
        e.g. a 6 dimensional grid with depth of 2 has 15 combinations:
        [0,1],[0,2]...[1,2],[1,3]...[2,3]...[3,4]...[4,5] = 5+4+3+2+1
        each item contains a tuple of three thing:
        sequence_bounds = the values for each path for that dimension combination.
        With a gridres of two, and a depth of 2, there are 16 combinations:
        A-A-A, A-A-B, A-A-C, A-B-B, A-B-D, A-C-C, A-C-D, B-B-B, B-B-D, B-D-D, B-D-C, C-C-C, C-C-D, C-D-D, D-D-D, C-D-B.? 
        sequences contains the details,
        hires_bounds - unused.
        The first is just
    """
    if sparse is None:
        sparse=False
    else:
        assert isinstance(sparse, int)

    dims = Xtrain.shape[1]
    boxstart = [0.0]*dims
    boxend = [1.0]*dims

    #print("Xtrain.shape:")
    #print(Xtrain.shape)

    ####TODO THIS ISN'T FAST ENOUGH WHEN WE NEED IT SPARSE
    #if k is None: k = GPy.kern.RBF(dims)
    m = GPy.models.GPClassification(Xtrain,Ytrain,k)
    m.inference_method = GPy.inference.latent_function_inference.Laplace()
    m.kern.lengthscale.fix(ls)
    m.kern.variance.constrain_bounded(v,v+1e-4)
    m.optimize()

    if sparse:
        print("Sparse...")
        K = m.kern.K(Xtrain,Xtrain)
        alpha = np.linalg.inv(K+np.eye(len(Xtrain))*(sigma**2)) @ m.inference_method.f_hat[:,0] #Ytrain[:,0]
        sparsem = GPy.models.SparseGPRegression(Xtrain,alpha[:,None],num_inducing=sparse) #todo chose number of inducing inputs
        sparsem.kern.lengthscale.fix(ls)
        sparsem.kern.variance.constrain_bounded(v,v+1e-4)
        sparsem.optimize()
        Z = np.array(sparsem.Z.tolist())
        Kuf = m.kern.K(Z,Xtrain)
        Kuu = m.kern.K(Z,Z)
        Sigma = np.linalg.inv((sigma**-2)*Kuf@Kuf.T + Kuu)
        alpha = (sigma**-2)*Sigma @ Kuf @ m.inference_method.f_hat[:,0]
        abXs = Z
    else:
        print("not sparse...")
        m = GPy.models.GPClassification(Xtrain,Ytrain,k)
        m.inference_method = GPy.inference.latent_function_inference.Laplace()
        m.kern.lengthscale.fix(ls)
        m.kern.variance.constrain_bounded(v,v+1e-4)
        m.optimize()
        K = m.kern.K(Xtrain,Xtrain)
        alpha = np.linalg.inv(K+np.eye(len(Xtrain))*(sigma**2)) @ m.inference_method.f_hat[:,0] #Ytrain[:,0]
        abXs = Xtrain
        sparsem = None


    #This didn't make it faster!
    #blocks = [[i,i+1] for i in range(0,dims,2)]
    #blocks[-1]=[i for i in blocks[-1] if i<dims]
    blocks = [[i] for i in range(dims)] 
    print("Starting...")
    results = []
    #print(blocks)
    for combo in combinations(blocks,depth):
        print(".",end="",flush=True)
        combo_block = []
        for c in combo: combo_block.extend(c)
        nsteps = [1]*dims
        for b in combo_block: nsteps[b]=nstep_per_dim
        ab = AdversBound()
        ab.configure(abXs,alpha,ls,sigma,v,boxstart,boxend,nsteps,gridres,k,dimthreshold)
        #print(combo_block)
        res = ab.compute(depth,hires=1,availdims=combo_block)
        results.append([res[0]]) ###SAVING SPACE, instead of saving all of res
#        results.append(res)
    #print("Done")
    if enhance is not None:
        print("Enhancing...")
        new_nstep_per_dim=enhance[0]
        import time
        for it in range(enhance[1]):
            start = time.time()
            print(".",end="",flush=True)
            i = np.argmax([np.max(res[0]) for res in results])
            combo=list(combinations(blocks,depth))[i]
            combo_block = []
            for c in combo: combo_block.extend(c)
            nsteps = [1]*dims
            for b in combo_block: nsteps[b]=new_nstep_per_dim

            ab = AdversBound()
            ab.configure(abXs,alpha,ls,sigma,v,boxstart,boxend,nsteps,gridres,k,dimthreshold)
            newres = ab.compute(depth,hires=1,availdims=combo_block)
            print("%0.4f --> %0.4f" % (np.max(results[i][0]),np.max(newres[0])))
            end = time.time()
            if (end>start+1000):
               print("Excessive time, aborting enhancement")
               break

            if (np.max(results[i][0])<=np.max(newres[0])):
               print("Insufficient improvement")
               print("Increasing step count:")
               new_nstep_per_dim+=1
               print(new_nstep_per_dim)
            results[i] = [newres[0]] ###SAVING SPACE, instead of saving all of newres
            new_bounds = np.array([np.max(res[0]) for res in results])
            print("%0.4f %0.4f %0.4f" % (np.min(new_bounds),np.mean(new_bounds),np.max(new_bounds)))

    accuracy = (np.mean(((m.kern.K(Xtest,abXs)@alpha)>0.5)==(Ytest[:,0]>0)))
    #print("Done.")
    #print("accuracy: %0.2f" % accuracy)
    
    sortedresults = np.sort(m.predict_noiseless(Xtrain)[0][:,0])
    abCI = sortedresults[int(len(Xtrain)*(1-0.95))],sortedresults[int(len(Xtrain)*(1-0.05))]

    debug = {'abXs':abXs,'alpha':alpha}
    return results, m, sparsem, accuracy, abCI, debug #ab.compute_CI()
    #all_results.append([nstp,np.max([np.max(res[0]) for res in results]),end-start])
