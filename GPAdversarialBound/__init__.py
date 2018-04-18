import numpy as np
from boundmixofgaussians import zeromean_gaussian_1d, zeromean_gaussian, findbound, PCA

def overlap(hypercube_starts,hypercube_ends,splitcube_index,f,ignoredim=None):
    """
    Test whether a pair of hypercubes overlap
    
    hypercube_starts, hypercube_ends = arrays of hypercube corners
    splitcube_index and f are the indices of the two cubes
    ignoredim = a dimension that we'll delete.
    
    Example:
        starts = [np.array([0,0]),np.array([1,0])]
        ends = [np.array([1,1]),np.array([2,1])]
        
            overlap(starts,ends,0,1) 
                returns False as the two squares don't overlap
        
        however
        
            overlap(starts,ends,0,1,0)
                returns True as, ignoring the 0th dimension the two squares
                do overlap.
    """
    s1 = hypercube_starts[f]
    e1 = hypercube_ends[splitcube_index]
    s2 = hypercube_starts[splitcube_index]
    e2 = hypercube_ends[f]
    if ignoredim is not None:
        s1 = np.delete(s1,ignoredim)
        e1 = np.delete(e1,ignoredim)
        s2 = np.delete(s2,ignoredim)
        e2 = np.delete(e2,ignoredim)
    return np.all(s1<e1) and np.all(e2>s2)

def splitcubes(hypercube_starts,hypercube_ends,forwardpaths,backwardpaths,splitcube_index,split_dim,diff_dim):
    """Given a list of starts and ends of all cubes, splits a hypercube, updating
    various lists.
    
    Note, the list of forward and backward paths through the cubes:
    these are linked lists of orderings that one could take passing
    from one cube to another along one axis (diff_dim).
    
    - forwardpaths and backwardpaths:
        typically we'd create two hypercubes to start with, one that's zero width
        and one that occupies the whole domain. This is just for convenience later.
        
        E.g. hypercube_starts = [[0,0],[0,0]]
             hypercube_ends = [[0,5],[5,5]]
             (diff_dim = 0)
             
             forwardpaths = [[1],[]]
             backwardpaths = [[],[0]]
        
        The forwardspaths from the 0th cube is to the 1st cube. From the 1st cube
        is to no where. The inverse is true of the backwardpaths.
        
        We have to keep track of forward and backward, so when splits happen we can
        quickly step back to the cubes affected.
    
    - splitcube_index specifies the cube to split
    
    - split_dim = the dimension to split in
    
    - diff_dim = the direction we're keeping track of with the paths.
    
    NOTE: Manipulates matrices in place."""
    start = hypercube_starts[splitcube_index]
    end = hypercube_ends[splitcube_index]
    splitpoint = (end[split_dim]+start[split_dim])/2
    #if we're splitting in the direction we'll be building a path along
    #(i.e. diff_dim) then we need to make the path longer,
    #otherwise we need to add other paths.
    if split_dim == diff_dim:
        hypercube_starts.append(start.copy())
        hypercube_starts[-1][split_dim] = splitpoint
        hypercube_ends.append(end.copy())
        hypercube_ends[splitcube_index][split_dim] = splitpoint
        oldforwardpaths = forwardpaths.copy()
        forwardpaths.append(forwardpaths[splitcube_index].copy())
        forwardpaths[splitcube_index] = [len(forwardpaths)-1]
        backwardpaths.append([splitcube_index])
        for f in oldforwardpaths[splitcube_index]:
            for i,b in enumerate(backwardpaths[f]):
                if b==splitcube_index:
                    backwardpaths[f][i] = len(oldforwardpaths)
    else:
        newindex = len(forwardpaths) #this will be the index of the new item
        hypercube_starts.append(start.copy())
        hypercube_starts[newindex][split_dim] = splitpoint
        hypercube_ends.append(end.copy())
        hypercube_ends[splitcube_index][split_dim] = splitpoint
        
        fps = forwardpaths[splitcube_index].copy()
        forwardpaths[splitcube_index] = []
        forwardpaths.append([])
        
        bps = backwardpaths[splitcube_index].copy()
        backwardpaths[splitcube_index] = []
        backwardpaths.append([])
        
        for f in fps:
            backwardpaths[f].remove(splitcube_index)
        for b in bps:
            forwardpaths[b].remove(splitcube_index)
         
        #Splitting cube 'splitcube_index' (creating cube at 'newindex')
        for f in fps: 
            if overlap(hypercube_starts,hypercube_ends,splitcube_index,f,ignoredim=diff_dim):
                forwardpaths[splitcube_index].append(f)
                backwardpaths[f].append(splitcube_index)
            if overlap(hypercube_starts,hypercube_ends,newindex,f,ignoredim=diff_dim):
                forwardpaths[newindex].append(f)
                backwardpaths[f].append(newindex)
        for b in bps:

            if overlap(hypercube_starts,hypercube_ends,splitcube_index,b,ignoredim=diff_dim):
                backwardpaths[splitcube_index].append(b)
                forwardpaths[b].append(splitcube_index)
            if overlap(hypercube_starts,hypercube_ends,newindex,b,ignoredim=diff_dim):
                backwardpaths[newindex].append(b)
                forwardpaths[b].append(newindex)
    assert len(forwardpaths)==len(backwardpaths)
    assert len(forwardpaths)==len(hypercube_starts)
    assert len(forwardpaths)==len(hypercube_ends)

def getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v):
    """
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls)
   
    Given a hypercube, specified by hypercube_start and hypercube_end
    what are the bounds on the greatest changes to the peak of each gaussian, in
    the sum of weighted Gaussians specified by EQcentres and EQweights. For example
    if we have a square from [0,0] to [1,1], and just one EQcentres at [0.5,0.5] of
    weight 2, if the gaussian equals 1 at [0,0.5] & [1,0.5], then the startchange will equal 1
    midchange will equal 0 (as over the whole square the mean hasn't changed), and
    0 at endchange (this is an upper bound, and so although it could be negative, it
    can't be positive). innerchange will equal 1 (going from 0 to 0.5).
    
    d = dimension we're looking at changing over, and ls = lengthscale.
    
    ls, v = lengthscale and variance of kernel
    
    Importantly we want several results:
    
     startchange = the amount the function can change from a point on the starting plane
        of the hypercube, to any point in the hypercube (along the d axis)
    
     midchange = the change over the whole width of the hypercube (along the d axis)
    
     endchange = the amount the function can change from any point in the hypercube to
        a point on the endplane (along the d axis)
    
     innerchange = the amount the function can change /within/ the hypercube (along the d axis)
     """
    
    #get the highest and lowest values of an EQ between the start and end, along dimension d
    startvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_start[d],ls,v )
    endvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_end[d],ls, v)

    #if the peak is inside the cube we keep the peak weight, otherwise it's not of interest
    midvals = EQweights.copy()
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
  
    return startchange, midchange, endchange, innerchange

def sig(z):
    """Logistic sigmoid function"""
    return 1/(1+np.exp(-z))
    
def sig_grad(z):
    """Gradient of the logistic sigmoid function"""
    return sig(z)*(1-sig(z))    
        
def getlogisticgradientbound(EQcentres,EQweights,hypercube_start,hypercube_end,ls,v,gridspacing=0.1):
    """
        EQcentres = the training locations
        EQweights = the weights of the Gaussians in the mixtures of gaussians (i.e. the alpha vector = k^-1 y)
        hypercube_start,hypercube_end = hypercube corners
        ls = lengthscale
    
    If we're doing classification then we need to find the change in the posterior, not the change in the 
    latent function. To do this we need the absolute value of the function as the logistic's gradient
    depends on it. However, we can't just transform the changes as this makes them non-gaussian etc.
    
    Instead we find a lower bound on how much the posterior changes wrt the latent function.
    
    To do that we consider the logistic function's derivative:  sigma_grad(z) = sigma(z)(1-sigma(z))
    
    We can find the largest value of z in the hypercube and the smallest (most negative).
    If they're opposite signs then the steepest gradient in the hypercube is sigma_grad(0)
    If not, then we find the smallest absolute value of both of them, and use that:
                           sigma_grad(min(abs(minv),abs(maxv)))
    
    Returns a value that is the lower bound on the gradient
    """
    
    if np.any(hypercube_start==hypercube_end): return 0.25 #handle any hypercubes that are flat
            
    maxv = findbound(EQcentres, EQweights, ls, v, EQcentres.shape[1], gridspacing, hypercube_start, hypercube_end)[0]
    minv = -findbound(EQcentres, -EQweights, ls, v, EQcentres.shape[1], gridspacing, hypercube_start, hypercube_end)[0]

    if np.sign(maxv)!=np.sign(minv):
        boundgrad = sig_grad(0)
    else:  
        minval = min(np.abs(maxv),np.abs(minv))
        boundgrad = sig_grad(minval)
        
    return boundgrad
    

def getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls,v,gridspacing,logistic_transform=False):
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
        startchange, midchange, endchange, innerchange = getchanges(EQcentres,EQweights,hypercube_start,hypercube_end,d,ls,v)
        
        #New code for handling conversion to classification
        if logistic_transform:
            logisticgradbound = getlogisticgradientbound(EQcentres,EQweights,hypercube_start,hypercube_end,ls,v,gridspacing=gridspacing)
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
        
        ##debug printing###
        #print("hypercube_start")
        #print(hypercube_start.shape)
        #print("d")
        #print(d)
        #print("EQpeak")
        #print(EQpeak.shape)
        #print("s,e")
        #print(s.shape,e.shape)
        #print("part")
        #print(part.shape)
        #print(EQpeak_incube.shape)
        ###################
        
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
def getbound(EQcentres,hypercube_start,hypercube_end,d,ls,v,change,gridspacing=0.1,fulldim=False,forceignorenegatives=False,dimthreshold=3):
    """
    EQcentres = training point locations
    hypercube_start,hypercube_end = corners of the hypercube
    d = dimension over which we flatten the search
    ls,v = lengthscale and variance of kernel
    change = the training 'y' value for each training point
    gridspacing= the resolution of the search grid (default 0.1)
    fulldim= Over 3d the algorithm falls back to using a low-dimensional linear manifold to reduce the
     search grid volume. Set to True to over-ride this behaviour. Default=False
    """
    EQcentres_not_d = np.delete(EQcentres,d,1)
    hc_start_not_d = np.delete(hypercube_start,d)
    hc_end_not_d = np.delete(hypercube_end,d)
    if np.all(hc_start_not_d==hc_end_not_d): return 0

    return findbound(EQcentres_not_d,change,ls=ls,v=v,d=EQcentres_not_d.shape[1],gridspacing=gridspacing,gridstart=hc_start_not_d-gridspacing,gridend=hc_end_not_d+gridspacing,fulldim=fulldim,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)

def getallpaths(forwardpaths):    
    def getpaths(currentcube,path):
        #global paths
        newpath = path.copy()
        newpath.append(currentcube)
        if len(forwardpaths[currentcube])==0:
            paths.append(newpath)
        for f in forwardpaths[currentcube]:
            getpaths(f,newpath)
    
    paths = []
    getpaths(0,[])
    return paths

def compute_full_bound(X,Y,sigma,ls,v,diff_dim,dims,cubesize,splitcount=5,gridspacing=0.2,forceignorenegatives=False,dimthreshold=3,K=None,logistic_transform=False):
    """
    maxval, hypercube_starts, hypercube_ends, maxseg, EQweights = compute_full_bound(X,Y,sigma,ls,diff_dim,dims,cubesize,splitcount=5,gridspacing=0.2,forceignorenegatives=False,dimthreshold=3)

    X,Y = training inputs and outputs.
    sigma = standard deviation of noise.
    ls,v = lengthscale and variance of RBF kernel
    diff_dim = dimension over which we're interested in the bound on change in the posterior mean.
    dims = number of dimensions, probably X.shape[1].
    cubesize = either a scalar or a vector describing the size of the volume over which we're testing. E.g. if there are three inputs, one can range between 0 and 1 and the other two between 0 and 255, then this is a vector [1,255,255].
    splitcount = number of times we split the space (you'll end up with this many hypercubes + 2)
    gridspacing = the spacing used during the bound computation (recommend about 10% of the cubesize, so for a 3d space we have ~1000 test points). A smaller value will improve the bound, but will be slower.
    forceignorenegatives=False this is for testing to see effect on a low-dimensional training set of ignoring negatives (this is necessary anyway at higher dimensions as we use a low-dimensional PCA approximation to search for an upper bound over for the mixture of gaussians.
    dimthreshold=3 more will potentially improve the bound but will be slower.
    
    returns:
    maxval = the largest positive change in the direction of dimension 'diff_dim'.
    hypercube_starts, hypercube_ends = corners of the hypercubes (shows how it's been split)
    maxseg = the sequence of hypercubes that led to the largest change in the mean.
    EQweights = the weights of the Gaussians in the mixtures of gaussians (i.e. the alpha vector = k^-1 y)
    """
    assert dims == X.shape[1]
    
    #First compute the representer-theorem equivalent of the GP mean function
    # i.e. f = sum_i k(x_i,x_*) alpha_i
    #save the training locations in EQcentres
    #and the 'alpha' weights in 'EQweights'
    
    if K is None: #compute it ourselves,
        K = np.empty([len(X),len(X)])
        for i in range(len(X)):
            K[i,:] = zeromean_gaussian(X-X[i,:],ls=ls,v=v) #using this function for the EQ RBF
            
    alpha = np.linalg.inv(K+np.eye(len(X))*(sigma**2)) @ Y
    EQcentres = X
    EQweights = alpha[:,0]


    #initialise the hypercubes with one spanning cubesize (and another 0-width one
    #at the start of the search dimension).
    hypercube_starts = [np.zeros(dims)*1.0,np.zeros(dims)*1.0]
    if type(cubesize)==np.ndarray:
        hypercube_ends = [cubesize.copy(),cubesize.copy()]
    else:
        hypercube_ends = [np.full(dims,cubesize)*1.0,np.full(dims,cubesize)*1.0]
    hypercube_ends[0][diff_dim] = 0
    forwardpaths = [[1],[]]
    backwardpaths = [[],[0]]


    #split up the hypercubes a bit (need to make this more intelligent, e.g. not just split the middle)
    print("Splitting...")
    wholecubechanges = None
    wholecubecounts = None 
    for splitit in range(splitcount):
        split_index = 1
        if wholecubechanges is not None: #switched from using innerchanges to using wholecubechanges
            split_index = np.argmax(wholecubechanges)
        split_dim = np.argmax(hypercube_ends[split_index]-hypercube_starts[split_index])
        splitcubes(hypercube_starts,hypercube_ends,forwardpaths,backwardpaths,split_index,split_dim,diff_dim)
        startchanges, midchanges, endchanges, innerchanges,wholecubechanges,wholecubecounts = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,diff_dim,ls,v,gridspacing,logistic_transform)


    #get all the straightline paths (in the direction we're differentiating) from the start and end of the hypercube
    #e.g. hypercube 0->1->3 and 0->2
    paths = getallpaths(forwardpaths)


    
    #get all the segments paths. so 0->1->3 & 0->2 has 8 possible inside paths:
    #0, 1, 3, 0->1, 0->1->3, 1->3, 0, 2, 0->2
    print("Computing Paths...")
    pathsegments = []
    for p in paths:
        for start in range(1,len(p)):
            pathsegments.append([p[start]])
            for end in range(start+1,len(p)):
                pathsegments.append(p[start:end+1])


    #just keep unique sequences so we don't test them twice            
    unique_pathsegments = [list(x) for x in set(tuple(x) for x in pathsegments)]


    #check all these path segments, get the maximum bound from all these.
    maxval = -np.inf
    print("Checking Segments...")
    for it,seg in enumerate(unique_pathsegments):
        print("\n%d/%d" % (it,len(unique_pathsegments)),end="")
        if len(seg)==1: #if it's just one segment (don't iterate over, instead use the innerchanges - as we'll just be moving within this cube).
            b = getbound(EQcentres,hypercube_starts[seg[0]],hypercube_ends[seg[0]],diff_dim,ls,v,innerchanges[seg[0],:],gridspacing=gridspacing,fulldim=False,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)
        else: #otherwise we need to combine the relevant start, mid and end parts
              #e.g. 0->1->3, add the starts from 0, the mids from 1 and the ends for 3.
            search_hypercube_start = np.full_like(hypercube_starts[0],-np.inf)
            search_hypercube_end = np.full_like(hypercube_ends[0],np.inf)
            cube_diff_dim_start = np.inf
            cube_diff_dim_end = -np.inf
            for s in seg:
                print(".",end="")
                search_hypercube_start = np.max(np.array([search_hypercube_start,hypercube_starts[s]]),0)
                search_hypercube_end = np.min(np.array([search_hypercube_end,hypercube_ends[s]]),0)
                
                cube_diff_dim_start = min(cube_diff_dim_start,hypercube_starts[s][int(diff_dim)])
                cube_diff_dim_end = max(cube_diff_dim_end,hypercube_ends[s][int(diff_dim)])                
            search_hypercube_start[int(diff_dim)] = cube_diff_dim_start
            search_hypercube_end[int(diff_dim)] = cube_diff_dim_end
            
            #not sure if this should happen! TODO.
            if np.any(search_hypercube_start>=search_hypercube_end):
                print(search_hypercube_start,search_hypercube_end)
                print("start>end")
                b = 0
            else: #we add together the starts, mids and ends, and treat it as a d-1 dimensional
                  #mixture of gaussians problem.
                changes = startchanges[seg[0],:] + np.sum(midchanges[seg[1:-1],:],0) + endchanges[seg[-1],:]
                b = getbound(EQcentres,search_hypercube_start,search_hypercube_end,diff_dim,ls,v,changes,gridspacing=gridspacing,fulldim=False,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)
                
        print(seg,b)
        if b>maxval:
            maxval = b
            maxseg = seg
            
            
            
    return maxval, hypercube_starts, hypercube_ends, maxseg, EQweights
    
def testing():
    """Testing various methods.
    overlap       - tested
    splitcubes    - tested
    getchanges    - tested
    getallchanges - tested
    getbound      -(basically a wrapper for findbound from another module)
    getallpaths   - tested
    compute_full_bound - untested!
    """
    
    ############test overlap############
    hstarts = [np.array([0,2]),np.array([0,1]),np.array([3,0])]
    hends = [np.array([3,4]),np.array([3,2]),np.array([4,4])]

    #4._______
    #3| . 0 | |
    #2|_____|2|
    #1. | 1 | |
    #0. |___|_|
    # 0 1 2 3 4
        
    assert overlap(hstarts,hends,0,0)==True
    assert overlap(hstarts,hends,1,1)==True
    assert overlap(hstarts,hends,2,2)==True
    assert overlap(hstarts,hends,0,1)==False
    assert overlap(hstarts,hends,0,1,0)==False
    assert overlap(hstarts,hends,0,1,1)==True
    assert overlap(hstarts,hends,1,0,0)==False
    assert overlap(hstarts,hends,1,0,1)==True
    assert overlap(hstarts,hends,0,2,1)==False
    assert overlap(hstarts,hends,0,2,0)==True
    assert overlap(hstarts,hends,1,2,1)==False
    assert overlap(hstarts,hends,1,2,0)==True
    
    ############test splitcubes############    
    hstarts = [np.array([0,0])]
    hends = [np.array([4,4])]
    forwardpaths = [[]]
    backwardpaths = [[]]
    splitcube_index = 0
    split_dim = 0
    diff_dim = 0
    splitcubes(hstarts,hends,forwardpaths,backwardpaths,splitcube_index,split_dim,diff_dim)
    assert np.all(hstarts[0]==np.array([0,0]))
    assert np.all(hends[0]==np.array([2,4]))
    assert forwardpaths == [[1],[]]
    assert backwardpaths == [[],[0]]
    splitcube_index = 1
    split_dim = 1
    diff_dim = 0
    splitcubes(hstarts,hends,forwardpaths,backwardpaths,splitcube_index,split_dim,diff_dim)
    splitcube_index = 0
    split_dim = 1
    diff_dim = 0
    splitcubes(hstarts,hends,forwardpaths,backwardpaths,splitcube_index,split_dim,diff_dim)
    #print(hstarts,hends,forwardpaths,backwardpaths)
    # ____________
    #|     |     |
    #|  3  |  2  |
    #|_____|_____|
    #|     |     |
    #|  0  |  1  |
    #|_____|_____|
    #
    #forward should be [[1],[],[],[2]] backward [[],[0],[3],[]]
    assert forwardpaths == [[1],[],[],[2]]
    assert backwardpaths == [[],[0],[3],[]]
    
    ############test getchanges############
    EQcentres = np.array([[-1,0],[-1,1]])
    EQweights = np.array([1.0,1.0])
    hypercube_start = np.array([0,0])
    hypercube_end = np.array([4,4])
    d = 0
    ls = 2.0
    v = 1.0
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v)


    #startchange: if this was a starting cube, starting at 0, with a peak at -1, it can't increase over the cube so we use zero
    assert np.all(startchange == 0)
    from boundmixofgaussians import zeromean_gaussian_1d, zeromean_gaussian, findbound, PCA
    #midchange: if it's a middle cube, we integrate over the whole of the cube
    assert np.all(midchange == zeromean_gaussian_1d(5,2)-zeromean_gaussian_1d(1,2))
    #similar to startchange, if it's the last cube, the whole function has a negative gradient over the cube, so
    #the largest value is zero.
    assert np.all(endchange == 0)
    #as above, for changes that start and end within the cube
    assert np.all(innerchange == 0)

    EQcentres = np.array([[1,0],[2,0],[3,0],[3,2]])
    EQweights = np.array([1.0,1.0,1.0,-1.0])
    hypercube_start = np.array([0,0])
    hypercube_end = np.array([4,4])
    d = 0
    ls = 2.0
    v = 1.0
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v)

    #if we are starting in this cube, then, (for a centre at x=1) although there's a +ve gradient from 0 to 1,
    #there's more negative on the far side, so it's "best" to start on the boundary
    #for a centre at x=2, the positive and negative cancel.
    #for a centre at x=3, there's an overall increase in the function

    assert startchange[0] == 0
    assert startchange[1] == 0
    assert startchange[2] == zeromean_gaussian_1d(1,2,1)-zeromean_gaussian_1d(3,2,1)

    #for a centre at x=3, with negative weight, the largest increase is for x=3 to x=4. (as it goes from negative to less-negative)
    assert startchange[3]==1-zeromean_gaussian_1d(1,2,1)

    #midchange: 0th one should be negative, as its left boundary is higher than its right (as the peak is more
    #to the left). 1st is zero (as the values at the two boundaries are equal). 2nd is positive. 3rd is like the
    #0th but inverted & flipped.
    assert midchange[0] == zeromean_gaussian_1d(3,2,1)-zeromean_gaussian_1d(1,2,1)
    assert midchange[1] == 0
    assert midchange[2] == -(zeromean_gaussian_1d(3,2,1)-zeromean_gaussian_1d(1,2,1))
    assert midchange[3] == (zeromean_gaussian_1d(3,2,1)-zeromean_gaussian_1d(1,2,1))

    #endchange
    #0th: will be the change from zeromean_gaussian_1d(0,2)-zeromean_gaussian_1d(1,2)
    assert endchange[0] == 1-zeromean_gaussian_1d(1,2,1)
    assert endchange[1] == 1-zeromean_gaussian_1d(2,2,1)
    assert endchange[2] == 1-zeromean_gaussian_1d(3,2,1)
    assert endchange[3] == 0

    #innerchange
    assert innerchange[0] == 1-zeromean_gaussian_1d(1,2,1)
    assert innerchange[1] == 1-zeromean_gaussian_1d(2,2,1)
    assert innerchange[2] == 1-zeromean_gaussian_1d(3,2,1)
    assert innerchange[3] == 1-zeromean_gaussian_1d(1,2,1)


    EQcentres = np.array([[1,0],[2,0],[3,0],[3,2]])
    EQweights = np.array([1.0,1.0,1.0,-1.0])
    hypercube_starts = np.array([[0,0]])
    hypercube_ends = np.array([[4,4]])
    d = 0
    ls = 2.0
    v = 1.0
    ############test getallchanges############
    startchanges, midchanges, endchanges, innerchanges,wholecubechanges = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls,v)

    startchanges, midchanges, endchanges, innerchanges

    assert np.all(startchanges == np.array([0, 0, zeromean_gaussian_1d(1,2,1)-zeromean_gaussian_1d(3,2,1), 1-zeromean_gaussian_1d(1,2,1)]))
    assert np.all(midchanges == np.array([zeromean_gaussian_1d(3,2,1)-zeromean_gaussian_1d(1,2,1),0, -(zeromean_gaussian_1d(3,2,1)-zeromean_gaussian_1d(1,2,1)),(zeromean_gaussian_1d(3,2,1)-zeromean_gaussian_1d(1,2,1))]))
    assert np.all(endchange==np.array([1-zeromean_gaussian_1d(1,2,1),1-zeromean_gaussian_1d(2,2,1),1-zeromean_gaussian_1d(3,2,1),0]))
    assert np.all(innerchange==np.array([1-zeromean_gaussian_1d(1,2,1),1-zeromean_gaussian_1d(2,2,1),
                                         1-zeromean_gaussian_1d(3,2,1),1-zeromean_gaussian_1d(1,2,1)]))
    
    ############test getallpaths############
    
    #0->1 & 2
    #   1-> 2
    #       2->3 4 5
    #            4->6
    #so the paths are:
    #0123,01246,0125,023,0246,025
    assert getallpaths([[1,2],[2],[3,4,5],[],[6],[],[]])==[[0, 1, 2, 3],[0, 1, 2, 4, 6],[0, 1, 2, 5],[0, 2, 3],[0, 2, 4, 6],[0, 2, 5]]

    
