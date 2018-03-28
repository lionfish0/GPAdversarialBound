import numpy as np
from boundmixofgaussians import zeromean_gaussian_1d, zeromean_gaussian, findbound, PCA

def overlap(hypercube_starts,hypercube_ends,splitcube_index,f,ignoredim=None):
    """
    Test whether a pair of hypercubes overlap
    hypercube_starts, hypercube_ends = arrays of hypercube corners
    splitcube_index and f are the indices of the two cubes
    ignoredim = a dimension that we'll delete.
    
    Example
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
    """Manipulates matrices in place."""
    start = hypercube_starts[splitcube_index]
    end = hypercube_ends[splitcube_index]
    splitpoint = (end[split_dim]+start[split_dim])/2
    #if we're splitting in the direction we'll be building a path along (i.e. diff_dim) then we need to make the path longer,
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
         
        #print("Splitting cube %d (newindex %d)" % (splitcube_index, newindex))
        #print("Forward paths")
        #print(fps)
        for f in fps:
            
            #print("Checking forward (splitindex=%d) %d" % (splitcube_index,f))
            if overlap(hypercube_starts,hypercube_ends,splitcube_index,f,ignoredim=diff_dim):
                #print("Appending %d to forwardpaths[%d]" % (f,splitcube_index))
                forwardpaths[splitcube_index].append(f)
                #print("Appending %d to backwardpaths[%d]" % (f,splitcube_index))
                backwardpaths[f].append(splitcube_index)
            #print("Checking forward (newindex=%d) %d" % (newindex,f))
            if overlap(hypercube_starts,hypercube_ends,newindex,f,ignoredim=diff_dim):
                #print("Appending %d to forwardpaths[%d]" % (f,newindex))
                forwardpaths[newindex].append(f)
                #print("Appending %d to backwardpaths[%d]" % (newindex,f))
                backwardpaths[f].append(newindex)
        #print("Backward paths")
        #print(bps)
        for b in bps:
            #print("Checking backward (splitindex=%d) %d" % (splitcube_index,b))
            if overlap(hypercube_starts,hypercube_ends,splitcube_index,b,ignoredim=diff_dim):
                #print("Appending %d to backwardpaths[%d]" % (b,splitcube_index))
                backwardpaths[splitcube_index].append(b)
                #print("Appending %d to forwardpaths[%d]" % (splitcube_index,b))
                forwardpaths[b].append(splitcube_index)
            #print("Checking backward (newindex=%d) %d" % (newindex,b))
            if overlap(hypercube_starts,hypercube_ends,newindex,b,ignoredim=diff_dim):
                #print("Appending %d to backwardpaths[%d]" % (b,newindex))
                backwardpaths[newindex].append(b)
                #print("Appending %d to forwardpaths[%d]" % (newindex,b))
                forwardpaths[b].append(newindex)
    assert len(forwardpaths)==len(backwardpaths)
    assert len(forwardpaths)==len(hypercube_starts)
    assert len(forwardpaths)==len(hypercube_ends)
#    for s,e in zip(hypercube_starts[1:],hypercube_ends[1:]):
#        assert np.all(e>s)


def getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls):
    #get the highest and lowest values of an EQ between the start and end, along dimension d

    startvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_start[d],ls )
    endvals = EQweights*zeromean_gaussian_1d(EQcentres[:,d]-hypercube_end[d],ls)

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

#def findchanges(EQcentres,EQweights,hypercube_start,hypercube_end,d,ls):
#    EQcentres_not_d = np.delete(EQcentres,d,1)
#    hc_start_not_d = np.delete(hypercube_start,d)
#    hc_end_not_d = np.delete(hypercube_end,d)
#    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d,ls)
#    #TODO I'm confused myself when the ignorenegatives should be set!
#    #startbound = findbound(EQcentres_not_d,startchange,ls,1,0.1,hc_start_not_d,hc_end_not_d,ignorenegatives=True)
#    #midbound = findbound(EQcentres_not_d,midchange,ls,1,0.1,hc_start_not_d,hc_end_not_d)
#    #endbound = findbound(EQcentres_not_d,endchange,ls,1,0.1,hc_start_not_d,hc_end_not_d,ignorenegatives=True)
#    #innerbound = findbound(EQcentres_not_d,innerchange,ls,1,0.1,hc_start_not_d,hc_end_not_d,ignorenegatives=True) #TODO note the d=1 in the params refers to the number of dims I think - need to make it vary depending on size of matrix, etc
#    return startchange, midchange, endchange, innerchange #startbound, midbound, endbound,innerbound

def getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls):
    """
    Combines the start, mid, end and inner changes.
    """
    startchanges =[]
    midchanges = []
    endchanges = []
    innerchanges = []
    wholecubechanges = []
    wholecubecount = []
    for hypercube_start, hypercube_end in zip(hypercube_starts,hypercube_ends):
        startchange, midchange, endchange, innerchange = getchanges(EQcentres,EQweights,hypercube_start,hypercube_end,d,ls)
        startchanges.append(startchange)
        midchanges.append(midchange)
        endchanges.append(endchange)
        innerchanges.append(innerchange)
        
        
        #remove axis we differentiated along
        EQpeak = np.delete(EQcentres,d,1)
        EQpeak_incube = EQpeak.copy()
        s = np.delete(hypercube_start,d)
        e = np.delete(hypercube_end,d)
        
        #where is the nearest point to a maximum?
        part = (EQpeak - (s+e)/2)

        for i in range(len(s)):
            EQpeak_incube[(innerchange>0) & (EQpeak_incube[:,i]<s[i]),i] = s[i]
            EQpeak_incube[(innerchange>0) & (EQpeak_incube[:,i]>e[i]),i] = e[i]       
            EQpeak_incube[(innerchange<0) & (part[:,i]>0),i] = s[i] #& (EQpeak_incube[:,i]>s[i])
            EQpeak_incube[(innerchange<0) & (part[:,i]<=0),i] = e[i] #& (EQpeak_incube[:,i]<e[i])        
        cubebound = np.sum(innerchange*zeromean_gaussian(EQpeak_incube-EQpeak,ls=ls))

    
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
def getbound(EQcentres,hypercube_start,hypercube_end,d,ls,change,gridspacing=0.1,fulldim=False,forceignorenegatives=False,dimthreshold=3):
    """
    EQcentres = training point locations
    hypercube_start,hypercube_end = corners of the hypercube
    d = dimension over which we flatten the search
    ls = lengthscale
    change = the training 'y' value for each training point
    gridspacing= the resolution of the search grid (default 0.1)
    fulldim= Over 3d the algorithm falls back to using a low-dimensional linear manifold to reduce the
     search grid volume. Set to True to over-ride this behaviour. Default=False
    """
    EQcentres_not_d = np.delete(EQcentres,d,1)
    hc_start_not_d = np.delete(hypercube_start,d)
    hc_end_not_d = np.delete(hypercube_end,d)
    if np.all(hc_start_not_d==hc_end_not_d): return 0
    
    
    
#    #print("Dimensionality: %d" % EQcentres.shape[1])
#    if EQcentres.shape[1]>3 and not fulldim:
#        print("Compacting to 3d manifold...")
#        lowd = 3
#        lowdX,evals,evecs,means = PCA(EQcentres_not_d.copy(),lowd)
#        ignorenegatives = True
#        print("NEED TO COMPUTE NEW LOCATIONS OF START AND END OF GRID IN LOW DIM MANIFOLD")
#        raise NotImplementedError
#    else:
#        lowdX = EQcentres_not_d
#        lowd = EQcentres_not_d.shape[1]
#        #print("Keeping current dimensionality (%d)" % lowd)
#        ignorenegatives = False
        
    #print("finding bound...")
    #print("Input Locations")
    #print(lowdX)
    #print("weights/changes")
    #print(change)
    #print("lengthscale")
    #print(ls)
    #print("dimensionality: %d" % lowd)
    #print("gridspacing: %0.2f" % gridspacing)
    #print("Start:")
    #print(hc_start_not_d-gridspacing)
    #print("End:")
    #print(hc_end_not_d+gridspacing)
    #print("ignorenegatives:")
    #print(ignorenegatives)
    
    return findbound(EQcentres_not_d,change,ls=ls,d=EQcentres_not_d.shape[1],gridspacing=gridspacing,gridstart=hc_start_not_d-gridspacing,gridend=hc_end_not_d+gridspacing,fulldim=fulldim,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)
    #return findbound(EQcentres_not_d,change,ls,EQcentres_not_d.shape[1],gridspacing,hc_start_not_d,hc_end_not_d,ignorenegatives=False)


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

def compute_full_bound(X,Y,sigma,ls,diff_dim,dims,cubesize,splitcount=5,gridspacing=0.2,forceignorenegatives=False,dimthreshold=3):
    K = np.empty([len(X),len(X)])
    for i in range(len(X)):
        K[i,:] = zeromean_gaussian(X-X[i,:],ls=ls) #using this function for the EQ RBF
    alpha = np.linalg.inv(K+np.eye(len(X))*(sigma**2)) @ Y
    EQcentres = X
    EQweights = alpha[:,0]


    hypercube_starts = [np.zeros(dims)*1.0,np.zeros(dims)*1.0]
    if type(cubesize)==np.ndarray:
        hypercube_ends = [cubesize.copy(),cubesize.copy()]
    else:
        hypercube_ends = [np.full(dims,cubesize)*1.0,np.full(dims,cubesize)*1.0]
    hypercube_ends[0][diff_dim] = 0
    forwardpaths = [[1],[]]
    backwardpaths = [[],[0]]

    print("Splitting...")
    wholecubechanges = None
    wholecubecounts = None 
    #split up the hypercubes a bit (need to make this more intelligent)
    for splitit in range(splitcount):
        split_index = 1
        if wholecubechanges is not None: #switched from using innerchanges to using wholecubechanges
            split_index = np.argmax(wholecubechanges)

        #if wholecubecounts is not None:   
        #    split_index = np.argmax(wholecubecounts)
        split_dim = np.argmax(hypercube_ends[split_index]-hypercube_starts[split_index])
        splitcubes(hypercube_starts,hypercube_ends,forwardpaths,backwardpaths,split_index,split_dim,diff_dim)
        startchanges, midchanges, endchanges, innerchanges,wholecubechanges,wholecubecounts = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,diff_dim,ls)

    #get all the straightline paths (in the direction we're differentiating) from the start and end of the hypercube
    #e.g. hypercube 0->1->3 and 0->2
    paths = getallpaths(forwardpaths)

    print("Computing Paths...")
    #get all the segments, e.g. 0,1,3,0->1,0->1->3,1->3,0,2,0->2
    pathsegments = []
    for p in paths:
        for start in range(1,len(p)):
            pathsegments.append([p[start]])
            for end in range(start+1,len(p)):
                pathsegments.append(p[start:end+1])

    #just keep unique ones so we don't test them twice            
    unique_pathsegments = [list(x) for x in set(tuple(x) for x in pathsegments)]

    #print("START,MID,END,INNER CHANGES")
    #print(startchanges, midchanges, endchanges, innerchanges)
    #check all these path segments, get the maximum value for each.
    maxval = -np.inf
    print("Checking Segments...")
    for it,seg in enumerate(unique_pathsegments):
        print("\n%d/%d" % (it,len(unique_pathsegments)),end="")
        if len(seg)==1:
            b = getbound(EQcentres,hypercube_starts[seg[0]],hypercube_ends[seg[0]],diff_dim,ls,innerchanges[seg[0],:],gridspacing=gridspacing,fulldim=False,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)
        else:
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
               
            if np.any(search_hypercube_start>=search_hypercube_end):
                print("start>end")
                b = 0
            else:
                changes = startchanges[seg[0],:] + np.sum(midchanges[seg[1:-1],:],0) + endchanges[seg[-1],:]
                #print(changes)
                #print(EQcentres)
                #print(search_hypercube_start,search_hypercube_end)
                b = getbound(EQcentres,search_hypercube_start,search_hypercube_end,diff_dim,ls,changes,gridspacing=gridspacing,fulldim=False,forceignorenegatives=forceignorenegatives,dimthreshold=dimthreshold)
        #print(seg,b)    
            #print("bound %0.2f" % b)
            #print("change:")
            #print(changes,EQcentres)
        if b>maxval:
            maxval = b
            maxseg = seg
    #print("MAXIMUM CHANGE BOUND %0.3f" % maxval)
    return maxval, hypercube_starts, hypercube_ends, maxseg, EQweights
    
def testing():
    "Testing various methods. Incomplete"
    
    import numpy as np

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
    
    EQcentres = np.array([[-1,0],[-1,1]])
    EQweights = np.array([1.0,1.0])
    hypercube_start = np.array([0,0])
    hypercube_end = np.array([4,4])
    d = 0
    ls = 2.0
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls)


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
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls)

    #if we are starting in this cube, then, (for a centre at x=1) although there's a +ve gradient from 0 to 1,
    #there's more negative on the far side, so it's "best" to start on the boundary
    #for a centre at x=2, the positive and negative cancel.
    #for a centre at x=3, there's an overall increase in the function

    assert startchange[0] == 0
    assert startchange[1] == 0
    assert startchange[2] == zeromean_gaussian_1d(1,2)-zeromean_gaussian_1d(3,2)

    #for a centre at x=3, with negative weight, the largest increase is for x=3 to x=4. (as it goes from negative to less-negative)
    assert startchange[3]==1-zeromean_gaussian_1d(1,2)

    #midchange: 0th one should be negative, as its left boundary is higher than its right (as the peak is more
    #to the left). 1st is zero (as the values at the two boundaries are equal). 2nd is positive. 3rd is like the
    #0th but inverted & flipped.
    assert midchange[0] == zeromean_gaussian_1d(3,2)-zeromean_gaussian_1d(1,2)
    assert midchange[1] == 0
    assert midchange[2] == -(zeromean_gaussian_1d(3,2)-zeromean_gaussian_1d(1,2))
    assert midchange[3] == (zeromean_gaussian_1d(3,2)-zeromean_gaussian_1d(1,2))

    #endchange
    #0th: will be the change from zeromean_gaussian_1d(0,2)-zeromean_gaussian_1d(1,2)
    assert endchange[0] == 1-zeromean_gaussian_1d(1,2)
    assert endchange[1] == 1-zeromean_gaussian_1d(2,2)
    assert endchange[2] == 1-zeromean_gaussian_1d(3,2)
    assert endchange[3] == 0

    #innerchange
    assert innerchange[0] == 1-zeromean_gaussian_1d(1,2)
    assert innerchange[1] == 1-zeromean_gaussian_1d(2,2)
    assert innerchange[2] == 1-zeromean_gaussian_1d(3,2)
    assert innerchange[3] == 1-zeromean_gaussian_1d(1,2)


    EQcentres = np.array([[1,0],[2,0],[3,0],[3,2]])
    EQweights = np.array([1.0,1.0,1.0,-1.0])
    hypercube_starts = np.array([[0,0]])
    hypercube_ends = np.array([[4,4]])
    d = 0
    ls = 2.0

    startchanges, midchanges, endchanges, innerchanges,wholecubechanges = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls)

    startchanges, midchanges, endchanges, innerchanges

    assert np.all(startchanges == np.array([0, 0, zeromean_gaussian_1d(1,2)-zeromean_gaussian_1d(3,2), 1-zeromean_gaussian_1d(1,2)]))
    assert np.all(midchanges == np.array([zeromean_gaussian_1d(3,2)-zeromean_gaussian_1d(1,2),0, -(zeromean_gaussian_1d(3,2)-zeromean_gaussian_1d(1,2)),(zeromean_gaussian_1d(3,2)-zeromean_gaussian_1d(1,2))]))
    assert np.all(endchange==np.array([1-zeromean_gaussian_1d(1,2),1-zeromean_gaussian_1d(2,2),1-zeromean_gaussian_1d(3,2),0]))
    assert np.all(innerchange==np.array([1-zeromean_gaussian_1d(1,2),1-zeromean_gaussian_1d(2,2),
                                         1-zeromean_gaussian_1d(3,2),1-zeromean_gaussian_1d(1,2)]))
    
    
    
