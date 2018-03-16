import numpy as np
from boundmixofgaussians import zeromean_gaussian_1d, zeromean_gaussian, findbound, PCA

def overlap(hypercube_starts,hypercube_ends,splitcube_index,f,ignoredim=None):
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
    
    innerchange = np.nanmax(np.array([midvals - startvals,endvals-midvals,endvals-startvals]),0)
    
    #middle cube: we're interested in the total change from start to end
    midchange = endvals - startvals #negative values can be left in this
  
    return startchange, midchange, endchange, innerchange

def findchanges(EQcentres,EQweights,hypercube_start,hypercube_end,d,ls):
    EQcentres_not_d = np.delete(EQcentres,d,1)
    hc_start_not_d = np.delete(hypercube_start,d)
    hc_end_not_d = np.delete(hypercube_end,d)
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d,ls)
    #TODO I'm confused myself when the ignorenegatives should be set!
    #startbound = findbound(EQcentres_not_d,startchange,ls,1,0.1,hc_start_not_d,hc_end_not_d,ignorenegatives=True)
    #midbound = findbound(EQcentres_not_d,midchange,ls,1,0.1,hc_start_not_d,hc_end_not_d)
    #endbound = findbound(EQcentres_not_d,endchange,ls,1,0.1,hc_start_not_d,hc_end_not_d,ignorenegatives=True)
    #innerbound = findbound(EQcentres_not_d,innerchange,ls,1,0.1,hc_start_not_d,hc_end_not_d,ignorenegatives=True) #TODO note the d=1 in the params refers to the number of dims I think - need to make it vary depending on size of matrix, etc
    return startchange, midchange, endchange, innerchange #startbound, midbound, endbound,innerbound

def getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls):
    startchanges =[]
    midchanges = []
    endchanges = []
    innerchanges = []
    for hypercube_start, hypercube_end in zip(hypercube_starts,hypercube_ends):
        startchange, midchange, endchange, innerchange = findchanges(EQcentres,EQweights,hypercube_start,hypercube_end,d,ls)
        startchanges.append(startchange)
        midchanges.append(midchange)
        endchanges.append(endchange)
        innerchanges.append(innerchange)
    startchanges = np.array(startchanges)
    midchanges = np.array(midchanges)
    endchanges = np.array(endchanges)
    innerchanges = np.array(innerchanges)
    return startchanges, midchanges, endchanges, innerchanges

#here we go through the combinations of starts and ends in this simple line of hypercubes
def getbound(EQcentres,hypercube_start,hypercube_end,d,ls,change,gridspacing=0.1,fulldim=False):
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
    print("Dimensionality: %d" % EQcentres.shape[1])
    if EQcentres.shape[1]>3 and not fulldim:
        print("Compacting to 3d manifold...")
        lowd = 3
        lowdX,evals,evecs,means = PCA(EQcentres.copy(),lowd)
        ignorenegatives = True
    else:
        print("Keeping current dimensionality")
        lowdX = EQcentres
        lowd = EQcentres.shape[1]
        ignorenegatives = False
    print("finding bound...")
    return findbound(lowdX,change,ls=ls,d=lowd,gridspacing=gridspacing,gridstart=np.min(lowdX,0)-gridspacing,gridend=np.max(lowdX,0)+gridspacing,ignorenegatives=ignorenegatives)
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

def compute_full_bound(EQcentres,EQweights,ls,diff_dim,dims,cubesize,splitcount=5,gridspacing=0.2):
    hypercube_starts = [np.zeros(dims)*1.0,np.zeros(dims)*1.0]
    hypercube_ends = [np.full(dims,cubesize)*1.0,np.full(dims,cubesize)*1.0]
    hypercube_ends[0][diff_dim] = 0
    forwardpaths = [[1],[]]
    backwardpaths = [[],[0]]


    innerchanges = None
    #split up the hypercubes a bit (need to make this more intelligent)
    for splitit in range(splitcount):
        split_index = 1
        if innerchanges is not None:
            split_index = np.argmax(np.sum(np.abs(innerchanges),1))
        split_dim = np.argmax(hypercube_ends[split_index]-hypercube_starts[split_index])
        splitcubes(hypercube_starts,hypercube_ends,forwardpaths,backwardpaths,split_index,split_dim,diff_dim)

        startchanges, midchanges, endchanges, innerchanges = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,diff_dim,ls)

    #get all the straightline paths (in the direction we're differentiating) from the start and end of the hypercube
    #e.g. hypercube 0->1->3 and 0->2
    paths = getallpaths(forwardpaths)

    #get all the segments, e.g. 0,1,3,0->1,0->1->3,1->3,0,2,0->2
    pathsegments = []
    for p in paths:
        for start in range(1,len(p)):
            pathsegments.append([p[start]])
            for end in range(start+1,len(p)):
                pathsegments.append(p[start:end+1])

    #just keep unique ones so we don't test them twice            
    unique_pathsegments = [list(x) for x in set(tuple(x) for x in pathsegments)]

    #check all these path segments, get the maximum value for each.
    maxval = -np.inf
    for seg in unique_pathsegments:
        print(seg)
        if len(seg)==1:
            b = getbound(EQcentres,hypercube_starts[seg[0]],hypercube_ends[seg[0]],diff_dim,ls,innerchanges[seg[0],:],gridspacing=gridspacing,fulldim=False)
        else:
            changes = startchanges[seg[0],:] + np.sum(midchanges[seg[1:-1],:],0) + endchanges[seg[-1],:]
            b = getbound(EQcentres,hypercube_starts[seg[0]],hypercube_ends[seg[-1]],diff_dim,ls,changes,gridspacing=gridspacing,fulldim=False)
        print("bound %0.2f" % b)
        if b>maxval:
            maxval = b
            maxseg = seg
    #print("MAXIMUM CHANGE BOUND %0.3f" % maxval)
    return maxval, hypercube_starts, hypercube_ends, maxseg