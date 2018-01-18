import numpy as np
from hypercuboid_integrator import sumovercuboids
import matplotlib.pyplot as plt

def k(X,Xprime,l):
    """Full kernel. Get covariance (using RBF with lengthscale l) between x and xprime"""
    res = np.zeros([X.shape[0],Xprime.shape[0]])
    for i,x in enumerate(X):
        for j,xprime in enumerate(Xprime):
            res[i,j] = np.exp(-.5*np.dot((x-xprime),(x-xprime).T)/(l**2))
    return res

def computeAlpha(X,Y,l):
    """Compute the alpha vector used in the sum over training points:
    sum_i alpha_i k(x_*,x_i)
    
    alpha = K^{-1} y
    so we have to provide the training inputs and outputs and lengthscale.
    
    Parameters:
        X = training inputs
        Y = training outputs
        l = lengthscale
    Returns: array of alpha values.
    
    NOTE: For now we assume \sigma^2 = 1.
    """
    return np.dot(np.linalg.inv(k(X,X,l)+np.eye(len(X))),Y)
    
def getpred2grad(Xtest,X,Y,dim,l,alpha=None):
    """
    Get the gradient at Xtest, wrt Xtest_dim

    Parameters:
        X = training inputs
        Y = training outputs
        dim = dimension that's being differentiated.
        l = lengthscale
        alpha = alpha values.
    Returns: gradient of the predictions.
    """
    assert len(Xtest.shape)==2
    assert Xtest.shape[1]==X.shape[1]
    assert X.shape[0]==Y.shape[0]
    assert dim<Xtest.shape[1]
    assert dim>=0
    assert l>=0
    rets = []
    for xtest in Xtest:
        pred = 0
        K = k(xtest[None,:],X,l)
        for i,(a,kval) in enumerate(zip(alpha,K.T)):
            pred+=a*kval*((X[i,dim]-xtest[dim])/(l**2))
        rets.append(pred)
    return np.array(rets)

def getpred(Xtest,X,Y,l):
    """Get the prediction at Xtest
    Parameters:
        Xtest = locations to test
        X = training inputs
        Y = training outputs
        l = lengthscale
    Returns:
        array of predictions
    """
    assert len(Xtest.shape)==2
    assert Xtest.shape[1]==X.shape[1]
    assert X.shape[0]==Y.shape[0]
    assert l>=0

    rets = []
    for xtest in Xtest:
        alpha = np.dot(np.linalg.inv(k(X,X,l)+np.eye(len(X))),Y)
        pred = 0
        K = k(xtest[None,:],X,l)  
        for i,(a,kval) in enumerate(zip(alpha,K.T)):
            pred+=a*kval
        rets.append(pred)
    return np.array(rets)

def compute_box_bound(b,X,Y,l,dim):
    """
    Computes an upper bound for the value of the gradient
    inside /one/ box, b
    
    Parameters:
        b: hypercube, specified by a Dx2 array, each row is a pair of
         start and end locations.
        X,Y: training inputs and outputs (NxD and Nx1 arrays, respectively).
        l: lengthscale
        dim: dimension we're computing the gradient for.
    Returns:
        peakgrad:
            upper bound on the gradient in dimension dim, for box b.
     [debugging] peaklocs, peakvals: (basically these are for debugging only)
            these are the locations and values for each of the training points
            (note that these are combined to produce the peakgrad)   
    """
    
    peakgrad = 0  #total over all training points
    peaklocs = [] #vector of gradient peaks
    peakvals = [] #values of the gradients at these peaks
    
    alpha = computeAlpha(X,Y,l)
    #loop over all training points.
    for i in range(X.shape[0]):
        a = alpha[i]
        x = X[i:(i+1),:]
        y = Y[i:(i+1),:]
        testx = x.copy()
        
        #testx contains the location at which the gradient is maximum
        #we know from simple calculus for the RBF with lengthscale 'l' that the maximum
        #gradient is at:
        testx[0,dim] = x[0,dim]-(l)*np.sign(y[0,0])
        
        
        ####Test code to confirm this is a gradient peak####
        testxdelta = testx.copy()
        delta = 0.01
        testxdelta[0,dim]+=delta
        #rather than use gradient - we'll just check neighbouring locations have lower gradient
        #assert np.abs((getpred2grad(testx,x,y,dim,l,alpha=a)-getpred2grad(testxdelta,x,y,dim,l,alpha=a))/delta)<0.01, \
        #    "Error %0.6f should be zero (dim=%d)" % ((getpred2grad(testx,x,y,dim,l,alpha=a)-getpred2grad(testxdelta,x,y,dim,l,alpha=a))/delta,dim)
        assert getpred2grad(testx,x,y,dim,l,alpha=a)>=getpred2grad(testxdelta,x,y,dim,l,alpha=a)-1e-10
        testxdelta = testx.copy()
        delta = 0.001
        testxdelta[0,dim]-=delta
        assert getpred2grad(testx,x,y,dim,l,alpha=a)>=getpred2grad(testxdelta,x,y,dim,l,alpha=a)-1e-10 #for reasons of numerical stability this sometimes needs a bit of help
        ####################################################
        
        #this is to store the location of the peak inside the boundary
        boundarymaxpeak = np.ones_like(testx)*np.NaN 

        #first we need to look at the derivative dimension, is the actual peak inside the boundary
        #(in this dimension)
        if (testx[0,dim]>b[dim][0]) & (testx[0,dim]<b[dim][1]):
            #the peak of the derivative-rbf in the differentiated axis is inside the boundary.
            boundarymaxpeak[0,dim] = testx[0,dim] #we can set this as the location of the maximum
            maxgrad = getpred2grad(testx,x,y,dim,l,alpha=a) #get the value of the actual peak
        else:            
            #the peak is outside the boundary, check at which boundary is most positive.
            #todo this could be done without actually computing the values (just look which is closer!!)
            testpoints = np.r_[testx.copy(),testx.copy()]
            testpoints[0,dim]=b[dim][0]
            testpoints[1,dim]=b[dim][1]
            boundaryedges = getpred2grad(testpoints,x,y,dim,l,alpha=a)
            maxgrad = np.max(boundaryedges) #saves the value of the peak, it might be negative!
            boundarymaxpeak[0,dim] = testpoints[np.argmax(boundaryedges),dim]
            
        #at this point 'boundarymaxpeak' contains NaNs except for the dimension we've differentiated
        
        #now we look at the other dimensions...
        #if maxgrad>0 then we're trying to get to the greatest point in each other dimension
        #as the slide taken at the maxgrad location is positive everywhere over the slide
        #if it's negative then we're trying to get the least negative point
        #(somewhere on the boundary)
        for d in range(x.shape[1]):
            if d==dim:
                continue #skip the differentiated dimension
            
            if (testx[0,d]>b[d][0]) & (testx[0,d]<b[d][1]): #if the peak is inside the boundary (in this dimension)
                if maxgrad>0: #if the peak is positive
                    boundarymaxpeak[0,d] = testx[0,d]
                    continue #we're done
                    #if it's not positive, then even though this dimension has the
                    #peak inside its boundaries, the peak is so far away that we actually
                    #have the negative peak inside/near the box. Hence we need to find
                    #the boundaries that are least negative.
            #if maxgrad is negative or we're outside the boundary, we need to check the
            #boundaries
            #todo: combine duplicate code...
            #this checks which boundary is most positive...
            testpoints = np.r_[testx.copy(),testx.copy()]
            testpoints[0,d]=b[d][0]
            testpoints[1,d]=b[d][1]
            testpoints[:,dim] = boundarymaxpeak[0,dim]
            boundaryedges = getpred2grad(testpoints,x,y,dim,l,alpha=a)
            #...and sets the appropriate coordinate of 'boundarymaxpeak' to the boundary with the
            #most positive value
            boundarymaxpeak[0,d] = testpoints[np.argmax(boundaryedges),d]
            
        peakval = getpred2grad(boundarymaxpeak,x,y,dim,l,alpha=a)
        peakgrad += peakval
        peaklocs.append(boundarymaxpeak)
        peakvals.append(peakval)
    return peakgrad, peaklocs, peakvals


def getmaxshift(B,peakgrads,graddim):
    """
    Given the space is sliced into segments described in B, and those segments
    have upper bounded gradients, in peakgrads, what's the largest possible
    change in the prediction, if we move along dimension 'graddim'?
    
    Assumes:
        B covers whole domain without overlap.
        
    Parameters:
        B:
            a LIST of arrays, each array has the start and end locations
            of the cuboid for each dimension, for example a 4d hypercube
            from the origin to location (2,2,2,2) would be:
              np.array([[0,2],[0,2],[0,2],[0,2]]) 
        peakgrads:
            a list of 1x1 arrays, each containing a single number (the
            maximum gradient of the prediction within the hypercuboids
            described in B.
        graddim:
            the gradient we want to integrate over.
    Returns:
        an upper bound on the greatest increase in the prediction over
        dimension 'graddim' over the whole space occupied by the segments
        in B.
    """
    B=np.array(B)
    peakgrads = np.array(peakgrads)[:,:,0]
    seglist = sumovercuboids(B,peakgrads,graddim)
    maxint = 0
    for s in seglist:
        maxint = max(maxint,s['int'])
    return maxint

def randomargmax(x):
    """
    To avoid repetitive (and potentially suboptimal) splitting of the domain
    always in the same order, we add a little randomness to the choice of
    slice dimension. We do this by replacing the np.argmax method (which
    returns the first element with the largest value) with this method
    which returns an element that has the largest value, selecting randomly
    if there are more than one element at the global maximum.
    
    Parameters:
     x = numpy array
    Returns:
     index of largest value.
    """
    return np.random.choice(np.where(x==np.max(x))[0])
    #tempx = (x*1.0)+np.random.rand(x.shape[0],x.shape[1])*0.01 #add tiny bit of randomness to the numbers!
    #return np.where(tempx==np.max(tempx))[0][0]

def getshiftboundsfordim(graddim,X,Y,l,totalits,earlystop,valmin,valmax):
    """
    Finds largest change in the prediction for a change in a given dimension.
    See getshiftbounds for parameters.
    """
    B = [np.repeat(np.array([[valmin,valmax]]),X.shape[1],0)]
    peakgrads =[np.NaN]*len(B)

    for it in range(totalits):
        #loop through all the boxes, and compute any that aren't yet computed (check if 'computed')
        for i in np.random.permutation(range(len(B))):
            if peakgrads[i] is np.NaN:
                peakgrad, peaklocs, peakvals = compute_box_bound(B[i],X,Y,l,graddim)            
                peakgrads[i] = peakgrad
        #print(np.max(np.array([p[0,0] for p in peakgrads if p is not np.NaN])))

        #stop here if last iteration so we don't split the box again
        steepest = np.max(np.array([p[0,0] for p in peakgrads if p is not np.NaN])) #temporary early stopping
        if (it==totalits-1) or (steepest<earlystop):
            #print("stopped at iteration %d"%it)
            break

        #pick box with max gradient bound
        #maxbox = randomargmax(np.array(peakgrads))

        #pick box with max gradient*width bound
        widths = np.array([np.diff(b[graddim]) for b in B])
        maxbox = randomargmax(np.array(peakgrads)[:,:,0]*widths) #np.argmax(np.array(peakgrads)[:,:,0]*widths)

        b = B[maxbox].copy()
        newb = []
        B.pop(maxbox)
        peakgrads.pop(maxbox)

        #decide on which way we're going to halve the box (we want the longest dimension)
        #dimtosplit = np.argmax(np.diff(b))
        dimtosplit = randomargmax(np.diff(b))
        mid = (b[dimtosplit][0]+b[dimtosplit][1])/2 #the midpoint of this dimension

        #find the start and end coordinates of the two new boxes
        #(they're the same as the old box, except for the dimtosplit which has
        #either the start or end coordinates at the mid point).

        newbstarts = b[:,0].copy()
        newbends = b[:,1].copy()
        newbends[dimtosplit] = mid
        newb = np.c_[newbstarts,newbends]
        B.append(newb) #add the 1st new box 
        newbstarts = b[:,0].copy()
        newbends = b[:,1].copy()
        newbstarts[dimtosplit] = mid
        newb = np.c_[newbstarts,newbends]
        B.append(newb) #add the 2nd new box
        #add the two entries to the peakgrads list with NaN to mark it for computation.
        peakgrads.extend([np.NaN,np.NaN])
    return {'B':B,'peakgrads':peakgrads}, getmaxshift(B,peakgrads,graddim)        

from dask import compute, delayed
from dask.distributed import Client
def getshiftbounds(X,Y,l=2000.0,totalits=100,earlystop=0.0,valmin=0.0,valmax=1.0,ip=None):
    """
    This iterates over all the inputs in X. For each it slices up the space
    in a 'recursive' binary way, picking on those regions with the greatest
    POSITIVE integral over their volume. The result is an upper bound for
    the change in the prediction as one moves along each of the inputs
    (from 0 to 1).
    
    This is the only function you are likely to use from this module.
    
    Parameters:
    X = training inputs
    Y = training outputs
    l = lengthscale (default 2000)
    totalits = total number of times the space will be divided (per input)
        (default 100 times).
    earlystop = if the improvement is less than this then it stops early
        (default 0)
    valmin, valmax = minimum and maximum of the input values (e.g. 0 and 255)
        (default 0 and 1)
        
    Returns:
        allshifts = bound on the largest *increase* in the prediction value
          for each dimension.
        debuginfo = contains the final segmentation of the space with the
          gradients of each segment.
    
    Important note:
        This method only returns a bound on the positive change in the
        prediction. If over the space the predictions are:
          5 4 3 4 5 5 4 3 2 1 1
        then it might return as an upper bound 3, which incorporates the
        increase from 3 to 5. However the prediction can *change* more by
        moving from 5 to 1.
        
        The best way to handle this is to negate all the Y values and run
        the algorithm again. This will then find the maximum change in the
        prediction moving the other way, so in the example above the values
        would become:
         -5 -4 -3 -4 -5 -5 -4 -3 -2 -1 -1
        and the greatest increase in prediction might be bound by, for
        example 5, which would incorporate the change from -5 to -1.
    """

    allshifts = []
    debuginfo = []
    
    if ip is None:
        for graddim in range(X.shape[1]):
            print(".",end="")
            debug, shift = getshiftboundsfordim(graddim,X,Y,l,totalits,earlystop,valmin,valmax)
            debuginfo.append(debug)
            allshifts.append(shift)
        print("")
        
    else:
        tocompute = [np.NaN]*X.shape[1]
        for graddim in range(X.shape[1]):
            tocompute[graddim] = delayed(getshiftboundsfordim,pure=True)(graddim,X,Y,l,totalits,earlystop,valmin,valmax)
        #debuginfo.append(debug)
        #allshifts.append(shift)
        print("Computing...")
        client = Client(ip+':8786')
        results = compute(*tocompute, get=client.get)
        
        allshifts = []
        debuginfo = []
        for res in results:
            allshifts.append(res[1])
            debuginfo.append(res[0])
    return allshifts,debuginfo
    
def plot2dB(B):
    """
    Helper function to plot a *2d* set of boxes.
    E.g. Pass the B in the debuginfo from the getshiftbounds function.
    """
    d = 0.001
    for b in B:
        plt.plot([b[0,0]+d,b[0,0]+d,b[0,1]-d,b[0,1]-d,b[0,0]+d],[b[1,0]+d,b[1,1]-d,b[1,1]-d,b[1,0]+d,b[1,0]+d],'k-')
    #plt.axis('equal')
    plt.ylim([0,1])
    plt.xlim([0,1])
