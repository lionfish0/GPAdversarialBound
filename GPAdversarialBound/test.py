from boundmixofgaussians import findpeak, compute_sum, compute_grad
import numpy as np
from GPAdversarialBound import getallchanges, getchanges, zeromean_gaussian, getbound, AdversBound
import GPAdversarialBound

def approxassert(a,b,accuracy=1e-3):
    assert np.all(np.abs(a-b)<accuracy), "%0.8f and %0.8f are not approximately equal (to %d decimal places)" % (a,b,-np.log10(accuracy))
    
def test_maxheight(ab,startpos,steps=10):
    #the findtop we set to starting in the [0,0] corner
    startheight = zeromean_gaussian(ab.EQcentres-startpos,ab.ls,ab.v) @ ab.EQweights
    #the maximum value of the sum of the two peaks
    guessmaxheight = 0.0
    for testx in np.linspace(0,2,steps):
        for testy in np.linspace(0,2,steps):
            for testz in np.linspace(0,2,steps):
                testv = zeromean_gaussian(ab.EQcentres-np.array([testx,testy,testz]),ab.ls,ab.v) @ ab.EQweights
                if testv > guessmaxheight:
                    guessmaxheight = testv
                    maxx = testx
                    maxy = testy
                    maxz = testz
    return(maxx,maxy,maxz,guessmaxheight-startheight)

def rand(startcoord,endcoord,n):
    """Return array of n points, that lie between points a and b in the hypercube"""
    endcoord = np.array(endcoord)
    startcoord = np.array(startcoord)
    dims = len(startcoord)
    return startcoord+np.random.rand(n,dims)*(endcoord-startcoord)
#def pltrec(s,e,i,lw=1):
#    sx = s[0]
#    sy = s[1]
#    ex = e[0]
#    ey = e[1]
#    style = [':b','-k','--r']
#    plt.plot([sx,sx,ex,ex,sx],[sy,ey,ey,sy,sy],style[i],lw=lw)
    

def test_cov(X1,X2,ls,v):
    K = np.empty([len(X1),len(X2)])
    for i in range(len(X1)):
        K[i,:] = zeromean_gaussian(X2-X1[i,:],ls=ls,v=v) #using this function for the EQ RBF
    return K

def testing():
    import numpy as np
    boxstart = [0.0,0.0]
    boxend = [2.0,2.0]
    X = np.array([[0.5,0.5],[0.5,1.5]])#,[.8,.8,.8],[1.2,1.2,1.2]])
    Y = np.array([[1,0]]).T
    dims = 2
    nsteps = [3]*dims
    ls = 0.5
    sigma = 0.1
    v = 1
    gridres = 1000
    dimthreshold = 2
    ab = AdversBound()
    
    K = test_cov(X,X,ls,v)
    alpha = (np.linalg.inv(K+np.eye(len(X))*(sigma**2)) @ Y)[:,0]
    ab.configure(X,alpha,ls,sigma,v,boxstart,boxend,nsteps,gridres,dimthreshold)
    sequence_bounds, sequences, _ = ab.compute(1)

    #test environment
    #
    # ________________
    # |    |    |    |
    # |(2) |(5) |(8) |
    # |__Bx|____|____|  B=0 (1st item)
    #1|    |    |    |
    #m|(1) |(4) |(7) |
    #i|____|____|____|
    #d|  Ax|    |    |  A=1 (0th item)
    # |(0) |(3) |(6) |
    # |____|____|____|
    #     dimension 0->

    assert np.all(ab.compute_indexes_of_line([0,0],[2,0],0)==np.array([0,3,6]))
    #
    #so in dimension 0 we expect the largest midchange (negative) to be in the middle column
    #i.e. 3,4,5, for the 0th column
    assert(np.all(np.where(ab.midchanges[0]<-0.5)[0]==np.array([3,4,5])))
    #
    #in the 1st dimension we expect the largest midchange to be the middle row (1,4,7)
    assert(np.all(np.where(ab.midchanges[1]<-0.5)[0]==np.array([1,4,7])))

    #in the 0th dimension the start change is large just in the left hand cells (0,1,2)
    #  and this upper bound corresponds to starting on the very left edge and moving across
    #  the whole cell
    correctresult = np.full([9,2],False); correctresult[0:3,0]=True
    assert np.all((ab.startchanges[0]>0.3)==correctresult)

    #oddly the endchanges nearly match the start changes, as we could start on the left edge
    #and end at the peak of the training points, so they're slightly larger values than
    #for the start changes.
    correctresult = np.full([9,2],False); correctresult[0:3,0]=True
    assert np.all((ab.endchanges[0]>0.3)==correctresult)

    #clearly the first column has the greatest increase too for innerchanges
    correctresult = np.full([9,2],False); correctresult[0:3,0]=True
    assert np.all((ab.innerchanges[0]>0.3)==correctresult)

    sequences, sequence_bounds = ab.findtop([0,0],1,[0])
    #if we consider moving from the [0,0] cell along dimension 0, then we can again look at the changes computed.
    #for the 0th input which is in that cell
    #specifically if we move from [0,0] to [0,0], [1,0] or [2,0]:
    #(ab.innerchanges[0][0]) #is about 0.39
    #(ab.startchanges[0][0]+ab.endchanges[0][6]) #is about 0.36
    #(ab.startchanges[0][0]+ab.midchanges[0][3]+ab.endchanges[0][6]) #is negative.
    #so we expect similar patterns in the three sequence outputs:

    sequences[2][0][0]==[0,0] #don't leave cube
    approxassert(sequences[2][0][1][0],0.39,accuracy=0.01)
    sequences[1][0][0]==[1,0]
    approxassert(sequences[1][0][1][0],0.36,accuracy=0.01)
    sequences[0][0][0]==[2,0]
    approxassert(sequences[0][0][1][0],-0.21,accuracy=0.01)

    #if we consider moving from the [0,2] cell along dimension 0, then we can again look at the changes computed.
    #we should look at 1th input (which is in that cell)
    #specifically if we move from [0,2] to [0,2], [1,2] or [2,2]:

    #(ab.innerchanges[0][0]) #is about 0.0073
    #(ab.startchanges[0][0]+ab.endchanges[0][6]) #is about 0.0394
    #(ab.startchanges[0][0]+ab.midchanges[0][3]+ab.endchanges[0][6]) #is 0.13359
    #but the larger 0th input will have an effect
    #so we expect similar patterns in the three sequence outputs:

    sequences2, sequence_bounds = ab.findtop([0,1],2,[0,1])
    #sequence_bounds

    fulllist = []
    for x in [0,1,2]:
        for y in [0,1,2]:
            fulllist.append((x,y))
    fulllist = set(fulllist)
    for s in sequences2:
        fulllist.remove((s[0][0][0],s[1][0][1]))
    assert len(fulllist)==0
    
    boxstart = [0.0,0.0,0.0]
    boxend = [2.0,2.0,2.0]
    startcell = [9,5,0]
    X = np.array([[0.5,0.5,0.5],[1.5,1.5,1.5]])#,[.8,.8,.8],[1.2,1.2,1.2]])
    Y = np.array([[1,0]]).T
    dims = 3
    nsteps = [10]*dims
    ls = 0.5
    sigma = 0.1
    v = 1
    gridres = 10
    dimthreshold = 2
    ab = AdversBound()
    
    
    K = test_cov(X,X,ls,v)
    alpha = (np.linalg.inv(K+np.eye(len(X))*(sigma**2)) @ Y)[:,0]
    ab.configure(X,alpha,ls,sigma,v,boxstart,boxend,nsteps,gridres,dimthreshold)
    sequences_highres, sequence_bounds = ab.findtop(startcell,3,[0,1,2])



    startpos = np.array(boxstart) + (np.array(boxend)-np.array(boxstart))*np.array([0,0,0])/nsteps
    maxx,maxy,maxz,maximpchange = test_maxheight(ab,startpos)
    assert sequence_bounds[-1]>maximpchange
    print(startpos,maxx,maxy,maxz,maximpchange,sequence_bounds[-1],sequences_highres[-1])
    
    boxstart = [0.0,0.0,0.0]
    boxend = [2.0,2.0,2.0]
    startcell = [3,4,2]
    X = np.array([[0.5,0.5,0.5],[1.5,1.5,1.5]])#,[.8,.8,.8],[1.2,1.2,1.2]])
    Y = np.array([[1,0]]).T
    dims = 3
    nsteps = [5]*dims
    ls = 0.5
    sigma = 0.1
    v = 1
    gridres = 10
    dimthreshold = 2
    ab = AdversBound()
    K = test_cov(X,X,ls,v)
    alpha = (np.linalg.inv(K+np.eye(len(X))*(sigma**2)) @ Y)[:,0]
    ab.configure(X,alpha,ls,sigma,v,boxstart,boxend,nsteps,gridres,dimthreshold)
    
    sequences_highres, sequence_bounds = ab.findtop(startcell,3,[0,1,2])


    startpos = np.array(boxstart) + (1.0*np.array(boxend)-np.array(boxstart))*np.array(startcell)/nsteps
    maxx,maxy,maxz,maximpchange = test_maxheight(ab,startpos)

    print(startpos,maxx,maxy,maxz,maximpchange,sequence_bounds[-1],sequences_highres[-1])
    assert sequence_bounds[-1]>maximpchange    
    
    for seq, seqb in zip(sequences_highres, sequence_bounds):
        bound = 0
        for s in seq:
            bound+=s[1][0]
        #print(seq,seqb,bound,seqb)
        approxassert(bound,seqb,1e-5)
        
    boxstart = [0.0,0.0]
    boxend = [2.0,2.0]
    startcell = [9,5,0]
    X = np.array([[0.5,0.5],[1.5,1.5]])
    Y = np.array([[1,0]]).T
    dims = 2
    nsteps = [3]*dims
    ls = 0.5
    sigma = 0.1
    v = 1
    gridres = 10
    dimthreshold = 2
    ab = AdversBound()
    K = test_cov(X,X,ls,v)
    alpha = (np.linalg.inv(K+np.eye(len(X))*(sigma**2)) @ Y)[:,0]
    ab.configure(X,alpha,ls,sigma,v,boxstart,boxend,nsteps,gridres,dimthreshold)
        
    sequence_bounds, sequences, hires_bounds = ab.compute(2,hires=2)

    

    testpoints = []
    testpoints_found = []
    for seq in sequences[0]:
        cell = ab.get_index_of_cell(seq[0])
        tps = rand(ab.hypercube_starts[cell], ab.hypercube_ends[cell],5)
        testpoints.append(tps)
        testpoints_found.append(np.full(len(tps),False))
    testpoints_found

    abhires = ab.hires_debug[0]['abhires']
    #looks for the test points in the appropriate locations

    backward_testpoints = []
    backward_testpoints_found = []
    for i in range(len(testpoints)):
        backward_testpoints.append([])
        backward_testpoints_found.append([])
    for seqs in ab.hires_debug[0]['hires_sequences']: #['abhires'].hypercube_starts

        for i,seq in enumerate(seqs):
            cell = abhires.get_index_of_cell(seq[0])
            s = abhires.hypercube_starts[cell]
            e = abhires.hypercube_ends[cell]
            #pltrec(s,e,i)
            testpoints_found[i][np.all((s<testpoints[i]) & (testpoints[i]<e),1)] = True

            tps = rand(s,e,5)
            backward_testpoints[i].extend(tps)
            backward_testpoints_found[i].extend(np.full(len(tps),False))
    for i in range(len(testpoints)):
        backward_testpoints_found[i] = np.array(backward_testpoints_found[i])


    for i,seq in enumerate(sequences[0]):
        cell = ab.get_index_of_cell(seq[0])
        s = ab.hypercube_starts[cell]
        e = ab.hypercube_ends[cell]
        #pltrec(s,e,i,lw=2)
        backward_testpoints_found[i][np.all((s<backward_testpoints[i]) & (backward_testpoints[i]<e),1)] = True
    assert np.all(backward_testpoints_found)
    assert np.all(testpoints_found)    
    ############test zeromean_gaussian_1d#############
    from boundmixofgaussians import zeromean_gaussian_1d
    assert zeromean_gaussian_1d(x=1.5,ls=3.0,v=0.5)==0.5*np.exp(-0.5*1.5**2/3.0**2)
    ############test getchanges############
    EQcentres = np.array([[-1,0],[-1,1]])
    EQweights = np.array([1.0,1.0])
    hypercube_start = np.array([0,0])
    hypercube_end = np.array([4,4])
    d = 0
    ls = 2.0
    v = 2.0
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v)


    #startchange: if this was a starting cube, starting at 0, with a peak at -1, it can't increase over the cube so we use zero
    assert np.all(startchange == 0)
    from boundmixofgaussians import zeromean_gaussian_1d, zeromean_gaussian, findbound, PCA
    #midchange: if it's a middle cube, we integrate over the whole of the cube
    assert np.all(midchange == zeromean_gaussian_1d(5,2,2)-zeromean_gaussian_1d(1,2,2))
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
    v = 0.5
    startchange, midchange, endchange, innerchange = getchanges(EQcentres, EQweights, hypercube_start, hypercube_end, d, ls, v)

    #if we are starting in this cube, then, (for a centre at x=1) although there's a +ve gradient from 0 to 1,
    #there's more negative on the far side, so it's "best" to start on the boundary
    #for a centre at x=2, the positive and negative cancel.
    #for a centre at x=3, there's an overall increase in the function

    assert startchange[0] == 0
    assert startchange[1] == 0
    assert startchange[2] == zeromean_gaussian_1d(1,2,0.5)-zeromean_gaussian_1d(3,2,0.5)

    #for a centre at x=3, with negative weight, the largest increase is for x=3 to x=4. (as it goes from negative to less-negative)
    assert startchange[3]==v-zeromean_gaussian_1d(1,2,0.5)

    #midchange: 0th one should be negative, as its left boundary is higher than its right (as the peak is more
    #to the left). 1st is zero (as the values at the two boundaries are equal). 2nd is positive. 3rd is like the
    #0th but inverted & flipped.
    assert midchange[0] == zeromean_gaussian_1d(3,2,0.5)-zeromean_gaussian_1d(1,2,0.5)
    assert midchange[1] == 0
    assert midchange[2] == -(zeromean_gaussian_1d(3,2,0.5)-zeromean_gaussian_1d(1,2,0.5))
    assert midchange[3] == (zeromean_gaussian_1d(3,2,0.5)-zeromean_gaussian_1d(1,2,0.5))

    #endchange
    #0th: will be the change from zeromean_gaussian_1d(0,2)-zeromean_gaussian_1d(1,2)
    assert endchange[0] == v-zeromean_gaussian_1d(1,2,0.5)
    assert endchange[1] == v-zeromean_gaussian_1d(2,2,0.5)
    assert endchange[2] == v-zeromean_gaussian_1d(3,2,0.5)
    assert endchange[3] == 0

    #innerchange
    assert innerchange[0] == v-zeromean_gaussian_1d(1,2,0.5)
    assert innerchange[1] == v-zeromean_gaussian_1d(2,2,0.5)
    assert innerchange[2] == v-zeromean_gaussian_1d(3,2,0.5)
    assert innerchange[3] == v-zeromean_gaussian_1d(1,2,0.5)

    ############test getallchanges############
    EQcentres = np.array([[1,0],[2,0],[3,0],[3,2]])
    EQweights = np.array([1.0,1.0,1.0,-1.0])
    hypercube_starts = np.array([[0,0]])
    hypercube_ends = np.array([[4,4]])
    d = 0
    ls = 2.0
    v = 0.2

    startchanges, midchanges, endchanges, innerchanges,wholecubechanges,wholecubecounts = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls,v)

    startchanges, midchanges, endchanges, innerchanges

    assert np.all(startchanges == np.array([0, 0, zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(3,2,v), v-zeromean_gaussian_1d(1,2,v)]))
    assert np.all(midchanges == np.array([zeromean_gaussian_1d(3,2,v)-zeromean_gaussian_1d(1,2,v),0, -(zeromean_gaussian_1d(3,2,v)-zeromean_gaussian_1d(1,2,v)),(zeromean_gaussian_1d(3,2,v)-zeromean_gaussian_1d(1,2,v))]))
    assert np.all(endchanges==np.array([v-zeromean_gaussian_1d(1,2,v),v-zeromean_gaussian_1d(2,2,v),v-zeromean_gaussian_1d(3,2,v),0])), (np.array([v-zeromean_gaussian_1d(1,2,v),v-zeromean_gaussian_1d(2,2,v),v-zeromean_gaussian_1d(3,2,v),0]),endchange)
    assert np.all(innerchanges==np.array([v-zeromean_gaussian_1d(1,2,v),v-zeromean_gaussian_1d(2,2,v),
                                         v-zeromean_gaussian_1d(3,2,v),v-zeromean_gaussian_1d(1,2,v)]))

    ############test getallchanges (out of box), d=1############
    EQcentres = np.array([[0,1],[2,4]])
    EQweights = np.array([1.0,-1.0])
    hypercube_starts = np.array([[0,2]])
    hypercube_ends = np.array([[4,3]])
    d = 1
    ls = 2.0
    v = 0.9

    startchanges, midchanges, endchanges, innerchanges,wholecubechanges,wholecubecounts = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls,v)

    startchanges, midchanges, endchanges, innerchanges

    assert np.all(startchanges == np.array([0,0]))
    assert np.all(midchanges == np.array([zeromean_gaussian_1d(2,2,v)-zeromean_gaussian_1d(1,2,v), zeromean_gaussian_1d(2,2,v)-zeromean_gaussian_1d(1,2,v)]))
    assert np.all(endchanges==np.array([0,0]))
    assert np.all(innerchanges==np.array([0,0]))
    
    ############test getallchanges again (out of box), d=1############
    EQcentres = np.array([[0,1],[2,4]])
    EQweights = np.array([-1.0,1.0])
    hypercube_starts = np.array([[0,2]])
    hypercube_ends = np.array([[4,3]])
    d = 1
    ls = 2.0
    v = 0.3
    
    startchanges, midchanges, endchanges, innerchanges,wholecubechanges,wholecubecounts = getallchanges(EQcentres,EQweights,hypercube_starts,hypercube_ends,d,ls,v)

    startchanges, midchanges, endchanges, innerchanges

    assert np.all(startchanges == np.array([zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v), zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v)]))
    assert np.all(midchanges == np.array([zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v), zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v)]))
    assert np.all(endchanges==np.array([zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v), zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v)]))
    assert np.all(innerchanges==np.array([zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v), zeromean_gaussian_1d(1,2,v)-zeromean_gaussian_1d(2,2,v)]))
   ##################################################################
    print("All tests successful!")
    
    
