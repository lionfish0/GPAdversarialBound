from sklearn.linear_model import LogisticRegression
import numpy as np

def invlogi(y):
    return -.5*np.log((1/y)-1)
    
def get_logistic_result(Xtrain,Ytrain,Xtest,Ytest):
    """
    Returns a table of C, Score, cumulativelatents[0], cumulativelatents[1], cumulativelatents[2], cumulativelatents[3], cumulativelatents[4], ci, pixelsneeded
    """
    result = []
    for Clog in np.arange(-5,5,0.2):
        C = np.exp(Clog)
        clf = LogisticRegression(random_state=0, solver='lbfgs', C=C,
                                 multi_class='multinomial').fit(Xtrain, Ytrain[:,0])
        score = clf.score(Xtest,Ytest[:,0])
        cumulativelatents = np.cumsum(np.sort(np.abs(clf.coef_))[0,::-1])
        s = np.sort(invlogi(clf.predict_proba(Xtrain)[:,1]))
        ci = s[int(len(s)*0.95)]-s[int(len(s)*0.05)]
        pixelsneeded = np.where(cumulativelatents>ci)[0][0]
        if len(cumulativelatents)>3:
            result.append([C,score*100,cumulativelatents[0],cumulativelatents[1],cumulativelatents[2],cumulativelatents[3],cumulativelatents[4],ci,pixelsneeded])
        else:
            result.append([C,score*100,cumulativelatents[0],cumulativelatents[1],cumulativelatents[2],cumulativelatents[3],ci,pixelsneeded])

    return result
