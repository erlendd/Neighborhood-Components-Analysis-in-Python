import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

def applyA(A, vec):
    #return np.matmul(A, vec)
    result = np.zeros(vec.shape)
    for i in range(A.shape[0]):
        result[i] = A[i, i] * vec[i]
    return result

def compute_softmax_norm_i(A, inp, i):
    softmax_norm = 0.
    for k in range(inp.shape[0]):
        if i == k: continue
        exponent = applyA(A, inp[i]) - applyA(A, inp[k])
        exponent = np.dot(exponent, exponent)
        softmax_norm += np.exp(-exponent)
    return softmax_norm
        

def compute_pij(A, inp, i, j):
    if i == j: return 0 # since pij == 0
    exponent = applyA(A, inp[i]) - applyA(A, inp[j])
    exponent = np.dot(exponent, exponent)
    pij = np.exp(-exponent) / compute_softmax_norm_i(A, inp, i)
    return pij

def nca(A, inp, label, lr=0.5):
    inp = transform(A, inp)
    for i in range(inp.shape[0]):
        p = 0.
        for j in range(inp.shape[0]):
	    if label[i] == label[j]:
	        p += compute_pij(A, inp, i, j)
       
        #print 'p=',p
	first_term = np.zeros( (inp.shape[1], inp.shape[1]) )
	second_term = np.zeros( (inp.shape[1], inp.shape[1]) )
	for k in range(inp.shape[0]):
	    if i == k: continue
	    xik = inp[i] - inp[k]
	    pik = compute_pij(A, inp, i, k)
            term = pik * np.outer(xik, xik)
            #print 'term=',term
	    first_term += term
	    if label[k] == label[i]:
	        second_term += term
	first_term *= p
        #print 'i,1st,2nd:',i, first_term, second_term
        A += lr * (first_term - second_term)
    return A

def transform(A, inp):
    out = np.zeros(inp.shape)
    for i in range(len(out)):
        out[i] = applyA(A, inp[i])
    return out 



if __name__ == "__main__":
    
    X = np.array( [ [0], [0.1], [0.9] ] )
    y = np.array( [0, 1, 1] )
    A = np.eye(X.shape[1])
  
    #print compute_pij(A, X, 1, 2)
    nca(A, X, y)
    print
    print 'X:',X
    print
    print 'A:',A
    print
    #Ascale = []
    #for iter in range(1000):
    #    print nca(A, X, y)
    #    Ascale.append( np.sum(A) )
    #
    #sns.plt.plot(Ascale)
    #sns.plt.show()
    #
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier as kNN
    from sklearn.model_selection import cross_val_score
    
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
    sns.plt.scatter(X[:, 0], X[:, 1], c=y)
    sns.plt.show()
    
    clf = kNN(weights='distance')
    scores = cross_val_score(clf, X, y, scoring='neg_log_loss', cv=25)
    print np.mean(scores)
    
    A = np.eye(X.shape[1])
    Xt = transform(A, X)
    print Xt.shape
    
    Ascale = []
    for iter in range(20):
        if iter % 5 == 0:
            print 'Iteration',iter,
        nca(A, X, y)
        #print 'A',A
        flattenedA = np.sum(A)
        Ascale.append( np.sum(A) )
        if iter % 5 == 0:
            print flattenedA,
            Xt = transform(A, X)
            scores = cross_val_score(clf, Xt, y, scoring='neg_log_loss', cv=25)
    	print np.mean(scores)
    
    
        #if iter % 20 == 0:
        #    sns.plt.scatter(Xt[:, 0], Xt[:, 1], c=y)
        #	sns.plt.show()
    
    sns.plt.plot(Ascale)
    sns.plt.show()


