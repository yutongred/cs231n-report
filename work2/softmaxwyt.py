from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_score = scores[y[i]]
        smax=np.max(scores)
        scores -= smax#为了预防取指数之后爆掉的问题，先减去最大值
        loss += -correct_score + smax + np.log(np.sum(np.exp(scores)))##对原来L函数改变后的推导+ smax
        for j in range(num_classes):
            dW[:, j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i, :]
        dW[:, y[i]] -= X[i, :]

    loss /= num_train
    loss += reg * np.sum(W*W)

    dW /= num_train
    dW += 2 * reg * W
#     num_train=X.shape[0]
#     num_class=W.shape[1]
#     for i in range(num_train):
#         scores=X[i].dot(W)
#         correct_score=scores[y[i]]
#         smax=np.max(scores)#为了预防取指数之后爆掉的问题，先减去最大值
#         scores-=smax
#         loss+=-correct_score+np.log(np.sum(np.exp(scores)))+smax##对原来L函数改变后的推导
#         for j in range(num_class):
#             dW[:,j]+=X[i]*np.exp(scores[j])/np.sum(np.exp(scores))
#             dW[:,y[i]]-=X[i]  
#     loss/=num_train
#     loss+=0.5*reg*np.sum(W*W)
#     dW/=num_train
#     dW+=W*reg
    
    

   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    num_class=W.shape[1]
    scores=X.dot(W)
    correct_scores=scores[range(num_train),y]
    smax=np.max(scores,axis=1,keepdims=True)
    scores-=smax
    loss+=-np.sum(correct_scores)+np.sum(smax)+np.sum(np.log(np.sum(np.exp(scores),axis=1)))
    dsoft=np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
    dsoft[range(num_train),y]-=1
    dW=X.T.dot(dsoft)
    loss/=num_train
    loss+=0.5*reg*np.sum(W*W)
    dW/=num_train
    dW+=W*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
