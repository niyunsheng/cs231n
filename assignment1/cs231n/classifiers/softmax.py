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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores = np.exp(scores)
        scores_sum = np.sum(scores)
        p = scores[y[i]] / scores_sum
        loss += -np.log(p)
        
        for j in range(num_classes):
            dW[:,j] += X[i,:].T * scores[j] / scores_sum
            if j == y[i]:
                dW[:,j] += -X[i,:].T
        
        
    loss /= num_train # 不要忘记正则化
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W) # 加入正则化损失
    dW += reg * W
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

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    scores = X.dot(W)
    max_class_score = np.max(scores,axis=1)#每一行的最大值
    max_class_score = np.reshape(max_class_score,(num_train,-1))# 为什么要必须加上reshape呢？不加的话运算结果会不同，跟numpy的广播机制有关
    
    scores = scores - max_class_score
    scores = np.exp(scores)
    
    scores_sum = np.sum(scores,axis=1)#每一行的和
    scores_sum = np.reshape(scores_sum,(num_train,-1))
    
    correct_class_score = scores[np.arange(num_train),y] 
    correct_class_score = np.reshape(correct_class_score,(num_train,-1)) 
    loss += np.sum(-np.log(np.true_divide(correct_class_score,scores_sum)))
    
    scores = np.true_divide(scores,scores_sum)
    scores[range(num_train),y] -= 1
    dW += np.dot(X.T,scores)
      
        
    loss /= num_train# 不要忘记正则化
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W) # 加入正则化损失
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
