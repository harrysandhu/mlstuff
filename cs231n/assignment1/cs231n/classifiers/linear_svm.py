from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    loss = 0.0
    # compute the loss and the gradient
    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
        scores = X[i].dot(W)
        syi = scores[y[i]]
        scores[y[i]] = 0
        for j in range(C):
            if j == y[i]:
                continue
            margin = np.maximum(0, scores[j] - syi + 1)
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    # mean 
    loss /= N
    dW /= N
    # regularization
    loss += reg * np.sum(W * W)
    dW += reg*W

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    N = X.shape[0]
    C = W.shape[1]
    scores = X.dot(W)
    y_scores = scores[np.arange(N), y.T].reshape(1,-1)
    margins = np.maximum(0, scores - y_scores.T + 1)
    margins[np.arange(N), y.T] = 0
    loss = np.mean(np.sum(margins, axis=1))
    
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(N), y.T] = -row_sum.T
    dW = X.T.dot(binary)
    dW /= N
    # regularization
    loss += reg * np.sum(W*W)
    dW += reg*W

    return loss, dW



