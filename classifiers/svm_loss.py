#FILE NAME: svm_loss.py

import numpy as np
from random import shuffle

from pip._internal import req


def svm_loss_naive(W, X, Y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - Y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[Y[i]]
    for j in range(num_classes):
      if j == Y[i]:
        continue   #ignore correct class
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #############################################################################
        # Compute the gradient of margin and add it to corresponding elements in dW #
        #############################################################################
        dW[:,j]+=X[i]
        dW[:,Y[i]]-=X[i]
        #############################################################################
        #              END OF YOUR CODE                                             #
        #############################################################################

  #################################################################################
  # Right now the loss and the gradient is a sum over all training examples, but  #
  # we want it to be an average instead,                                          #
  # So average out grad and loss by dividing by num_train.                        #
  #                                                                               #
  # Then, add regularization loss to the loss and                                 #
  # regularization gradienet to dW                                                #
  #################################################################################
  loss /=num_train
  dW /=num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, Y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute scores  (N x C)
  # grab correct class scores
  # compute margins
  # add reg loss to the loss
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss (e.g. margins).                                                      #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
