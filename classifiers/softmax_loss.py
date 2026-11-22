#FILE NAME: softmax_loss.py

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, Y, reg):
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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    # compute unnormlaized log probs
    # unorm_log_probs = ??
    unorm_log_probs = X[i].dot(W)

    # for numerical stability in softmax loss add the following line to your code
    # unorm_log_probs -= np.max(unorm_log_probs)
    unorm_log_probs -= np.max(unorm_log_probs)

    # get class probabilities
    exp_log_probs = np.exp(unorm_log_probs)
    probs = exp_log_probs / np.sum(exp_log_probs)
    # compute loss and add it to the loss so far
    loss += -np.log(probs[Y[i]])
    # compute gradient and add it to the gradients so far
    for j in range(num_classes):
      dW[:,j] += (probs[j] - (j == Y[i])) * X[i]

  # average out grad and loss
  loss /= num_train
  dW /= num_train
  # add regularization loss to total loss and regularization gradienet to dW
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, Y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute unnormlaized log probs
  # unorm_log_probs = ??
  unorm_log_probs = X.dot(W)
  # unorm_log_probs -= np.max(unorm_log_probs)
  unorm_log_probs -= np.max(unorm_log_probs, axis=1, keepdims=True)
  # compute class probabilities
  exp_scores = np.exp(unorm_log_probs)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # compute data and reg loss
  correct_class_probs = probs[np.arange(num_train), Y]
  loss = -np.sum(np.log(correct_class_probs)) / num_train
  loss += reg * np.sum(W * W)

  # compute gradient dW
  probs[np.arange(num_train), Y] -= 1
  dW = X.T.dot(probs) / num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

