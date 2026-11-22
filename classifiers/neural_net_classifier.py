#FILE NAME: neural_net_classifier.py

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, Y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - Y: Vector of training labels. Y[i] is the label for X[i], and each Y[i] is
          an integer in the range 0 <= Y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If Y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If Y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None    # array of shape (N, C)

        #############################################################################
        # Perform the forward pass, computing the class scores for the input.       #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        hidden_layer = X.dot(W1) + b1
        hidden_layer = np.maximum(0, hidden_layer)
        scores = hidden_layer.dot(W2) + b2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if Y is None:
            return scores

        #Compute the loss
        loss = None

        #############################################################################
        # Finish the forward pass, and compute the loss. This should include        #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        unorm_log_probs = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(unorm_log_probs)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_class_probs = probs[np.arange(N), Y]
        data_loss = -np.sum(np.log(correct_class_probs)) / N
        reg_loss = reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}

        #############################################################################
        #       Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        dscores = probs.copy()
        dscores[np.arange(N), Y] -= 1
        dscores /= N
        grads['W2'] = hidden_layer.T.dot(dscores) + 2 * reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        dhidden = dscores.dot(W2.T)
        dhidden[hidden_layer <= 0] = 0
        grads['W1'] = X.T.dot(dhidden) + 2 * reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, Y, X_val, Y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - Y: A numpy array f shape (N,) giving training labels; Y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - Y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)  # 49000/200 = 245

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            Y_batch = None

            #########################################################################
            #       Create a random minibatch of training data and labels, storing  #
            # them in X_batch and Y_batch respectively.                             #
            #########################################################################
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            Y_batch = Y[indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, Y=Y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            #       Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for param in self.params:
                self.params[param] -= learning_rate * grads[param]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d:   loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:    #end of one epoch
                # Check accuracy
                train_acc = (self.predict(X_batch) == Y_batch).mean()
                val_acc = (self.predict(X_val) == Y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {                                  # return a dictionary
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict probs for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - Y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, Y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        Y_pred = None

        ###########################################################################
        #       Implement this function; it should be VERY simple!                #
        ###########################################################################
        scores = self.loss(X)
        Y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return Y_pred


