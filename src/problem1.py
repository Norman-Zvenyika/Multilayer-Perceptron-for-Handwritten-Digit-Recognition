# -------------------------------------------------------------------------
'''
    Problem 1: Implement activation and loss functions.

    Note: A static method is also a method which is bound to the class and not the object of the class.
        @staticmethod is used before the function definition to indicate that the function is static.
'''

import numpy as np
from scipy.special import logsumexp
from scipy.special import xlogy, xlog1py

# --------------------------
class Activation():
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def activate(Z):
        raise NotImplementedError

    @staticmethod
    def gradient(Z):
        raise NotImplementedError

# --------------------------
class Sigmoid(Activation):
    """
    Implement the sigmoid activation function and its gradient
    """

    @staticmethod
    def activate(Z):
        """
        Sigmoid of each element of Z.
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise sigmoid of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return (1/(1 + np.exp(-Z)))
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of sigmoid at Z
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise gradient of sigmoid at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return np.multiply(Sigmoid.activate(Z), (1 - Sigmoid.activate(Z)))
        #########################################

# --------------------------
class Softmax(Activation):
    """
    Implement the softmax activation function, for multi-class classification.
    """

    @staticmethod
    def activate(Z):
        """
        Transform each column of Z into a probability distribution through the Softmax mapping.
        Read this article

            https://blog.feedly.com/tricks-of-the-trade-logsumexp/

        to avoid numerical problem in calculating the denominator

        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: each column of Z get transformed to a probability distribution.
        """
        #########################################
        ## INSERT YOUR CODE HERE
        # Normalize each column of the input to avoid numerical issues with exponentials of large numbers.
        # This is done by subtracting the maximum value in each column from every element in the same column.
        Z = Z - np.max(Z, axis=0)

        # Apply the exponential function to all elements in the normalized input.
        # Since the largest value in each column is now 0, the largest result here will be exp(0) = 1, 
        # eliminating the possibility of overflow (numbers too large to represent).
        expZ = np.exp(Z)

        # The softmax function is implemented by dividing each column by its sum.
        # This creates a probability distribution across classes for each input example.
        return expZ / np.sum(expZ, axis=0)
        #########################################

    @staticmethod
    def gradient(Z):
        """
        No need to implement gradient of softmax wrt Z. The reason is that,
        cross_entropy_loss ( softmax(w^T x + b), Y) can be differentiated wrt w^T x + b = Z directly.
        This is implemented in the gradient of cross entropy.
        """
        raise NotImplementedError

# --------------------------
class Identity(Activation):
    """
        Implement the identify activation function, for regression problems.
        """

    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: just return the argument and you don't need to do anything here.
        """
        #########################################
        ## DISREGARD THIS FUNCTION
        #########################################
        return Z

    @staticmethod
    def gradient(Z):
        """
        No need to implement the gradient and the reason is similar to the case of softmax activation.
        The gradient is found when when calculating the gradient of some loss function (e.g., MSE).
        """
        raise NotImplementedError


# --------------------------
class Tanh(Activation):
    """
    Implement the tanh activation function and its gradient
    """

    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise tanh of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return np.tanh(Z)
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of tanh at Z
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise gradient of tanh at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return 1 - np.multiply(np.tanh(Z), np.tanh(Z))
        #########################################

# --------------------------
class ReLU(Activation):
    """
    Implement the ReLU activation function and its gradient
    """
    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise tanh of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return np.maximum(0, Z)
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of ReLU at Z
        Z: n x m matrix, where each column of Z is the vector for an example of dimension n.
        :return: elementwise gradient of tanh at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        # Generate ReLU gradient: 1 for Z > 0, 0 otherwise.
        return np.where(Z > 0, 1, 0)
        #########################################

# --------------------------
class Loss():
    @staticmethod
    def loss(Y, Y_hat):
        raise NotImplementedError

    @staticmethod
    def gradient(Y, Y_hat):
        raise NotImplementedError

# --------------------------
class CrossEntropyLoss(Loss):

    """
    Define the cross entropy loss and its gradient with respect to the input linear term (see below)
    Cross entropy loss is defined here:

        https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy

    """
    @staticmethod
    def loss(Y, Y_hat):
        """
        Refer to
        https://stackoverflow.com/questions/50018625/how-to-handle-log0-when-using-cross-entropy
        to see how to avoid taking log of 0.
        Think: when and why log 0 happens?
        Y: k x m the ground truth labels of m training examples. Y[j, i]=1 if the i-th example has label j. k is the number of classes.
        Y_hat: k x m the multi-class or multi-label predictions on the m training examples. This is the output of the softmax function.
        :return: a scalar that is the averaged cross entropy loss
        """
        #########################################
        ## INSERT YOUR CODE HERE

        # Compute the cross entropy for each example by using the xlogy function, which avoids issues with log(0).
        cross_entropy = np.sum(xlogy(Y, Y_hat), axis=0)

        # Return the average cross entropy loss over all examples.
        return -np.mean(cross_entropy)
        #########################################

    @staticmethod
    def gradient(Y, Y_hat):
        """
        It is the gradient of cross entropy loss with respect to Z that is used to compute Y_hat, NOT wrt Y_hat.
        Y: k x m the ground truth labels of m training examples. Y[j, i]=1 if the i-th example has label j. k is the number of classes.
        Y_hat: k x m the multi-class or multi-label predictions on the m training examples
        :return: k x m vector, the gradients on the m training examples
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return Y_hat - Y
        #########################################

# --------------------------
class MSELoss(Loss):

    """
    Define the Mean Square Error loss and its gradient with respect to the input linear term (see below)
    """
    @staticmethod
    def loss(Y, Y_hat):
        """
        Y: k x m the ground truth k-values of m training examples.
        Y_hat: k x m the regression predictions on the m training examples
        :return: a scalar that is the averaged MSE loss
        """
        #########################################
        ## DISREGARD THIS FUNCTION
        #########################################

    @staticmethod
    def gradient(Y, Y_hat):
        """
        It is the gradient of MSE loss with respect to Y_hat.
        Y: k x m the ground truth k-values of m training examples.
        Y_hat: k x m the regression predictions on the m training examples
        :return: k x m vector, the gradients on the m training examples
        """
        #########################################
        ## DISREGARD THIS FUNCTION
        #########################################
