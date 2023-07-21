# -------------------------------------------------------------------------
'''
    Problem 2: Implement a simple feedforward neural network.
'''

from problem1 import *
import numpy as np
from sklearn.metrics import accuracy_score

class NN:
    #--------------------------
    def __init__(self, dimensions, activation_funcs, loss_func, rand_seed = None):
        """
        Specify an L-layer feedforward network.
        Design consideration: we don't include data in this neural network class.
        Use these passed-in parameters to initialize the hyper-parameters
            (width of each layer, number of layers (depth), activation functions)
        and parameters (W, b) of your NN.

        Also define the variables A and Z to be computed.

        It is recommended to use a dictionary with key = layer index and value = parameters/functions
            for easy referencing to these objects using the 1-based indexing of the layers (excluding the input layer).
            For example, W[l] and b[l] will be referring to the parameters of the l-th layer.
                        A[l] and Z[l] will be the activation and linear terms at the l-th layer.
                        g[l] will be the activation function at the l-th layer.

        Being consistent with the notation in the lecture note will make coding and debugging easier.

        dimensions: list of L+1 integers , with dimensions[l+1] and dimensions[l]
                            being the number of rows and columns for the W at layer l+1.
                            dimensions[0] is the dimension of the input data.
                            dimensions[L] is the dimension of output units.
                            dimension[l] = n[l] is the width of layer l in our lecture note.
        activation_funcs: dictionary with key=layer number, value = an activation class (e.g., ReLU)
        loss_func: loss function at the top layer
        rand_seed: set this to a number if you want deterministic experiments.
                    This will be useful for reproducing your bugs for debugging.
        """

        if rand_seed is not None:
            np.random.seed(rand_seed)

        self.num_layers = len(dimensions) - 1
        self.loss_func = loss_func

        self.W = {}
        self.b = {}
        self.g = {}
        num_neurons = {}
        for l in range(self.num_layers):
            num_neurons[l + 1] = dimensions[l + 1]
            # Xavier initialization
            # self.W[l + 1] = np.random.rand(dimensions[l + 1], dimensions[l])
            # self.b[l + 1] = np.random.rand(dimensions[l + 1], 1)
            nin, nout = dimensions[l], dimensions[l + 1]
            sd = np.sqrt(2.0 / (nin + nout))
            self.W[l + 1] = np.random.normal(0.0, sd, (nout, nin))
            self.b[l + 1] = np.zeros((dimensions[l + 1], 1))
            self.g[l + 1] = activation_funcs[l + 1]

        self.A = {}
        self.Z = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}

    #--------------------------
    def forward(self, X):
        """
        Forward computation of activations at each layer.

        X: n[0] x m matrix. m examples with n[0] dimension features.
        Returns: n[L] x m matrix. The activations at output layer with n[L] neurons.
        """

        # Assign the input matrix X to the activation values of the 0-th layer.
        self.A[0] = X

        # Loop through each layer from 1 to L (inclusive).
        for i in range(self.num_layers):
            
            # Compute the pre-activation value Z for the i-th layer by applying 
            # the linear transformation (weight matrix times the previous layer's activations plus bias).
            self.Z[i + 1] = np.dot(self.W[i + 1], self.A[i]) + self.b[i + 1]

            # Compute the activation value A for the i-th layer by applying 
            # the corresponding activation function to the pre-activation value Z.
            self.A[i + 1] = self.g[i + 1].activate(self.Z[i + 1])
        
        # Return the activation values of the final layer as an array.
        return np.asarray(self.A[self.num_layers]) 
        #########################################

    #--------------------------
    def backward(self, Y):
        """
        Back propagation to compute the gradients of parameters at all layers.
        Use the A and Z cached in forward.
        Vectorize as much as possible and the only loop is to go through the layers.
        You should use the gradient of the activation and loss functions defined in problem1.py

        :param Y: an k x m matrix. Each column is the one-hot vector of the label of an training example.

        :return: two dictionaries of gradients of W and b respectively.
                dW[i] is the gradient of the loss to W[i]
                db[i] is the gradient of the loss to b[i]
        """
        #########################################
        ## INSERT YOUR CODE HERE
   
        #intialize with the last L
        loss = self.loss_func.gradient(Y, self.A[self.num_layers])
        self.dW[self.num_layers] = (1/Y.shape[1]) * np.dot((np.subtract(self.A[self.num_layers], Y)), self.A[self.num_layers-1].T)
        self.dZ[self.num_layers] = self.loss_func.gradient(Y, self.A[self.num_layers])
        self.db[self.num_layers] = np.asmatrix(np.mean(self.dZ[self.num_layers].A, axis=1, keepdims=True))
        
        #do for the remaining layers going backwards
        for i in range(self.num_layers-1, 0, -1): 
            try:
                self.dZ[i] = np.multiply(self.W[i+1].T * self.dZ[i+1], self.g[i].gradient(self.Z[i]))
            except:
                self.dZ[i] = loss
            self.dW[i] = (1/Y.shape[1]) * self.dZ[i] * self.A[i - 1].T
            self.db[i] = np.asmatrix(np.mean(self.dZ[i].A, axis=1, keepdims=True))
        
        return self.dW, self.db
        #########################################

    #--------------------------
    def update_parameters(self, lr, weight_decay = 0.001):
        """
        Use the gradients computed in backward to update all parameters

        :param lr: learning rate.
        """
        #########################################
        ## INSERT YOUR CODE HERE
        for i in range (self.num_layers, 0, -1):
            self.W[i] = np.asarray(self.W[i] - (lr * (self.dW[i])) - (weight_decay*self.W[i]))
            self.b[i] = np.asarray(self.b[i] - (lr * self.db[i]))
        #########################################
        
    #--------------------------------------------------------------------
    
    def train(self, **kwargs):
        """
        Implement mini-batch stochastic gradient descent.

        :param kwargs:
        :return: the loss at the final step
        """
        X_train = kwargs['Training X']
        Y_train = kwargs['Training Y']
        num_samples = X_train.shape[1]
        iter_num = kwargs['max_iters']
        lr = kwargs['Learning rate']
        weight_decay = kwargs['Weight decay']
        batch_size = kwargs['Mini-batch size']

        record_every = kwargs['record_every']

        losses = []
        grad_norms = []

        # iterations of mini-batch stochastic gradient descent
        rng = np.random.default_rng()
        
        for it in range(iter_num):
            
            for p in range(0,batch_size): 

                #generate random indexes of size 
                random_indexes = rng.choice(num_samples,batch_size,False)
            
                #pick a random batch
                X_mini = X_train[:,random_indexes]
                Y_mini = Y_train[:,random_indexes]
        
                #forward propagation
                y_hat = self.forward(X_mini)
                
                #backward propagation
                self.backward(Y_mini)
                
                #calculate the gradients
                norm_dw, norm_db = 0, 0
                for key in self.dW:
                    norm_dw += np.linalg.norm(self.dW[key])
                for key in self.db:
                    norm_db += np.linalg.norm(self.db[key])
                
                grad_norms.append(norm_dw + norm_db)
            
                #calculate the losses
                l = self.loss_func.loss(Y_mini, y_hat)
                losses.append(l)
                
                #update the parameters
                self.update_parameters(lr, weight_decay)

            if (it + 1) % record_every == 0:
                if 'Test X' in kwargs and 'Test Y' in kwargs:
                    prediction_accuracy = self.test(**kwargs)
                    print(', test error = {}'.format(prediction_accuracy))
        
        return losses[-1] #return loss at the final step

    #--------------------------
    def test(self, **kwargs):
        """
        Test accuracy of the trained model.
        :return: classification accuracy (for classification) or
                    MSE loss (for regression)
        """
        X_test = kwargs['Test X']
        Y_test = kwargs['Test Y']

        loss_func = kwargs['Test loss function name']

        output = self.forward(X_test)

        if loss_func == '0-1 error':
            predicted_labels = np.argmax(output, axis = 0)
            true_labels = np.argmax(Y_test, axis = 0)
            return 1.0 - accuracy_score(np.array(true_labels).flatten(), np.array(predicted_labels).flatten())
        else:
            # return the MSE (=Frobenius norm of the difference between y and y_hat, divided by (2m))
            return np.linalg.norm(output - Y_test) ** 2 / (2 * Y_test.shape[1])

    # --------------------------
    def explain(self, x, y):
        """
        Given MNIST images from the same class,
            output the explanation of the neural network's prediction of all the 10 classes.

        :return: an matrix of size n x 10, where n is the number of features of a MNIST image.
            We will visualize this in a IPython Notebook.
        """
        #########################################
        ## INSERT YOUR CODE HERE
        #########################################
