# Part 1: Defining the neural network.
import numpy as np
import random

class NeuralNetwork():
       
    def __init__(self, inputs, hidden, outputs):
        """
        Initialize a simple neural network with a single hidden layer.
        This method randomly initializes the parameters of the model,
        saving them as private variables.
        
        Each layer is parameterized by a weight matrix and a bias vector; 
        a useful trick is store the weights and biases for a layer together,
        as a single matrix.
        
        Args:
            inputs: int, input dimension
            hidden: int, number of hidden neurons
            outputs: int, number of output neurons
        Returns:
            None
        """
        if(inputs > 0):
            self.inputs = inputs
        else:
            print('Inputs must be greater than zero')
            
        if(hidden > 0):
            self.hidden = hidden
        else:
            print('Number of hidden neurons must be greater than zero')
            
        if(outputs > 0):
            self.outputs = outputs
        else:
            print('Number of output neurons must be greater than zero')
                    
        # Initialize the weights and biases of the neural network as
        # private variables. Store a weight matrix for each layer. 
        
        np.random.seed(0)
                
        # initialize weights and bias for hidden layer
        self.w1 = np.random.randn(inputs, hidden)
        self.b1 = np.random.randn(1, hidden)
        self.wb1 = np.vstack((self.w1, self.b1)) # add row for bias
               
        # initialize weights and bias for output layer
        self.w2 = np.random.randn(hidden, outputs)
        self.b2 = np.random.randn(1, outputs)
        self.wb2 = np.vstack((self.w2, self.b2)) # add row for bias
            
    """ ReLU activation function """
    def relu_activation(self, n_inputs):
        return np.maximum(n_inputs, 0)
    
    """ ReLU derivative function """
    def relu_derivative(self, x):
        if x > 0:
            return 1
        return 0
            
    """ Softmax activation function """
    def softmax_activation(self, n_inputs):
        # softmax is e^input / sum( e^input for all inputs )
        # add axis=1 to perform operation within each row in the matrix
        # subtract the max of the row to prevent overflow error
        try:
            e_values = np.exp(n_inputs - np.max(n_inputs, axis=1).reshape(-1, 1)) # shape (N, D) 
        except:
            print('softmax error')
        return e_values / np.sum(e_values, axis=1).reshape(-1, 1) #change np sum to column to broadcast element division
 
    def predict(self, X):
        """
        Make predictions on inputs X.
        Args:
            X: NxM numpy array where n-th row is an input.
        Returns: 
            y_pred: NxD array where n-th row is vector of probabilities.
        """
        # add column of ones to X so that we can multiply this by the last row of the wb1 matrix (which is the bias)
        # and get the same bias. Then multiply inputs by weights
        # X shape is (N, M+1) 
        X = np.hstack( (X, np.ones( (X.shape[0], 1) ) ) )
            
        # multiply inputs by weights plus a bias
        # X shape is (N, 784+1), self.wb1 is (784+1, # hidden neurons)
        # hidden_preact shape = (N, # hidden neurons)
        hidden_preact = np.dot(X, self.wb1)
        
        # save for back propagation
        self.h_preact = hidden_preact

        # pass through ReLU activation function for each element layout_output
        # relu_output shape = (N, # hidden neurons)
        relu_output = self.relu_activation(hidden_preact)
        
        # take relu_output and multiply by another layer to get to the output vector of 10
        # relu_output shape is (N, # hidden neurons + 1), self.wb2 is shape (# hidden neurons + 1, 10)
        # softmax_preact shape is (N,10)
        
        # add column of ones for bias for relu_output
        relu_output = np.hstack( (relu_output, np.ones( (relu_output.shape[0], 1) )))
        
        # save for back propagation
        self.relu_out = relu_output
        
        softmax_preact = np.dot(relu_output, self.wb2)

        # pass output through softmax activation function to get prediction scores
        # y_pred shape is (N,10)
        y_pred = self.softmax_activation(softmax_preact)
        
        # save for back propagation
        self.yhat = y_pred
                        
        return y_pred
    
    def loss(self, y_true, y_pred):
        """
        Compute categorical cross-entropy loss function. 
        
        Sum loss contributions over the outputs (axis=1), but 
        average over the examples (axis=0)
        
        Args: 
            y_true: NxD numpy array with N examples, D outputs (one-hot labels).
            y_pred: NxD numpy array with N examples, D outputs (probabilities).
        Returns:
            loss: array of length N representing loss for each example.
        """                        
        # sum loss over cross entropy elements
        # clip for log(0) error
        loss = -np.sum(np.ma.log(y_pred) * y_true, axis = 1)
        
        return loss
        
    def evaluate(self, X, y):
        """
        Make predictions and compute loss.
        Args:
            X: NxM numpy array where n-th row is an input.
            y: NxD numpy array with N examples and D outputs (one-hot labels).
        Returns:
            loss: array of length N representing loss for each example.
        """
        predictions = self.predict(X)
        loss = np.mean(self.loss(y_true=y, y_pred=predictions))
        
        return loss
        
    def accuracy(self, y_pred, y_true):
        accuracy = 0
        for i in range(y_true.shape[0]):
            if(y_pred[i].argmax() == y_true[i].argmax()):
                accuracy += 1

        accuracy = accuracy / y_pred.shape[0]
        return accuracy
        
    def train(self, X, y, lr=0.0001, max_epochs=10, x_val=None, y_val=None, batch_size=1):
        """
        Train the neural network using stochastic gradient descent.
        
        Args:
            X: NxM numpy array where n-th row is an input.
            y: NxD numpy array with N examples and D outputs (one-hot labels).
            lr: scalar learning rate. Use small value for debugging.
            max_epochs: int, each epoch is one iteration through the train data.
            x_val: numpy array containing validation data inputs.
            y_val: numpy array containing validation data outputs.
            batch_size: the number of inputs to create a batch with
        Returns:
            history: dict with the following key, value pairs:
                     'loss' -> list containing the training loss at each epoch
                     'loss_val' -> list for the validation loss at each epoch
        """
        # loss history
        history = {'loss': [], 'loss_val': []}
        
        # each epoch is a pass through the model and back
        for epoch in range(max_epochs):

            # make a copy of training data
            X_copy = X.copy()
            y_copy = y.copy()
            
            # keep track of batch losses
            batch_losses = []
            
            # loop through batches of copied training data used to update weights and biases until there is no more
            while X_copy.shape[0] > 0:
                
                # get sample of hyperparameter batch_size to perform stochastic gradient descent with
                # if there is not enough data to to a full batch_size with, use the rest of the training data
                # train_batch_X shape is (min(batch_size, X_copy.shape[0]), 784)
                # train_batch_y  shape is (min(batch_size, y_copy.shape[0]),10)
                
                xt_size = min(batch_size, X_copy.shape[0])
                yt_size = min(batch_size, y_copy.shape[0])               
                
                train_batch_X = X_copy[:xt_size]
                train_batch_y = y_copy[:yt_size]
                                
                # get predictions from batch with randomly initialized weights and biases with first pass
                # but updated weights and biases after backpropagation
                #batch_pred = self.predict(train_batch_X)
                
                # evaluate the loss from batch predictions and append
                # this will set predictions and other variables to what was for the batch
                batch_losses.append(self.evaluate(X=train_batch_X, y=train_batch_y))
                
                # calculate gradients
                
                """ self.wb2 gradient target shape is (# hidden neurons + 1 , 10)
                
                        self.relu_out shape is (batch_size, # hidden neurons + 1)

                        self.relu_out.T shape is (# hidden neurons + 1, batch_size)
                        
                        dldZ shape is (batch_size, 10 )                        
                        
                        np.dot(self.relu_out.T, dldZ) shape is (# hidden neurons + 1, 10)
                """                            
                dLdZ = (self.yhat - train_batch_y) #shape should be (batch_size, 10)
                assert dLdZ.shape == (xt_size, 10)
                assert self.relu_out.T.shape == (self.hidden+1, xt_size)
                dLdW2 = np.dot(self.relu_out.T, dLdZ)
                    
                #update weights and bias for second layer
                try:
                    self.wb2 -= dLdW2 * lr
                except:
                    print('wb2 subtraction error')

                """ self.wb1 target shape is (M+1, # of hidden neurons)
                 
                     dldZ shape is (batch_size, 10 )

                     self.wb2 shape is (# hidden neurons + 1, 10)

                     self.wb2.T shape is (10, # hidden neurons + 1)

                     np.dot(dLdZ, self.wb2.T) shape is (batch_size, # hidden neurons + 1)

                     self.relu_out shape is (batch_size, # hidden neurons + 1)
                     
                     train_batch_X shape is (batch_size, M+1)
                     
                     train_batch_X.T shape is (M+1, batch_size)
                     
                     np.dot(train_batch_X.T, wb1_delta[:,:-1]) shape is (M+1, # of hidden neurons)
                     
                """
                
                # add column of ones to multiply bias by 
                train_batch_X = np.hstack( (train_batch_X, np.ones( (train_batch_X.shape[0], 1) ) ) )
                
                # calculate gradient for layer 1
                wb1_delta = np.dot(dLdZ, self.wb2.T) * (self.relu_out > 0).astype(np.float)                
                
                try:
                    self.wb1 -= np.dot(train_batch_X.T, wb1_delta[:,:-1]) * lr
                except:
                    print('wb1 subtraction error')
                
                X_copy = X_copy[batch_size:]
                y_copy = y_copy[batch_size:]
            
            # evaluate the loss of validation batch with one epoch of trained weights
            loss_val = self.evaluate(x_val, y_val)
                        
            # average losses from batches
            loss = np.mean(batch_losses)
            
            # get accuracy
            #acc = self.accuracy(y_pred=self.predict(x_val), y_true=y_val)            
                        
            # add losses to history dictionary
            history['loss'].append(loss)
            history['loss_val'].append(loss_val)     
        
        return history