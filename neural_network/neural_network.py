# Part 1: Defining the neural network.
import numpy as np

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
        self.h_weight = np.random.randn(inputs, hidden)
        self.h_bias = np.random.randn(1, hidden)
               
        # initialize weights and bias for output layer
        self.o_weight = np.random.randn(hidden, outputs)
        self.o_bias = np.random.randn(1, outputs)
            
    """ ReLU activation function """
    def relu_activation(self, n_inputs):
        return np.maximum(n_inputs, 0)
    
    """ Softmax activation function """
    def softmax_activation(self, n_inputs):
        # softmax is e^input / sum( e^input for all inputs )
        # add axis=1 to perform operation within each row in the matrix
        # subtract the max of the row to prevent overflow error
        e_values = np.exp(n_inputs - np.max(n_inputs, axis=1))       
        return e_values / np.sum(e_values, axis=1)
                                  
    def predict(self, X):
        """
        Make predictions on inputs X.
        Args:
            X: NxM numpy array where n-th row is an input.
        Returns: 
            y_pred: NxD array where n-th row is vector of probabilities.
        """
        # our array of predictions for rows of X
        y_pred = []
        
        # for each row predict the probability that the image is a label
        # based on the trained weights and biases. Append the prediction
        for row in X:
            
            # multiply inputs by weights plus a bias
            # row shape is (1, 784), self.h_weights is (784, # hidden neurons)
            # transpose row shape to be (784, 1) to do matrix multiplication
            # layer_output shape = (1, # hidden neurons)
            layer_output = np.dot(row.T, self.h_weight) + self.h_bias
        
            # pass through ReLU activation function for each element layout_output
            # relu_output shape = (1, # hidden neurons)
            relu_output = self.relu_activation(layer_output)
            
            # take relu_output and multiply by another layer to get to the output vector of 10
            # relu_output shape is (1, # hidden neurons), self.o_weight is shape (# hidden neurons, 10)
            # output shape is (1,10)
            output = np.dot(relu_output, self.o_weight) + self.o_bias

            # pass output through softmax activation function to get prediction scores
            # softmax_output shape is (1,10)
            softmax_output = self.softmax_activation(output)
        
            y_pred.append(softmax_output)
            
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
        loss = -np.sum(np.log(y_pred) * y_true)
        
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
        
    def train(self, X, y, lr=0.0001, max_epochs=10, x_val=None, y_val=None):
        """
        Train the neural network using stochastic gradient descent.
        
        Args:
            X: NxM numpy array where n-th row is an input.
            y: NxD numpy array with N examples and D outputs (one-hot labels).
            lr: scalar learning rate. Use small value for debugging.
            max_epochs: int, each epoch is one iteration through the train data.
            x_val: numpy array containing validation data inputs.
            y_val: numpy array containing validation data outputs.
        Returns:
            history: dict with the following key, value pairs:
                     'loss' -> list containing the training loss at each epoch
                     'loss_val' -> list for the validation loss at each epoch
        """
        # how many times through the network
        epoch = 1
        
        # what batch size to use for training
        batch_size = 1
        data_points = X.shape[0]
        
        # start with a really large number
        min_loss = 9999999

        # loss history
        history = {}
        
        while(epoch < max_epochs):
            
            for(batch_size < data_points):

                loss = self.evaluate(X[batch_size:], y[batch_size:])                  
                
                batch_size += 1
            
            #derivative of categorical cross entropy loss function
            


            #update the class weights and biases
            self.h_weight +=
            self.h_bias +=
            self.o_weight +=
            self.o_bias +=
            
            epoch += 1
        
        return history