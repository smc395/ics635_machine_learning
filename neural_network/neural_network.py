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
        self.h_bias = np.random.randn(1,hidden)
               
        # initialize weights and bias for output layer
        self.o_weight = np.random.randn(hidden, outputs)
        self.o_bias = np.random.randn(1, outputs)
            
    """ ReLU activation function """
    def relu_activation(self, n_inputs):
        return np.maximum(n_inputs, 0)
    
    """ Softmax activation function """
    def softmax_activation(self, n_inputs):
        return np.exp(n_inputs) / np.sum(np.exp(n_inputs), axis=1)
                   
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
        
        # y_true one-hot label cross_entropy loss simplifies to -log(y_pred)
        ce_loss = -np.log(y_pred)
                
        # sum loss over cross entropy elements
        loss = np.sum(ce_loss, axis=1)
        
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
        loss = self.loss(y, predictions)
        
        return loss
        
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
            layer_output = np.dot(row.T, self.h_weight) + self.h_bias
        
            # pass through ReLU activation function
            relu_output = self.relu_activation(layer_output)
            
            # take relu_output and multiply by another layer
            # to get to the output vector of 10
            output = np.dot(relu_output.T, self.o_weight) + self.o_bias

            # pass final output through softmax
            softmax_output = self.softmax_activation(output)
        
            y_pred.append(softmax_output)        
        
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
        # 
        epoch = 1
        
        batch_size = 100
        
        while(epoch < max_epochs):
            
            #TODO sample X to calculate the gradient descent

            layer_output = np.dot(X.T, self.h_weight) + self.h_bias

            # pass through activation function
            relu_output = self.relu_activation(layer_output)

            # pass final output through softmax
            softmax_output = self.softmax_activation(relu_output)

            #optimize


            #update the class weights and biases
        
        return
                