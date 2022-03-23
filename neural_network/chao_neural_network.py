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
        
        # initial arrays that will hold the predicted values from the forward propagation
        # to be used later when doing backpropagation
        #self.f_layer_outputs = []
        #self.f_relu_outputs = []
        #self.f_second_layer_outputs = []
        #self.f_softmax_outputs = []
            
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
            #self.f_layer_outputs.append(layer_output[0])
        
            # pass through ReLU activation function for each element layout_output
            # relu_output shape = (1, # hidden neurons)
            relu_output = self.relu_activation(layer_output)
            #self.f_relu_outputs.append(relu_output[0])
            
            # take relu_output and multiply by another layer to get to the output vector of 10
            # relu_output shape is (1, # hidden neurons), self.o_weight is shape (# hidden neurons, 10)
            # output shape is (1,10)
            second_layer_output = np.dot(relu_output, self.o_weight) + self.o_bias
            #self.f_second_layer_outputs.append(second_layer_output[0])

            # pass output through softmax activation function to get prediction scores
            # softmax_output shape is (1,10)
            softmax_output = self.softmax_activation(second_layer_output)
            #self.f_softmax_outputs.append(softmax_output[0])
            y_pred.append(softmax_output[0])
            
        # convert to numpy arrays
        y_pred = np.array(y_pred)
        #self.f_second_layer_outputs = np.array(self.f_second_layer_outputs)
        #self.f_relu_outputs = np.array(self.f_relu_outputs)
        #self.f_layer_outputs = np.array(self.f_layer_outputs)
        #self.f_softmax_outputs = np.array(self.f_softmax_outputs)
            
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
            batch_size: the number of elements
        Returns:
            history: dict with the following key, value pairs:
                     'loss' -> list containing the training loss at each epoch
                     'loss_val' -> list for the validation loss at each epoch
        """
        # loss history
        history = {'loss': [], 'loss_val': []}
        
        # get rows processing
        total_rows = X.shape[0]
        
        # check that the batch_size given is not greater than what is the population size
        #if batch_size > total_rows:
        #    print('batch size given is too large, please lower the batch size')
        #    batch_size = 1
        if batch_size % total_rows != 0:
            batch_size = 1
            
        index_start = 0
        index_end = batch_size
        
        for i in range(max_epochs):
            #random_start = np.random.randint(low=0, high=pop_size)
            #random_end = random_start + batch_size
            #train_batch_X = X[random_start:random_end, :]
            #train_batch_y = y[random_start:random_end, :]
            # get sample of hyperparameter batch_size to perform stochastic gradient descent with
            train_batch_X = X[index_start:index_end, :]
            train_batch_y = y[index_start:index_end, :]
             
            test_batch_X = x_val[index_start:index_end, :]
            test_batch_y = y_val[index_start:index_end, :]
                
            # calculate the loss with each sample batch
            loss = self.evaluate(train_batch_X, train_batch_y)
            loss_val = self.evaluate(test_batch_X, test_batch_y)
            
            # get weights to use
            f_layer_outputs = []
            f_relu_outputs = []
            f_second_layer_outputs = []
            f_softmax_outputs = []
            
            # predicted values from the forward propagation to be used later when doing backpropagation
            for row in train_batch_X:
                layer_output = np.dot(row.T, self.h_weight) + self.h_bias
                f_layer_outputs.append(layer_output)
        
                # pass through ReLU activation function for each element layout_output
                # relu_output shape = (1, # hidden neurons)
                relu_output = self.relu_activation(layer_output)
                f_relu_outputs.append(relu_output)
                
                # take relu_output and multiply by another layer to get to the output vector of 10
                # relu_output shape is (1, # hidden neurons), self.o_weight is shape (# hidden neurons, 10)
                # output shape is (1,10)
                second_layer_output = np.dot(relu_output, self.o_weight) + self.o_bias
                f_second_layer_outputs.append(second_layer_output)

                # pass output through softmax activation function to get prediction scores
                # softmax_output shape is (1,10)
                softmax_output = self.softmax_activation(second_layer_output)
                f_softmax_outputs.append(softmax_output)
        
            f_layer_outputs = np.array(f_layer_outputs)
            f_relu_outputs = np.array(f_relu_outputs)
            f_second_layer_outputs = np.array(f_second_layer_outputs)
            f_softmax_outputs = np.array(f_softmax_outputs)
        
            # take derivative of categorical cross entropy loss function with respect to
            # the weights and biases
            #yhat = self.predict(train_batch_X)
            
            # dloss_dz = softmax_output (yhat) - train_batch_y
            # dL_w2= np.outer(dloss_dz, relu_output.T)
            # dL_zhid = np.dot((dloss_dz.T * output.T), float(layer_output > 0)).T
            # dL_w1 = np.outer(dL_zhid, train_batch_X)
            dloss_dz = f_softmax_outputs - train_batch_y
            dL_w2 = np.outer(dloss_dz, f_relu_outputs.T)
            dL_zhid = np.dot((dloss_dz.T * f_second_layer_outputs.T), (f_layer_outputs > 0).astype(np.float)).T
            dL_w1 = np.outer(dL_zhid, train_batch_X)
            
            #update the class weights and biases
            self.h_weight += self.h_weight - (dL_w1.T * lr) # old weight - (new weight derivative value * lr)
            self.h_bias += self.h_bias - (dL_zhid.T * lr)
            self.o_weight += self.o_weight - (dL_w2.T *lr)
            self.o_bias += self.o_bias - (dloss_dz.T * lr)

            # redo the loop but optimized weights and biases
            # on validation data

            # update batch of data used to train
            index_start += batch_size
            index_end += batch_size
            
            # add losses to history
            history['loss'].append(loss)
            history['loss_val'].append(loss_val)     
        
        return history