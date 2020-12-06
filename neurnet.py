import numpy as np
from scipy.stats import truncnorm

# sigmioid activation function
@np.vectorize
def sigmoid(x):
    return 1/(1 + np.e**(-x))

activation_function = sigmoid

# truncated normal function
def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# neural network
class NeuralNetwork:

    # constructor
    def __init__(self, network_structure, learning_rate):

        self.structure = network_structure # [n_nodes_layer_1, n_nodes_layer_2, ..., n_nodes_layer_n]
        self.learning_rate = learning_rate
        self.create_weights_matrices()

    # initialize weight matrices
    def create_weights_matrices(self):

        # initialize objects
        self.weights_matrices = []
        no_of_layers = len(self.structure)
        layer_index = 1

        # loop through layers
        while layer_index < no_of_layers:

            # grab number of input and output nodes
            nodes_in = self.structure[layer_index - 1]
            nodes_out = self.structure[layer_index]

            # parameterize truncated normal distr. based on no. of input nodes
            # (weights and biases)
            n = (nodes_in + 1)*nodes_out
            lim = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2, sd=1, low=-lim, upp=lim)
            
            # make random weights and biases matrix based on above distr.
            wm = X.rvs(n).reshape((nodes_out, nodes_in + 1))
            self.weights_matrices.append(wm)

            # update counter
            layer_index += 1

    # train for single case
    def train(self, input_vector, target_vector):

        # grab no. layers
        no_of_layers = len(self.structure)

        # forward propagation

        # initialize objects
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 1
        activations = [input_vector]

        # loop downstream through layers
        while layer_index < no_of_layers:

            # append bias node to input vector
            in_vector = activations[-1]
            in_vector = np.concatenate((in_vector, [[1]]))
            activations[-1] = in_vector

            # compute output vector
            out_vector = activation_function(
                np.dot(self.weights_matrices[layer_index - 1], in_vector))
            activations.append(out_vector)

            # update counter
            layer_index += 1
            
        # backpropagation

        # initialize objects
        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T

        # compute error in output layer
        output_errors = target_vector - out_vector

        # loop upstream through layers
        while layer_index > 0:

            # grab activations in upstream and downstream layers
            out_vector = activations[layer_index]
            in_vector = activations[layer_index - 1]

            # remove the 1 (for bias) for all layers except output layer
            if layer_index != (no_of_layers - 1):
                out_vector = out_vector[:-1, :].copy()

            # update weights and biases matrix
            tmp = output_errors*out_vector*(1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)
            self.weights_matrices[layer_index - 1] += self.learning_rate*tmp

            # compute errors for upstream layer, removing last element (for bias)
            output_errors = np.dot(self.weights_matrices[layer_index - 1].T, output_errors)
            output_errors = output_errors[:-1, :]

            # update counter
            layer_index -= 1

    # predict single case
    def predict(self, input_vector):
        no_of_layers = len(self.structure)
        input_vector = np.concatenate((input_vector, [1]))
        in_vector = np.array(input_vector, ndmin=2).T
        layer_index = 1
        while layer_index < no_of_layers:
            out_vector = activation_function(
                np.dot(self.weights_matrices[layer_index - 1], in_vector))
            in_vector = out_vector
            in_vector = np.concatenate((in_vector, [[1]]))
            layer_index += 1
        return out_vector

    # compute quality indicators
    def quality(self, X, y):

        # initialize arrays
        cm = np.zeros((10, 10), int)
        p = np.zeros((10))
        r = np.zeros((10))
        corrects, wrongs = np.array([0]), np.array([0])

        # build confusion matrix
        for i in range(len(X)):
            res = self.predict(X[i])
            res_max = res.argmax()
            target = y[i]
            cm[res_max, int(target)] += 1
            if target == res_max:
                corrects += 1
            else:
                wrongs += 1
            
        # compute precision and recall
        for i in range(10):
            row = cm[int(i), :]
            col = cm[:, int(i)]
            p[i] = cm[i, i]/row.sum()
            r[i] = cm[i, i]/col.sum()

        return cm, p, r, corrects, wrongs

   

