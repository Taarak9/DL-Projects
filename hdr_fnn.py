from generalized_fnn import FNN
from mnist_loader import load_data_wrapper 

# MNIST data split
training_data, validation_data, test_data = load_data_wrapper()

# handwritten digit recognizer 
# Loss function: Cross Entropy
hdr = FNN(784, "ce")
hdr.add_layer(80, "sigmoid")
hdr.add_layer(10, "sigmoid")

# Trained for 30 epochs using Stochastic Gradient Descent (mini-batch size = 10 and learning_rate = 3)
# Displaying Test results
hdr.SGD(training_data, 30, 10, 3.0, test_data=test_data, task="classification")

# epoch vs error graph
hdr.logging(test_data)
