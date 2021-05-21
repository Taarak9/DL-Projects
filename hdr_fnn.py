from fnn import FNN
from mnist_loader import load_data_wrapper 

training_data, validation_data, test_data = load_data_wrapper()
nn = FNN([784, 30 , 10])
nn.SGD(training_data, 100, 10, 3.0, test_data=test_data)
