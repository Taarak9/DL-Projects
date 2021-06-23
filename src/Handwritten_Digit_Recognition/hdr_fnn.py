import numpy as np
from matplotlib import pyplot as plt
from mnist_loader import load_data_wrapper 
from customdl import FNN

# MNIST data split
training_data, validation_data, test_data = load_data_wrapper()

# handwritten digit recognizer 
# Loss function: Cross Entropy
hdr = FNN(784, "ce")
hdr.add_layer(80, "sigmoid")
hdr.add_layer(10, "sigmoid")

hdr.compile(training_data, test_data)

def display_image(input_vector) :
    image = np.reshape(input_vector, (28, 28))

    plt.imshow(image, cmap='gray')
    plt.show()

# test results for few samples
for i in np.arange(0, 19):
  display_image(test_data[i][0])
  print("HDR output: ", np.argmax(hdr.feedforward(test_data[i][0])))
