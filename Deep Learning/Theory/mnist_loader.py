"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data. For details, see the doc strings
for `load_data` and `load_data_wrapper`.
"""

#### Libraries
import pickle
import gzip
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training, validation, and test data."""
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with 1.0 in the jth position."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    """Return (training_data, validation_data, test_data) in a format convenient for neural networks."""
    tr_d, va_d, te_d = load_data()
    
    # Training data: (x, y) where x is input, y is vectorized output
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))  # wrap zip with list for Python 3

    # Validation data: (x, y) where y is just the digit
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # Test data: (x, y) where y is just the digit
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)
