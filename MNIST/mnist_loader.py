# Libraries
import cPickle
import gzip
import numpy as np

def load_data():

    # training_data is returned as tuple with two entries
    # each entry has numpy array with 50,000 entries
    # first entry is training images
    # second entry is digit values

    # validation_data and test_data similar to training data, but with 10,000 entries

    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():

    # modify loaded data into format more easily in neural network
    tr_d, va_d, te_d = load_data()

    # training_data is list containing 50,000 tuples
    # first entry in each tuple is image
    # second entry in each tuple is vectorized result (see next function)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    # similar to training_data, but second entry in each tuple is result in integer
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):

    # return 10-dimensional unit vector with 1.0 as element j
    # e.g. if j = 5, return [0,0,0,0,1.0,0,0,0,0]
    v = np.zeros((10, 1))
    v[j] = 1.0
    return v