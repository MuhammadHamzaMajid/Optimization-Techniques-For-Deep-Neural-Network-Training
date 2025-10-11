import numpy as np
import math

#1. Create the mini-batches
def random_mini_batches(X, Y, mini_batch_size):
    m = X.shape[1]
    mini_batches = []
    #a. shuffle the dataset
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    inc = mini_batch_size

    #b. partition the dataset
    #case where the mini-batch size is complete
    num_complete_minibatches = math.floor(m / inc)
    for k in range(num_complete_minibatches):
        start = k*inc
        end = (k+1)*inc
        mini_batch_X = shuffled_X[:, start:end]
        mini_batch_Y = shuffled_Y[:, start:end]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    #the last minibatch
    if m % inc != 0:
        mini_batch_X = shuffled_X[:, m - m % inc:m]
        mini_batch_Y = shuffled_Y[:, m - m % inc:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

'''
#intuition for mini-btach gradient descent
for i in num_epochs:
    for j in len(minibatches):
        forward prop
        cost
        backward prop
        update parameters
    end
end'''