import numpy as np
def update_learning_rate(learning_rate0, epoch_num, decay_rate):
    learning_rate = learning_rate0 * (1 / (1 + decay_rate*epoch_num))
    return learning_rate

def schedule_learning_rate_decay(learning_rate0, epoch_num, decay_rate, time_interval = 1000):
    learning_rate = learning_rate0 * (1 / (1+decay_rate*np.floor(epoch_num / time_interval)))
    return learning_rate