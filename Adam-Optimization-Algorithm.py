import numpy as np
def initialize_adam(parameters):
    L = len(parameters) // 2 #get the number of layers in the neural network
    #initialize gradient descent with momentum, and RMS prop separately
    v = {}
    s = {}
    for l in range(1, L+1):
        v["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        v["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)
        s["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        s["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)
    return v, s

#Note: all positional arguments must come before keyword arguments

def update_parameters_with_adam(parameters, grads, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    #parameters = paramters of a leyer's neurons
    #grads = grads of a layer's neurons
    #v = velocity of grad update for gradient descent with momentum
    #s = squared velocity update for RMSprop
    #t = number of steps taken of ADAM(Adaptive Moment Estimation)
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(1, L+1):
        #Moving average(velocity) of the gradients
        v["dW"+str(l)] = beta1*v["dW"+str(l)] + (1-beta1)*grads["dW"+str(l)]
        v["db"+str(l)] = beta1*v["db"+str(l)] + (1-beta1)*grads["db"+str(l)]

        #Moving average of squared gradients(for RMSprop)
        s["dW"+str(l)] = beta2*["dW"+str(l)] + (1-beta2)*np.power(grads["dW"+str(l)], 2)
        s["db"+str(l)] = beta2*["db"+str(l)] + (1-beta2)*np.power(grads["db"+str(l)], 2)

        #corrected velocities
        v_corrected["dW"+str(l)] = v["dW"+str(l)] / (1 - np.power(beta1, t))
        v_corrected["db"+str(l)] = v["db"+str(l)] / (1 - np.power(beta1, t))

        #corrected average of squared gradients
        s_corrected["dW"+str(l)] = s["dW"+str(l)] / (1 - np.power(beta2, t))
        s_corrected["db"+str(l)] = s["db"+str(l)] / (1 - np.power(beta2, t))

        #update parameters
        parameters["W"+str(l)] -= learning_rate*(v_corrected["dW"+str(l)] / (np.sqrt(s_corrected["dW"+str(l)])+epsilon))
        parameters["b"+str(l)] -= learning_rate*(v_corrected["db"+str(l)] / (np.sqrt(s_corrected["db"+str(l)])+epsilon))

    return parameters, v, s, v_corrected, s_corrected

    

