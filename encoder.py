import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def auto_encoder(Nhidden=2,iterations=100):
    X = [[1,-1,-1,-1,-1,-1,-1,-1],
        [-1,1,-1,-1,-1,-1,-1,-1],
        [-1,-1,1,-1,-1,-1,-1,-1],
        [-1,-1,-1,1,-1,-1,-1,-1],
        [-1,-1,-1,-1,1,-1,-1,-1],
        [-1,-1,-1,-1,-1,1,-1,-1],
        [-1,-1,-1,-1,-1,-1,1,-1],
        [-1,-1,-1,-1,-1,-1,-1,1]]
    X = np.array(X)
    T  = X
    Xbias = np.ones((X.shape[0], 1))
    X = np.concatenate((X, Xbias), axis = 1)

    weightMean = 0
    weightSigma = 0.5
    weightScale = 10 ** (-1)
    W = np.random.normal(weightMean, weightSigma, (X.shape[1], Nhidden)) * weightScale

    # Initialize the weights matrix V
    veightMean = 0
    veightSigma = 0.5
    veightScale = 10 ** (-1)

    V = np.random.normal(veightMean, veightSigma, (Nhidden+1, T.shape[0])) * veightScale
    backprop(X, T, W, V,iterations)

# Backprop - The generalized Delta rule
def backprop(X, T, W, V, iters=100, eta=0.001, momentum = True, alpha = 0.0005):
    Ndata = X.shape[0]
    it = 0
    # for it in range(iters):
    while True:
        it+=1
        # Forward pass
        hin = np.dot(X, W)
        hout = phi(hin)
        extendH = np.ones((Ndata, 1))
        hout = np.concatenate((hout, extendH), axis=1)

        oin = np.dot(hout, V)
        out = phi(oin)

        # Backward pass
        # Find deltaO and deltaH
        Y = out - T
        phi_derivative = phiDerivative(out)
        deltaO = np.multiply(Y, phi_derivative)
        # deltaO = np.resize(deltaO.T, 2)
        Y = np.matmul(deltaO, V.T)
        phi_derivative = phiDerivative(hout)
        deltaH = Y * phi_derivative
        deltaH = deltaH[:, :-1]

        deltaW = -eta * np.matmul(np.transpose(X), deltaH)
        deltaV = -eta * np.matmul(np.transpose(hout), deltaO)

        if momentum == False:
            # Weight update
            # Re-estimate deltaW and deltaV and update W, V

            W += deltaW.T
            V += deltaV.T
        else:
            deltaW = (deltaW * alpha) - np.dot(X.T,deltaH) * (1 - alpha)
            deltaV = (deltaV * alpha) - np.dot(hout.T, deltaO) * (1 - alpha)

            W = W + deltaW * eta
            V = V + deltaV * eta
        out[out >= 0] = 1
        out[out < 0] = -1

        if np.array_equal(out, T):
            break
    print("Output is", out)
    print("Iterations to converge", it)


def phiDerivative(x):
    return ((1 + x) * (1 - x)) * 0.5


#Transfer function [phi(x)] => tanh(x/2)
def phi(x):
    phix = np.divide(2, (1+np.exp(-x))) - 1
    return phix

auto_encoder(Nhidden=3)