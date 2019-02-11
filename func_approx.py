import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# np.random.seed(1234)
MSE_list= []
def generate_data(Nhidden=4):

    x = (np.arange(-5,5.5,0.5)).T
    y = (np.arange(-5,5.5,0.5)).T
    # Only for visualization.
    z = (np.exp(-np.multiply(x,x) * 0.1) * np.exp(-np.multiply(y,y) * 0.1)).T - 0.5
    print(x.shape)
    print(y.shape)
    print(z.shape)
    # Plot the bell-shaped 'z'
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x, y, z)
    # plt.show()
    ndata = len(x) * len(y)
    [xx, yy] = np.meshgrid(x, y)
    # print(xx.shape)
    # print(np.reshape(xx, ndata))

    patterns = np.asarray([np.reshape(xx, ndata),np.reshape(yy, ndata)])
    noise = np.random.multivariate_normal([0], [[0.05]], (21))
    print(noise.shape)
    z = (np.exp(-np.multiply(xx, xx) * 0.1) * np.exp(-np.multiply(yy, yy) * 0.1)).T - 0.5 + noise
    print(z.shape)
    Xbias = np.ones((1,patterns.shape[1]))
    patterns = np.concatenate((patterns, Xbias), axis=0)
    targets = np.reshape(z, (1, ndata))

    ax = plt.axes(projection='3d')
    ax.plot3D(patterns[0], patterns[1], targets.flatten())
    plt.show()

    weightMean = 0
    weightSigma = 0.5
    weightScale = 10 ** (-1)
    W = np.random.normal(weightMean, weightSigma, (patterns.shape[0], Nhidden)) * weightScale

    # Initialize the weights matrix V
    veightMean = 0
    veightSigma = 0.5
    veightScale = 10 ** (-1)

    V = np.random.normal(veightMean, veightSigma, (Nhidden + 1, targets.shape[0])) * veightScale
    # shuffleIndices = np.arange(patterns.shape[1])
    # np.random.shuffle(shuffleIndices)
    # patterns = patterns[:, shuffleIndices]
    # targets = targets[:, shuffleIndices]
    patterns_train = patterns[:, 0:round(ndata * 0.8)]
    targets_train = targets[:, 0:round(ndata * 0.8)]
    patterns_test = patterns[:, round(ndata * 0.8):]
    targets_test = targets[:, round(ndata * 0.8):]
    print(patterns.shape)
    print(targets.shape)
    MSE_list.append(backprop(patterns_train, targets_train,patterns_test,targets_test,W,V))
    # print(MSE_list)


    # fig = plt.figure()
    # animation.FuncAnimation(fig, backprop, fargs=(patterns, targets,W,V),interval=30,blit=False)

# Backprop - The generalized Delta rule
def backprop(X, T,A,B, W, V, iters=500, eta=0.005, momentum = True, alpha = 0.0005):
    Ndata = X.shape[1]

    MSE=[]
    MSE_test=[]
    for it in range(iters):
        # Forward pass
        hin = np.dot(X.T, W)
        hout = phi(hin)
        extendH = np.ones((Ndata, 1))
        hout = np.concatenate((hout, extendH), axis=1)
        oin = np.dot(hout, V)
        out = phi(oin).T
        # Backward pass
        # Find deltaO and deltaH
        Y = out - T
        phi_derivative = phiDerivative(out)
        deltaO = np.multiply(Y, phi_derivative)
        # deltaO = np.resize(deltaO.T, 2)
        Y = np.matmul(deltaO.T, V.T)
        phi_derivative = phiDerivative(hout)
        deltaH = Y * phi_derivative
        deltaH = deltaH[:, :-1]

        deltaW = -eta * np.matmul(X, deltaH)
        deltaV = -eta * np.matmul(hout.T, deltaO.T)

        if momentum == False:
            # Weight update
            # Re-estimate deltaW and deltaV and update W, V

            W += deltaW.T
            V += deltaV.T
        else:
            deltaW = (deltaW * alpha) - np.dot(X,deltaH) * (1 - alpha)
            deltaV = (deltaV * alpha) - np.dot(hout.T, deltaO.T) * (1 - alpha)

            W = W + deltaW * eta
            V = V + deltaV * eta
        test_op = A[0] * deltaW[0][0] + A[1] * deltaW[0][1] + deltaW[0][2]

        mse = ((out - T) ** 2).mean(axis=None)
        mse_test = ((test_op - B) ** 2).mean(axis=None)
        MSE.append(mse)
        MSE_test.append(mse_test)

    print(it)
    # ax = plt.axes(projection='3d')
    # ax.plot3D(X[0], X[1], out.flatten())
    # plt.show()
    # plt.plot(range(1,it+1), MSE[1:],label='training error')
    # plt.plot(range(1,it+1), MSE_test[1:],label='test error')
    # plt.ylabel('MSE')
    # plt.xlabel('number of epochs')
    # plt.legend()
    # plt.title('Training and testing error over epochs')
    # plt.show()
    return mse


def phiDerivative(x):
    return ((1 + x) * (1 - x)) * 0.5


#Transfer function [phi(x)] => tanh(x/2)
def phi(x):
    phix = np.divide(2, (1+np.exp(-x))) - 1
    return phix

Nhidden_nodes = [3,5,10,15,20,25]
for n in Nhidden_nodes:
    generate_data(Nhidden=n)


print(MSE_list)
plt.plot(Nhidden_nodes, MSE_list)
plt.xlabel("Number of nodes in the hidden layer")
plt.ylabel("MSE after 100 epochs")
plt.show()

