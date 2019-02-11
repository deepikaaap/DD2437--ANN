import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Generation of (non-linearly-seperable) data. Used for 3.1.3
def genData(n=100, mA=[1.0, 0.3], mB=[0.0, -0.1], sigmaA=0.2, sigmaB=0.3):
    classA1 = np.concatenate(
        (np.random.normal(-mA[0], sigmaA, round(n / 2)), np.random.normal(mA[0], sigmaA, round(n / 2))))
    classA2 = np.random.normal(mA[1], sigmaA, n)
    classA = np.stack((classA1, classA2))
    classB1 = np.random.normal(mB[0], sigmaB, n)
    classB2 = np.random.normal(mB[1], sigmaB, n)
    classB = np.stack((classB1, classB2))

    # Shuffling the data
    shuffleA = np.arange(classA.shape[1])
    np.random.shuffle(shuffleA)
    classA = classA[:, shuffleA]
    shuffleB = np.arange(classB.shape[1])
    np.random.shuffle(shuffleB)
    classB = classB[:, shuffleB]

    # Plot the datasets
    # plt.plot(classA[0], classA[1], 'r*')
    # plt.plot(classB[0], classB[1], 'bx')
    # plt.show()

    return classA, classB


# 3.1.2 Classification with a single-layer perceptron and analysis
def singleLayerPerceptronClassification():
    classA, classB = genLinearlySeparableData()
    X = np.concatenate((classA, classB), axis=1)
    targetsA = np.ones((1, classA.shape[1]))
    targetsB = -1 * np.ones((1, classB.shape[1]))
    T = np.concatenate((targetsA, targetsB), axis=1)

    # Shuffling the data
    shuffleIndices = np.arange(X.shape[1])
    np.random.shuffle(shuffleIndices)
    X = X[:, shuffleIndices]
    T = T[:, shuffleIndices]

    # Add the bias term Xbias
    Xbias = np.ones((1, X.shape[1]))
    X = np.concatenate((X, Xbias))
    X = X.T
    T = T.T

    # Initialize the weights matrix W
    weightMean = 0
    weightSigma = 0.5
    weightScale = 10 ** (-1)
    W = np.random.normal(weightMean, weightSigma, (T.shape[1], X.shape[1])) * weightScale


# Does the batch learning for delta and perceptron
# Plots the decision boundary and MSE over epochs
def batch_learning_plots(X, T, W, classA, classB):
    # 3.1.2.0 Plot decision boundaries
    eta = 0.001
    percW, percError, percMSE, percX, percY = linearBatchLoop(X, T, W, eta, "perceptron")
    deltaW, deltaError, deltaMSE, deltaX, deltaY = linearBatchLoop(X, T, W, eta, "delta")

    # Plot decision boundary
    plt.plot(classA[0], classA[1], 'r*')
    plt.plot(classB[0], classB[1], 'bx')
    plt.plot(percX, percY, 'g--', label='Perceptron decision boundary')
    plt.plot(deltaX, deltaY, 'k:', label='Delta decision boundary')
    plt.legend()
    plt.title('Decision boundary for delta rule and perceptron')
    plt.show()

    # Plot MSE
    plt.plot(percMSE, 'g--', label='Perceptron MSE')
    plt.plot(deltaMSE, 'k:', label='Delta MSE')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over epochs')
    plt.show()


# Plots MSE over epochs for varying learning rates
def learning_rate_comparison(X, T, W):
    # 3.1.2.1 Convergence comparison for different learning rates
    eta_list = (0.0001, 0.0005, 0.001)

    for eta in eta_list:
        percW, percError, percMSE, percX, percY = linearBatchLoop(X, T, W, eta, "perceptron")
        plt.plot(percMSE, label=eta)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over epochs for varying learning rates of perceptron')
    plt.show()
    for eta in eta_list:
        deltaW, deltaError, deltaMSE, deltaX, deltaY = linearBatchLoop(X, T, W, eta, "delta")
        plt.plot(deltaMSE, label=eta)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over epochs for varying learning rates of delta rule')
    plt.show()

    # Delta-rule vs perceptron comparison for huge steps
    eta = 0.005
    percW, percError, percMSE, percX, percY = linearBatchLoop(X, T, W, eta, "perceptron")
    deltaW, deltaError, deltaMSE, deltaX, deltaY = linearBatchLoop(X, T, W, eta, "delta")
    plt.plot(percMSE, 'r--', label='Perceptron MSE')
    plt.plot(deltaMSE, 'k:', label='Delta MSE')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over epochs for large learning rates')
    plt.show()


def sequential_and_batch_omparison(X, T, W, classA, classB):
    # 3.1.2.2 Compare sequential with a batch learning approach for the Delta rule.
    eta = 0.000005
    deltaW, deltaError, deltaMSE, deltaX, deltaY = linearBatchLoop(X, T, W, eta, "delta")
    seqDeltaW, seqDeltaError, seqDeltaMSE, seqDeltaX, seqDeltaY = linearSequentialLoop(X, T, W, eta, "delta")

    # Plot decision boundary
    plt.plot(classA[0], classA[1], 'r*')
    plt.plot(classB[0], classB[1], 'bx')
    plt.plot(seqDeltaX, seqDeltaY, 'g--')
    plt.plot(deltaX, deltaY, 'k:')
    plt.show()

    # Plot MSE
    plt.plot(seqDeltaMSE, 'r--', label='Sequential delta MSE')
    plt.plot(deltaMSE, 'k:', label='Batch delta MSE')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over epochs for delta rule')
    plt.show()

    # Plot error ratio
    plt.plot(seqDeltaError, 'r--', label='Sequential delta error')
    plt.plot(deltaError, 'k:', label='Batch delta error')
    plt.xlabel('epochs')
    plt.ylabel('Error rate')
    plt.legend()
    plt.title('Error rate over epochs for delta rule')
    plt.show()


# Loop used by singleLayerPerceptronClassification()
def linearBatchLoop(X, T, W, eta, learningType):
    epochs = 100
    Error = np.ones((epochs, 1))
    MSE = np.ones((epochs, 1))
    N = X.shape[0]

    for epoch in range(0, epochs):
        WX = np.dot(W, X.T)
        Y = WX.T

        if (learningType == "perceptron"):
            Y[Y >= 0] = 1.0
            Y[Y < 0] = -1.0

            del_W = eta * np.dot((T - Y).T, X)
            W = W + del_W

        elif (learningType == "delta"):
            del_W = eta * np.dot((T - Y).T, X)
            W = W + del_W

            Y[Y >= 0] = 1.0
            Y[Y < 0] = -1.0

        # Compute errorRate and MSE each epoch
        correctlyClassified = np.sum(Y == T)
        accuracy = correctlyClassified / N
        errorRate = 1 - accuracy
        Error[epoch] = errorRate
        mse = ((Y - T) ** 2).mean(axis=None)
        MSE[epoch] = mse

        # Compute decision boundary
        WT = W[0].T
        if len(W) == 3:
            decisionSlope = (-WT[2] / WT[1]) / (WT[2] / WT[0])
            decisionOffset = -WT[2] / WT[1]
            plotX = np.linspace(-10, 10)
            plotY = plotX * decisionSlope + decisionOffset
        else:
            decisionSlope = (-WT[0] / WT[1])
            decisionOffset = 0
            plotX = np.linspace(-10, 10)
            plotY = plotX * decisionSlope + decisionOffset

    # print(learningType, "batch-loop complete")
    return W, Error, MSE, plotX, plotY


# Sequential, only written for delta rule
def linearSequentialLoop(X, T, W, eta, learningType):
    epochs = 100
    Error = np.ones((epochs, 1))
    MSE = np.ones((epochs, 1))
    N = X.shape[0]

    for epoch in range(0, epochs):
        if (learningType == "delta"):
            Y = np.zeros((N, 1))
            for sample in range(0, N):
                Wx = np.dot(W, X[sample].T)
                y = Wx.T
                Y[sample] = y
                del_W = eta * np.dot((T[sample] - y)[0], X[sample])
                W = W + del_W

            Y[Y >= 0] = 1.0
            Y[Y < 0] = -1.0

        # Compute errorRate and MSE each epoch
        correctlyClassified = np.sum(Y == T)
        accuracy = correctlyClassified / N
        errorRate = 1 - accuracy
        Error[epoch] = errorRate
        mse = ((Y - T) ** 2).mean(axis=None)
        MSE[epoch] = mse

        # Compute decision boundary
        WT = W[0].T
        decisionSlope = (-WT[2] / WT[1]) / (WT[2] / WT[0])
        decisionOffset = -WT[2] / WT[1]
        plotX = np.linspace(-3, 3)
        plotY = plotX * decisionSlope + decisionOffset

    # print(learningType, "sequential-loop complete")
    return W, Error, MSE, plotX, plotY


# Perceptron without bias
def perceptron_without_bias():
    # Well-separable classes
    # muA = [-5, -3]
    # muB = [3, 3]
    muA = [-8, -5]
    muB = [5, 4]

    # Classes closer to each other
    # muA = [-2.5, -3.5]
    # muB = [1, -1]
    classA, classB = genLinearlySeparableData(muA, muB)
    X = np.concatenate((classA, classB), axis=1)
    targetsA = np.ones((1, classA.shape[1]))
    targetsB = -1 * np.ones((1, classB.shape[1]))
    T = np.concatenate((targetsA, targetsB), axis=1)

    # Shuffling the data
    shuffleIndices = np.arange(X.shape[1])
    np.random.shuffle(shuffleIndices)
    X = X[:, shuffleIndices]
    T = T[:, shuffleIndices]
    # Initialize the weights matrix W
    weightMean = 0
    weightSigma = 0.5
    weightScale = 10 ** (-1)
    eta = 0.00001
    W = np.random.normal(weightMean, weightSigma, (T.shape[1], X.shape[1])) * weightScale

    percW, percError, percMSE, percX, percY = linearBatchLoop(X, T, W, eta, "perceptron")
    deltaW, deltaError, deltaMSE, deltaX, deltaY = linearBatchLoop(X, T, W, eta, "delta")
    # Plot decision boundary
    plt.plot(classA[0], classA[1], 'r*')
    plt.plot(classB[0], classB[1], 'bx')
    plt.plot(percX, percY, 'g--', label='Perceptron Decision boundary')
    plt.plot(deltaX, deltaY, 'k:', label='Delta Decision Boundary')
    plt.legend()
    plt.title('Decision boundary for delta rule and perceptron')
    plt.show()

    # Plot MSE
    plt.plot(percMSE, 'g--', label='Perceptron MSE')
    plt.plot(deltaMSE, 'k:', label='Delta MSE')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE Vs. epochs')
    plt.show()


# 3.1.3
def nonSeparableClassification():
    classA, classB = genData()
    N = classA.shape[1] + classB.shape[1]

    # print(classA.shape)
    # print(classB.shape)

    # Apply Delta learning rule in batch mode for:
    # - Full dataset
    # (Do nothing)

    # - Remove random 25% from each class
    # classA = classA[:, round(N/8):]
    # classB = classB[:, round(N/8):]

    # - Remove random 50% from classA
    # classA = classA[:, round(N/4):]

    # - Remove random 50% from classB
    # classB = classB[:, round(N/4):]

    # - Remove 20% from classA for which classA(1,:)<0
    # 		and 80% from classA for which classA(1,:)>0
    classApos = classA[:, classA[0, :] >= 0]
    classApos = classApos[:, round(N / 5):]
    classAneg = classA[:, classA[0, :] < 0]
    classAneg = classAneg[:, round(N / 20):]
    classA = np.concatenate((classApos, classAneg), axis=1)

    # Shuffling the data
    shuffleA = np.arange(classA.shape[1])
    np.random.shuffle(shuffleA)
    classA = classA[:, shuffleA]
    shuffleB = np.arange(classB.shape[1])
    np.random.shuffle(shuffleB)
    classB = classB[:, shuffleB]

    X = np.concatenate((classA, classB), axis=1)
    targetsA = np.ones((1, classA.shape[1]))
    targetsB = -1 * np.ones((1, classB.shape[1]))
    T = np.concatenate((targetsA, targetsB), axis=1)

    # Shuffling the data
    shuffleIndices = np.arange(X.shape[1])
    np.random.shuffle(shuffleIndices)
    X = X[:, shuffleIndices]
    T = T[:, shuffleIndices]

    # Add the bias term Xbias
    Xbias = np.ones((1, X.shape[1]))
    X = np.concatenate((X, Xbias))
    X = X.T
    T = T.T

    # Initialize the weights matrix W
    weightMean = 0
    weightSigma = 0.5
    weightScale = 10 ** (-1)
    W = np.random.normal(weightMean, weightSigma, (T.shape[1], X.shape[1])) * weightScale

    eta = 0.0005
    deltaW, deltaError, deltaMSE, deltaX, deltaY = linearBatchLoop(X, T, W, eta, "delta")

    # Plot decision boundary
    plt.plot(classA[0], classA[1], 'r*')
    plt.plot(classB[0], classB[1], 'bx')
    plt.plot(deltaX, deltaY, 'k:', label='Delta decision boundary')
    plt.legend()
    plt.title('Decision boundary for delta rule')
    plt.show()

    # Plot MSE
    plt.plot(deltaMSE, 'k:', label='Delta MSE')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE over epochs')
    plt.show()

    # Plot error ratio
    plt.plot(deltaError, 'k:', label='Delta error rate')
    plt.xlabel('epochs')
    plt.ylabel('Error rate')
    plt.legend()
    plt.title('Error rate over epochs')
    plt.show()


