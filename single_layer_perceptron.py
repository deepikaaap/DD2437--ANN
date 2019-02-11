import numpy as np
import matplotlib.pyplot as plt


#____________________________________________________________________________#
# 				3	Assignment - Part 1
#____________________________________________________________________________#


#____________________________________________________________________________#
#			3.1 Classification with a single-layer perceptron
#____________________________________________________________________________#



# 3.1.1 Generation of linearly-seperable data
def genLinearlySeparableData():
	n = 100
	mA = [0.75, 1.5]
	mB = [-2.5, -1.75]
	sigmaA = 0.3
	sigmaB = 0.5

	classA1 = np.random.normal(mA[0], sigmaA, n)
	classA2 = np.random.normal(mA[1], sigmaA, n)
	classA = np.stack((classA1, classA2))
	classB1 = np.random.normal(mB[0], sigmaB, n)
	classB2 = np.random.normal(mB[1], sigmaB, n)
	classB = np.vstack((classB1, classB2))

	return classA, classB

def linearSepLoop(X, T, W, learningType):
	eta = 0.001
	epochs = 1000
	for it in range(0, epochs-1):
		misclassifications = 0
		WX = np.multiply(W,X)

		if (learningType=="perceptron"):
			# targets are 0 or 1
			T = [0 if i < 0 else 1 for i in T]
			Y = np.sum(WX, axis=0)
			# Classifies point as 0 or 1 based on the weight matrix
			Y = [1 if i > 0 else 0 for i in Y]
			Y = np.array(Y)
			print(Y.shape)
			print(X.shape)
			# Only considers the misclassified points for changing weights.
			del_W = eta*(T-Y).T*X


		elif (learningType=="delta"):
			Y = np.sum(WX,axis=0)
			del_W = eta*(T-Y).T*X.T

		W = W + del_W
		print(W.shape)
		errorRate = misclassifications/Y.shape[0]


		if (del_W.all() == 0):
			print("In break")
			break
	Y = [1 if i > 0 else -1 for i in Y]
	return Y, W



# 3.1.2 Classification with a single-layer perceptron and analysis
def singleLayerPerceptronClassification():
	# Generate data for classification
	classA, classB = genLinearlySeparableData()

	# Stacks the two classes into the input matrix
	X = np.concatenate((classA, classB), axis=1)
	# Adding a row to the data matrix to account for the bias.
	X = np.vstack((X, np.ones(X.shape[1])))

	# Label the inputs to their respective classes -1/1
	targetsA = np.ones((classA.shape[1]))
	targetsB = -1*np.ones((classB.shape[1]))
	T = np.concatenate((targetsA, targetsB))


	# Initializing a weight matrix
	W = np.random.normal(0, 0.1, X.shape)
	# To shuffle the data from the two classes.
	shuffleIndices = np.arange(X.shape[1])
	np.random.shuffle(shuffleIndices)
	X = X[:, shuffleIndices]
	T = T[shuffleIndices]

	pred, trained_W = linearSepLoop(X, T, W, "perceptron")
	# pred, trained_W = linearSepLoop(X, T, W, "delta")
	a = -trained_W[0][0] / trained_W[1][0]
	b = -trained_W[2][0] / trained_W[1][0]
	plotX = np.linspace(-5, 5)
	plotY = plotX * a + b
	pred = np.array(pred)

	plt.plot(plotX, plotY, '--g', label='Decision Boundary')
	plt.plot(classA[0], classA[1], 'r*')
	plt.plot(classB[0], classB[1], 'bx')
	plt.show()



singleLayerPerceptronClassification()