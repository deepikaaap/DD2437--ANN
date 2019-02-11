import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import neural_network as nn
from sklearn import metrics

# Data generating function - Mackey glass
def findEulerSolution(t, beta, gamma, n, tau):
    X = [0.0 for x in range(t)]
    X[0] = 1.5
    for i in range(t-1):
        X[i+1] = X[i] + ((beta * X[i-tau]) / (1 + ((X[i-tau]**n)))) - gamma * X[i]
    return X

# Input generator for the model
def generate_MG_data():
    beta = 0.2
    gamma = 0.1
    tau = 25
    n = 10
    start = 301
    end = 1500

    input = np.zeros((1199, 5))
    target = np.zeros((1199,1))
    X = findEulerSolution(end+5, beta, gamma, n, tau)
    # print(" len of X = ", len(X))
    for ind,t in enumerate(range(start,end)):
        # print("t-start = ", ind)
        input[t-start][0] = X[t-20]
        input[t-start][1] = X[t-15]
        input[t-start][2] = X[t-10]
        input[t-start][3] = X[t-5]
        input[t-start][4] = X[t]
        target[t-start][0] = X[t+5]

    print(input.shape)
    print(target.shape)
    plt.plot(range(start,end), target.flatten())
    plt.xlabel("time")
    plt.ylabel("Output of mackey glass time series equation")
    plt.title("plot of mackey glass time series")
    plt.show()
    return input, target

# Splits the data for training, validation and test
def data_split():
    input, target = generate_MG_data()
    print("shape of input =", input.shape)

    # Split training data
    input_train = input[:1000]
    target_train = target[:1000]
    target_train = np.asarray(target_train)

    # Split data as validation set for early stopping
    input_val = input_train[round(len(input_train) * 0.6):]
    target_val = np.asarray(target_train[round(len(input_train) * 0.6):])

    # Split test data
    input_test = input[1000:]
    target_test = target[1000:]
    target_test = np.asarray(target_test)

    return input_train,input_test,input_val,target_train,target_test,target_val

# Trains the regression model
def learn_regressor(nodes,learning_rate,epochs,tol__runs,input_train, target_train):
    # Initialize parameters for the model
    hidden_size = (nodes,)
    # The initial learning rate used. It controls the step-size in updating the weights
    lr = learning_rate
    epochs = epochs
    # Maximum number of epochs to not meet tol improvement
    not_meet_tol = tol__runs

    reg_model = nn.MLPRegressor(hidden_layer_sizes=hidden_size, alpha=0.2, learning_rate_init=lr, max_iter=epochs, shuffle=False, \
    momentum=0.9, early_stopping=True, validation_fraction=0.4, beta_1=0.9,beta_2=0.999, epsilon=1e-08, n_iter_no_change=not_meet_tol, solver='sgd')
    reg_model.fit(input_train, target_train)

    return reg_model

# Predicts unseen data using the trained model.
def predict_outputs(model):
    reg_model = model
    pred_op = reg_model.predict(input_test)
    MSE = metrics.mean_squared_error(pred_op, target_test)

    # Predict
    pred_op_test = reg_model.predict(input_test)
    MSE = metrics.mean_squared_error(pred_op_test, target_test)

    print("Target op",len(target_test))
    print("Pred op",len(pred_op_test))
    print("reg model = ",[coef.shape for coef in reg_model.coefs_])
    print("reg_model.n_iter_ = ", reg_model.n_iter_)

    print(MSE)



# Script run
input_train,input_test,input_val,target_train,target_test,target_val = data_split()
model = learn_regressor(5,0.01,2000,10,input_train, target_train)
predict_outputs(model)