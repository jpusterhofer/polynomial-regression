import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Calculate h(x)
def h(x, weights):
    y = 0
    for i in range(len(weights)):
        y += weights[i] * x**i
    return y

#Calculate the model's weights
def calc_weights(data, degree):
    alpha = 0.001
    weights = np.zeros(degree + 1)
    updates = np.ones(degree + 1)

    #Update weights until updates are small
    while(max(abs(updates)) >= 0.00001):
        #Calculate an update for every weight
        for i in range(len(updates)):
            update = 0
            for point in data:
                update += (h(point[0], weights) - point[1]) * point[0]**i
            update = update / len(data) * alpha
            updates[i] = update

        #Check if the updates are diverging, for testing purposes
        if updates[0] > 10:
            print("Diverging")

        #Update weights
        weights -= updates

    return weights

#Use the model to make predictions for the given inputs
def predict(data, weights):
    predictions = np.zeros((len(data), 2))
    for row in range(len(data)):
        predictions[row, 0] = data[row, 0]
        predictions[row, 1] = h(data[row, 0], weights)
    return predictions

#Calculate the mean squared error for the predictions
def get_error(data, predictions):
    error = 0
    for row in range(len(data)):
        error += (predictions[row, 1] - data[row, 1])**2
    error /= len(data)
    return error

#Use polynomial regression to predict the outputs of a dataset
#Prints the weights and error of the model
#Returns the model's predictions
def regression(data, degree):
    weights = calc_weights(data, degree)
    predictions = predict(data, weights)
    error = get_error(data, predictions)

    print("Degree " + str(degree))
    print("\tWeights: " + str(weights))
    print("\tError: " + str(error))
    return predictions

#Make plots of the data and regression model
for num in range(1, 4):
    print("Synthetic Data " + str(num))

    data = np.genfromtxt("./synthetic-" + str(num) + ".csv", delimiter=",")
    data = data[data[:,0].argsort()] #Sort the data points
    degrees = [1, 2, 4, 7]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Synthetic Data " + str(num))
    axs = axs.flatten()
    for i in range(4):
        axs[i].set_title("Degree " + str(degrees[i]))
        axs[i].scatter(data[:, 0], data[:, 1])
        predictions = regression(data, degrees[i])
        axs[i].plot(predictions[:, 0], predictions[:, 1], "r")
    fig.tight_layout()
    fig.savefig("pic" + str(num) + ".png")
