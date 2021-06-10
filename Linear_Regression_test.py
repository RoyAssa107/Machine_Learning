# Imports
import numpy as np
from matplotlib import pyplot as plt
import random

# Constants
N = 2000  # Amount of Vectors (rows)
M = 10  # Amount of features in each vector (columns)
N_TRAIN = 100  # Amount of Vectors (rows)
M_TRAIN = 10  # Amount of features in each vector (columns)
# SIGMA = 1  # Deviation
SIGMA_TEST = 100


# Function that creates new matrix with rows and columns (using uniform distribution)
def CreateUniformMatrix(rows, columns):
    return [[random.uniform(1, 100) for j in range(columns)] for i in range(rows)]


# Function that creates new matrix with rows and columns (using gaussian distribution)
def CreateGaussianMatrix(rows, columns, MU, SIGMA_CHOSEN):
    return [[round(random.gauss(MU, SIGMA_CHOSEN), 2) for j in range(columns)] for i in range(rows)]


# Function that creates new Vector with size column
def CreateVector(rows, columns):
    return [[random.uniform(1, 2)] for i in range(rows)]


# Function that prints given matrix
def PrintMatrix(matrix, name):
    print("The {0} Matrix is:".format(name))
    print(matrix)
    print()


def Calculate_MSE_Per_SIGMA(SIGMA_CHOSEN):
    EPSILON = np.array(CreateGaussianMatrix(N_TRAIN, 1, 0, SIGMA_CHOSEN))
    X_T = X.transpose()
    global Y
    Y = X.dot(BETA)

    # Calculating Y
    for i in range(N_TRAIN):
        Y[i] += EPSILON[i]

    # Now, we will apply the linear regression formula to find the BETA_BAR
    # Which minimizes the square error of the estimation Y
    try:
        X_INV = np.linalg.inv(X_T.dot(X))

    except np.linalg.LinAlgError:
        print("Unable to Inverse Matrix X!!")
    finally:
        BETA_BAR = X_INV.dot(X_T.dot(Y))

    DELTA_BETA = [abs(BETA_BAR[i][0] - BETA[i][0]) for i in range(len(BETA))]

    ##############################################
    #       CALCULATING MSE (IN 2 WAYS)          #
    ##############################################
    # Option1 => Calculate the MSE using BETA_BAR
    Y_BAR = np.array(X.dot(BETA_BAR))  # This is the estimated Y
    DELTA_Y = np.array([abs(Y_BAR[i][0] - Y[i][0]) for i in range(N_TRAIN)])
    MSE = round((1 / N_TRAIN) * sum(pow(DELTA_Y[i], 2) for i in range(N_TRAIN)), 4)

    return MSE


if __name__ == "__main__":
    ####################################
    #       CREATING MATRICES          #
    ####################################
    DATA = np.array(CreateUniformMatrix(N, M))
    X = np.array(DATA)[:N_TRAIN]
    BETA = np.array(CreateVector(M, 1))
    SIGMA = [i/100 for i in range(SIGMA_TEST)]
    global Y
    #######################################
    #       CALCULATING MATRICES          #
    #######################################
    MSE_PER_SIGMA = [Calculate_MSE_Per_SIGMA(SIGMA[i]) for i in range(SIGMA_TEST)]

    ###################################
    #       PLOTTING RESULTS          #
    ###################################
    x_axis = SIGMA
    y_axis = MSE_PER_SIGMA
    plt.scatter(x_axis, y_axis)
    plt.plot(x_axis, y_axis)

    plt.xlabel('Standard deviation')
    plt.ylabel('Error')
    plt.title(' Error with relation to Standard deviation ')
    # Display a figure.
    plt.show()
