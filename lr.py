
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
#import plot_db

def scaling(x):
    #mean of featurn column x
    mean = x.mean()
    #std of feature column x
    std = np.std(x)
    return (x - mean)/std

def predict(features_matrix, betas_matrix):
   return np.dot(features_matrix, betas_matrix)

def update_betas(y, x, betas_matrix, alpha):
    n = y.shape[0]
    betas = []
    beta0 = 0
    beta1 = 0
    beta2 = 0

    for i in range(0,n):
        beta0 += (betas_matrix[0] + betas_matrix[1] * x[i, 0] + betas_matrix[2] * x[i, 1] - y[i])
        beta1 += (betas_matrix[0] + betas_matrix[1] * x[i, 0] + betas_matrix[2] * x[i, 1] - y[i])*(x[i, 0])
        beta2 += (betas_matrix[0] + betas_matrix[1] * x[i, 0] + betas_matrix[2] * x[i, 1] - y[i])*(x[i, 1])
    beta0 = betas_matrix[0] - alpha * 1/n * (beta0)
    beta1 = betas_matrix[1] - alpha * 1/n * (beta1)
    beta2 = betas_matrix[2] - alpha * 1/n * (beta2)
    betas.append(beta0)
    betas.append(beta1)
    betas.append(beta2)
    return np.vstack(np.array(betas))

def R(y, fx):
    sum = 0
    n = fx.shape[0]
    for i in range(0,n):
        sum += (fx[i]-y[i])**2
    return sum/(2*n)

def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    data = pd.read_csv(sys.argv[1], header=None)
    features = np.asmatrix(data.iloc[:, :-1])
    heights = np.asmatrix(data.iloc[:, 2:])
    num_of_features = features.shape[1]

    # Data Preparation and Normalization
    for i in range(num_of_features):
        x = features[0:, i]
        x = scaling(x)
        features[0:, i] = x

    r = features.shape[0]
    intercept_column = np.ones([r, 1])
    features_matrix = np.c_[intercept_column, features]
    results = {}

    # Implementing gradient descent
    learning_rates = [0.001, 0.005, 0.01, 0.04, 0.05, 0.1, 0.5, 1, 5, 10]
    iterations = 100
    for alpha in learning_rates:
        R_value = None
        # Initialize betas to zero
        betas = np.vstack(np.zeros(features_matrix.shape[1]))
        loss = []
        # Repeat 100 iterations per alpha value
        for i in range(1,iterations+1):
            #Get predicted value
            fx = predict(features_matrix, betas)
            if R_value is None:
                R_value = R(heights, fx).copy()
            betas = update_betas(heights, features_matrix[0:, 1:3], betas, alpha)
            loss.append(float(R(heights, fx)))

            # Check if R is decreasing after each iteration
            if R_value - R(heights, fx) < 0:
                break

        results[alpha] = [betas, i]  # alpha: [betas, iter_num]
        # Plotting cost function R against number of iterations per alpha value
        # plt.plot([i for i in range(1, iterations+1)], loss)
        # plt.title("Linear Regression")
        # plt.show()

    results2 = ""
    for alpha, item in results.items():
        # α, num_iters, bias, b age, b weight
        num_iter = str(item[1])
        results2 += str(alpha) + ", " + str(num_iter) + ", "
        betas = [str(b[0]) for b in item[0]]
        results2 += ", ".join(betas) + "\n"

    f = open(sys.argv[2], "w")
    f.write(results2)
    f.close()

if __name__ == "__main__":
    main()