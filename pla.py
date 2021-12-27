import pandas as pd
import numpy as np
import plot_db
import matplotlib.pyplot as plt
import sys


def main():

    data = pd.read_csv("hw.csv",  header=None)
    features = np.asmatrix(data.iloc[:, :-1])
    labels = np.asmatrix(data.iloc[:, 2:])
    num_of_features = features.shape[1]
    w = [0 for i in range(num_of_features+1)]
    #positive data points
    x1 = features[:4, 0:1].flatten()
    y1 = features[:4, 1:2].flatten()
    print(x1)
    print(y1)
    plt.scatter([x1], [y1], c='green')
    #negative data points
    x2 = features[4:, 0:1].flatten()
    y2 = features[4:, 1:2].flatten()
    print(x2)
    print(y2)
    plt.scatter([x2], [y2], c='red')
    plt.show()
    exit()
    convergence = False
    weights_vector = []
    #print(features)
    while not convergence:
        correct_prediction = 1

        for x, label in zip(features, labels):

            #print("x = ", x)

            y = int(label[0]) #true value
            #print("y = ", y)
            x = np.insert(arr=x, obj=2, values=1) #to include the bias into the summation

            activation = np.dot(w, x.transpose())
            #print("activation", activation.tolist()[0][0])
            if activation.tolist()[0][0] <= 0:
                #print("activation less than 0")
                fx = -1
            else:
                #print("activation > 0")
                fx = 1 #predicted_value
            #print("y*fx = ", y*fx)
            #check for error
            if (y*fx) <= 0:

                #update all weights
                w[2] = w[2] + y #updating the bias
                #print("w[2] = ", w[2])
                #print("num feat = ", num_of_features)
                #print("x = ", x)

                for i in range(num_of_features):
                    w[i] = w[i] + (y*x[0:1, 0+i:1+i][0,0])
                    #print("w[", i, "] = ", w[i])
                wi = w.copy()
                #print("wi = ", wi)
                #exit()
                weights_vector.append(wi)
            else:
                correct_prediction += 1
        #print("current prediction = ", correct_prediction)
        #print("len(labels) = ", len(labels))
        #print(weights_vector)

        #check for convergence
        if correct_prediction == len(labels):
                convergence = True
                break

    weights_output = ""
    for w in weights_vector:
        value = [str(i) for i in w]
        weights_output += ",".join(value) + "\n"

    print(weights_output)
    exit()
    f = open("2b.csv", "w")
    f.write(weights_output)
    f.close()


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()