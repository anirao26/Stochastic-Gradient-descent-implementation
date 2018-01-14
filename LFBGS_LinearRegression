import numpy as	np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def squared_error_linear_regression(x, *args):
	squared_error = 0
	feature_list = args[0]
	label_list = args[1]
	for j in range(len(feature_list)):
		features = np.append(feature_list[j], 1.0)
		activation = np.dot(x.T, features)
		true_label = label_list[j]
		squared_error += np.square(true_label - activation)
	return squared_error


def gradient_linear_regression(x, *args):
	feature_list = args[0]
	label_list = args[1]
	gradient = np.zeros(train_data.shape[1])
	for j in range(len(feature_list)):
		features = np.append(feature_list[j], 1.0)
		activation = np.dot(x.T, features)
		true_label = label_list[j]
		gradient += (-2)*features*(true_label-activation)
	return gradient


def get_test_data_accuracy_rate_linear_regression(test_data, W):
	error_count = 0
	test_data_feature_values = test_data.iloc[:, 1:].values
	test_data_feature_labels = test_data.iloc[:, 0].values
	test_data_feature_values = test_data_feature_values.astype(float)
	test_data_feature_labels = test_data_feature_labels.astype(float)
	for i in range(len(test_data_feature_values)):
		features = np.append(test_data_feature_values[i], 1.0) #Adding the bias as a constant
		activation = np.dot(W.T, features)
		y_hat = np.sign(activation)
		if test_data_feature_labels[i]!=y_hat:
			error_count += 1

	error_rate = float(error_count/len(test_data_feature_values))
	return(1.0-error_rate)


def plot_test_data_accuracy(x_test,y_test,w,b):
    error_count=0
    test_iterations=[]
    test_accuracy=[]
    size=1
    y_hat = np.sign(w.dot(x_test.T) + b)
    for i in range(len(y_test)):
        if y_test[i] != y_hat[i]:
            error_count += 1
        test_accuracy.append(1 - (error_count/float(size)))
        test_iterations.append(i)
        size += 1
    plt.figure(figsize=(15,5))
    plt.style.use('ggplot')
    plt.title("Accuracy as a function of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Test set accuracy")
    plt.plot(test_iterations,test_accuracy)
    plt.savefig("bfgs_plot.png")


train_data = pd.read_csv('A3.train.csv')
train_data = train_data.sample(frac = 1)

train_data_feature_values = train_data.iloc[:, 1:].values
train_data_feature_labels = train_data.iloc[:, 0].values

test_data = pd.read_csv('A3.test.csv')

W = np.zeros(train_data.shape[1])

args = (train_data_feature_values, train_data_feature_labels)

res = fmin_l_bfgs_b(squared_error_linear_regression, W, fprime=gradient_linear_regression, args = args, epsilon = 0.005)

print("\nThe weight vector obtained is: ", res[0])
print("\nThe number of iterations it took to converge is: ",res[2]['nit'])
print("\nGetting test set accuracy:\n")
print("The test set accuracy comes to be: ", get_test_data_accuracy_rate_linear_regression(test_data, res[0]))

test_data_feature_values = test_data.iloc[:, 1:].values
test_data_feature_labels = test_data.iloc[:, 0].values
test_data_feature_values = test_data_feature_values.astype(float)
test_data_feature_labels = test_data_feature_labels.astype(float)

plot_test_data_accuracy(test_data_feature_values, test_data_feature_labels, res[0][:-1], res[0][-1])
