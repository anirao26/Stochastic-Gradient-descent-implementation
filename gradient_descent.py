import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random
plt.style.use('ggplot')

COLUMNS = ["num_preg", "PGC", "DBP", "tricept", "insulin", "BMI", "ped_func", "age"]


def read_data(file_path, file_type):
	if(file_type == "csv"):
		data = pd.read_csv(file_path, encoding="latin-1", header=0)
		return data


def initialize_weight_and_bias(col_num):
	return(np.zeros(col_num))


def predict_label_linear_regression(activation):
	return np.sign(activation)


def implement_sgd_linear_regression(df_train, df_test, learning_rate, maxIterations):
	print("\n")
	W = initialize_weight_and_bias(df_train.shape[1])
	train_data_feature_values = df_train.iloc[:, 1:].values
	train_data_feature_labels = df_train.iloc[:, 0].values
	train_data_feature_values = train_data_feature_values.astype(float)
	train_data_feature_labels = train_data_feature_labels.astype(float)
	train_data_indexes = df_train.index
	squared_error = 0
	average_squared_error_list = []
	weight_norm_list = []
	test_set_accuracy_list = []

	for i in range(1, (maxIterations+1)):
		example_index = random.choice(train_data_indexes)
		example_features = train_data_feature_values[example_index]
		example_features = np.append(example_features, 1.0) #Adding the bias as a constant
		example_label = train_data_feature_labels[example_index]
		with np.errstate(all='raise'):
			try:
				gradient = (-2)*example_features*(example_label-np.dot(W.T, example_features))
				W = W - learning_rate*gradient
				activation = np.dot(W.T, example_features)
				squared_error += np.square(example_label - activation)	
				if(i%50==0):
					average_squared_error_list.append((i,squared_error/i))
					test_set_accuracy_list.append((i, get_test_data_accuracy_rate_linear_regression(df_test, W)))
			except FloatingPointError:
				break

	print("Test data accuracy for stochastic gradient descent for learning rate: ", learning_rate, ": ", test_set_accuracy_list[-1][1], "\n")
	return W, average_squared_error_list, test_set_accuracy_list


def implement_gradient_descent_linear_regression(df_train, df_test, learning_rate, maxIterations):
	print("Learning Rate: ", learning_rate)
	W = initialize_weight_and_bias(df_train.shape[1])
	dev_data = df_train.sample(frac=0.1).reset_index(drop=True)
	train_data = df_train.drop(dev_data.index).reset_index(drop=True)
	train_data_feature_values = train_data.iloc[:, 1:].values
	train_data_feature_labels = train_data.iloc[:, 0].values
	train_data_feature_values = train_data_feature_values.astype(float)
	train_data_feature_labels = train_data_feature_labels.astype(float)
	train_data_indexes = train_data.index
	squared_error = 0
	average_squared_error_list = []
	weight_norm_list = []
	test_set_accuracy_list = []
	gradient = initialize_weight_and_bias(df_train.shape[1])	

	for i in range(1, (maxIterations+1)):		
		with np.errstate(all='raise'):
			try:
				for j in range(len(train_data_feature_values)):
					features = np.append(train_data_feature_values[j], 1.0)
					activation = np.dot(W.T, features)
					true_label = train_data_feature_labels[j]
					gradient += (-2)*features*(true_label-activation)
					squared_error += np.square(true_label - activation)
				W = W - learning_rate*gradient/len(features)
				if(i%50==0):
					average_squared_error_list.append((i, squared_error/((j+1)*i)))
					test_set_accuracy_list.append((i, get_test_data_accuracy_rate_linear_regression(df_test, W)))
			except FloatingPointError:
				break
	
	dev_data_accuracy_final = get_test_data_accuracy_rate_linear_regression(dev_data, W)
	print("Dev data accuracy for learning rate: ", learning_rate, ": ", dev_data_accuracy_final, "\n")
	return W, dev_data_accuracy_final, average_squared_error_list, test_set_accuracy_list


def get_test_data_accuracy_rate_linear_regression(test_data, W):
	error_count = 0
	test_data_feature_values = test_data.iloc[:, 1:].values
	test_data_feature_labels = test_data.iloc[:, 0].values
	test_data_feature_values = test_data_feature_values.astype(float)
	test_data_feature_labels = test_data_feature_labels.astype(float)
	for i in range(len(test_data_feature_values)):
		features = np.append(test_data_feature_values[i], 1.0) #Adding the bias as a constant
		activation = np.dot(W.T, features)
		y_hat = predict_label_linear_regression(activation)
		if test_data_feature_labels[i]!=y_hat:
			error_count += 1

	error_rate = float(error_count/len(test_data_feature_values))
	return(1.0-error_rate)


def plot_avg_square_error(average_squared_error_list, average_squared_error_list_sgd, learning_rate, regression_type, maxIteration):
	iteration, avg_squared_errs = zip(*average_squared_error_list)
	iteration, avg_squared_errs_sgd = zip(*average_squared_error_list_sgd)

	plt.figure(figsize = (14,6))
	plt.plot(iteration, avg_squared_errs, color = "blue", label='Gradient descent')
	plt.plot(iteration, avg_squared_errs_sgd, color = "red", label='Stochastic Gradient descent')
	plt.xlabel("Iteration")
	plt.ylabel("Average Squared Error")
	plt.legend()
	plt_name_avg_sqrd_error = "gradient_descent_avg_sqr_err_linear_"+str(learning_rate)+"_"+str(maxIteration)+".png"
	title = "Linear regression avg squared error plot for learning rate: "+str(learning_rate)

	plt.title(title)
	plt.savefig(plt_name_avg_sqrd_error)


def plot_test_set_accuracy_across_iterations(test_set_accuracy_list_gd, test_set_accuracy_list_sgd, learning_rate, regression_type, maxIteration):
	iteration, test_set_accuracy_values = zip(*test_set_accuracy_list_gd)
	iteration, test_set_accuracy_values_sgd = zip(*test_set_accuracy_list_sgd)

	plt.figure(figsize = (14,6))
	plt.plot(iteration, test_set_accuracy_values, color = "blue", label='Gradient descent')
	plt.plot(iteration, test_set_accuracy_values_sgd, color = "red", label='Stochastic Gradient descent')
	plt.xlabel("Iteration")
	plt.ylabel("Test set accuracy")
	plt.legend()
	plt_name_test_accuracy = "test_set_accuracy_linear_"+str(learning_rate)+"_"+str(maxIteration)+".png"
	title = "Linear regression plot for test set accuracy for learning rate: "+str(learning_rate)

	plt.title(title)
	plt.savefig(plt_name_test_accuracy)


def main():
	train_data = read_data('A3.train.csv', "csv")
	test_data = read_data('A3.test.csv', "csv")

	weight_vector_index_bmi = COLUMNS.index("BMI")
	weight_vector_index_insulin = COLUMNS.index("insulin")
	weight_vector_index_pgc = COLUMNS.index("PGC")

	learning_rate_list =  [0.05, 0.005, 0.01, 0.001, 0.0001]
	maxIterations = 10000

	dev_data_accuracy_list_linear_regression = []
	for i in learning_rate_list:
		dev_data_accuracy_list_linear_regression.append((i, implement_gradient_descent_linear_regression(train_data, test_data, i, maxIterations)))

	learning_rate_linear, final_weights_average_squared_error_test_accuracy_list = zip(*dev_data_accuracy_list_linear_regression)
	final_weights_linear, dev_data_accuracy_values_linear, average_squared_errors, test_accuracy_list = zip(*final_weights_average_squared_error_test_accuracy_list)

	max_accuracy_index_linear = dev_data_accuracy_values_linear.index(max(dev_data_accuracy_values_linear))
	print("The linear regression model performs the best for a learning rate of ", learning_rate_linear[max_accuracy_index_linear])
	best_weight_linear = final_weights_linear[max_accuracy_index_linear]
	
	print("The accuracy on the test data for this model comes to ", get_test_data_accuracy_rate_linear_regression(test_data, best_weight_linear))
	print("\nComparing model performance to SGD for the learning rate of: ", learning_rate_linear[max_accuracy_index_linear])

	final_weights_linear_sgd, average_squared_errors_sgd, test_accuracy_list_sgd = implement_sgd_linear_regression(train_data, test_data, learning_rate_linear[max_accuracy_index_linear], maxIterations)

	plot_avg_square_error(average_squared_errors[0], average_squared_errors_sgd, learning_rate_linear[max_accuracy_index_linear], "linear", maxIterations)
	plot_test_set_accuracy_across_iterations(test_accuracy_list[max_accuracy_index_linear], test_accuracy_list_sgd, learning_rate_linear[max_accuracy_index_linear], "linear", maxIterations)


if __name__ == '__main__':
	main()