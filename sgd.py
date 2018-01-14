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


def sigmoid(x):
	return (1.0 / (1.0 + np.exp(-x)))


def predict_label_logistic_regression(label, activation):
	sigmoid_value = sigmoid(activation)
	if (sigmoid_value)>0.5:
		return 1
	else:
		return -1


def predict_label_linear_regression(activation):
	return np.sign(activation)


def implement_sgd_logistic_regression(df_train, df_test, learning_rate, maxIterations):
	print("Learning Rate: ", learning_rate)
	W = initialize_weight_and_bias(df_train.shape[1])
	dev_data = df_train.sample(frac=0.1)
	train_data = df_train.drop(dev_data.index)
	dev_data = dev_data.reset_index(drop=True)
	train_data = train_data.reset_index(drop=True)
	df_test = df_test.sample(frac=1)

	train_data_feature_values = train_data.iloc[:, 1:].values
	train_data_feature_labels = train_data.iloc[:, 0].values
	train_data_feature_values = train_data_feature_values.astype(float)
	train_data_feature_labels = train_data_feature_labels.astype(float)
	train_data_indexes = train_data.index
	
	squared_error = 0
	average_squared_error_list = []
	weight_norm_list = []
	test_set_accuracy_list = []

	for i in range(1, (maxIterations+1)):
		example_index = random.choice(train_data_indexes)
		example_features = train_data_feature_values[example_index]
		example_features = np.append(example_features, 1.0) #Adding the bias as a constant
		example_label = train_data_feature_labels[example_index]
		activation = np.dot(W.T, example_features)
		y_hat = predict_label_logistic_regression(example_label, activation)
		squared_error += np.square(example_label - y_hat)
		gradient = -1*example_label*example_features/(1+np.exp(example_label*activation))
		W = W - learning_rate*gradient
		
		if(i%100==0):
			average_squared_error_list.append((i, squared_error/i))
			weight_norm_list.append((i, np.linalg.norm(W[:-1])))
			test_set_accuracy_list.append((i, get_test_data_accuracy_rate_logistic_regression(df_test, W)))

	plot_avg_square_error(average_squared_error_list, learning_rate, "logistic")
	plot_weight_norms(weight_norm_list, learning_rate, "logistic")
	plot_test_set_accuracy_across_iterations(test_set_accuracy_list, learning_rate, "logistic")
	# print("Final Weight vector")

	dev_data_accuracy_final = get_test_data_accuracy_rate_logistic_regression(dev_data, W)
	print("Dev data accuracy for learning rate: ", learning_rate, ": ", dev_data_accuracy_final, "\n")
	return W, dev_data_accuracy_final


def implement_sgd_linear_regression(df_train, df_test, learning_rate, maxIterations):
	print("Learning Rate: ", learning_rate)
	W = initialize_weight_and_bias(df_train.shape[1])
	dev_data = df_train.sample(frac=0.1)
	train_data = df_train.drop(dev_data.index)
	dev_data = dev_data.reset_index(drop=True)
	train_data = train_data.reset_index(drop=True)

	train_data_feature_values = train_data.iloc[:, 1:].values
	train_data_feature_labels = train_data.iloc[:, 0].values
	train_data_feature_values = train_data_feature_values.astype(float)
	train_data_feature_labels = train_data_feature_labels.astype(float)
	train_data_indexes = train_data.index
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
				if(i%100==0):
					average_squared_error_list.append((i, squared_error/i))
					weight_norm_list.append((i, np.linalg.norm(W[:-1])))
					test_set_accuracy_list.append((i, get_test_data_accuracy_rate_linear_regression(df_test, W)))
			except FloatingPointError:
				break

	plot_avg_square_error(average_squared_error_list, learning_rate, "linear")
	plot_weight_norms(weight_norm_list, learning_rate, "linear")
	plot_test_set_accuracy_across_iterations(test_set_accuracy_list, learning_rate, "linear")

	dev_data_accuracy_final = get_test_data_accuracy_rate_linear_regression(dev_data, W)
	print("Dev data accuracy for learning rate: ", learning_rate, ": ", dev_data_accuracy_final, "\n")

	return W, dev_data_accuracy_final


def get_test_data_accuracy_rate_logistic_regression(test_data, W):
	error_count = 0
	test_data_feature_values = test_data.iloc[:, 1:].values
	test_data_feature_labels = test_data.iloc[:, 0].values
	test_data_feature_values = test_data_feature_values.astype(float)
	test_data_feature_labels = test_data_feature_labels.astype(float)
	for i in range(len(test_data_feature_values)):
		features = np.append(test_data_feature_values[i], 1.0) #Adding the bias as a constant
		activation = np.dot(W.T, features)
		y_hat = predict_label_logistic_regression(test_data_feature_labels[i], activation)
		if test_data_feature_labels[i]!=y_hat:
			error_count += 1

	error_rate = float(error_count/len(test_data_feature_values))
	return(1.0-error_rate)


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


def plot_avg_square_error(average_squared_error_list, learning_rate, regression_type):
	iteration, avg_squared_errs = zip(*average_squared_error_list)
	plt.figure(figsize = (14,6))
	plt.plot(iteration, avg_squared_errs, color = "blue")
	plt.xlabel("Iteration")
	plt.ylabel("Average Squared Error")
	
	if(regression_type=="linear"):
		plt_name_avg_sqrd_error = "Avg_sqr_err_linear_"+str(learning_rate)+".png"
		title = "Linear regression avg squared error plot for learning rate: "+str(learning_rate)
	else:
		plt_name_avg_sqrd_error = "Avg_sqr_err_logistic_"+str(learning_rate)+".png"
		title = "Logistic regression avg squared error plot for learning rate: "+str(learning_rate)

	plt.title(title)
	plt.savefig(plt_name_avg_sqrd_error)


def plot_weight_norms(weight_norm_list, learning_rate, regression_type):
	iteration, weight_norm = zip(*weight_norm_list)
	plt.figure(figsize = (14,6))
	plt.plot(iteration, weight_norm, color = "blue")
	plt.xlabel("Iteration")
	plt.ylabel("L2 norm of the weights")
	
	if(regression_type=="linear"):
		plt_name_weight_norm = "weight_norm_linear_"+str(learning_rate)+".png"
		title = "Linear regression plot for weight norms for learning rate: "+str(learning_rate)
	else:
		plt_name_weight_norm = "weight_norm_logistic_"+str(learning_rate)+".png"
		title = "Logistic regression plot for weight norms for learning rate: "+str(learning_rate)

	plt.title(title)
	plt.savefig(plt_name_weight_norm)


def plot_test_set_accuracy_across_iterations(test_set_accuracy_list, learning_rate, regression_type):
	iteration, test_set_accuracy_values = zip(*test_set_accuracy_list)
	plt.figure(figsize = (14,6))
	plt.plot(iteration, test_set_accuracy_values, color = "blue")
	plt.xlabel("Iteration")
	plt.ylabel("Test set accuracy")
	
	if(regression_type=="linear"):
		plt_name_test_accuracy = "test_set_accuracy_linear_"+str(learning_rate)+".png"
		title = "Linear regression plot for test set accuracy for learning rate: "+str(learning_rate)
	else:
		plt_name_test_accuracy = "test_set_accuracy_logistic_"+str(learning_rate)+".png"
		title = "Logistic regression plot for test set accuracy for learning rate: "+str(learning_rate)

	plt.title(title)
	plt.savefig(plt_name_test_accuracy)


def main():
	train_data = read_data('A3.train.csv', "csv")
	test_data = read_data('A3.test.csv', "csv")

	weight_vector_index_bmi = COLUMNS.index("BMI")
	weight_vector_index_insulin = COLUMNS.index("insulin")
	weight_vector_index_pgc = COLUMNS.index("PGC")

	print("Logistic Regression:\n")
	dev_data_accuracy_list_logistic_regression = []
	learning_rate_list = [0.8, 0.001, 0.00001]
	maxIterations = 100000

	for i in learning_rate_list:
		dev_data_accuracy_list_logistic_regression.append((i, implement_sgd_logistic_regression(train_data, test_data, i, maxIterations)))	
	
	learning_rate_logistic, final_weights_accuracy_values_logistic = zip(*dev_data_accuracy_list_logistic_regression)
	final_weights_logistic, dev_data_accuracy_values_logistic = zip(*final_weights_accuracy_values_logistic)

	max_accuracy_index_logistic = dev_data_accuracy_values_logistic.index(max(dev_data_accuracy_values_logistic))
	print("\nThe logistic regression model performs the best for a learning rate of ", learning_rate_logistic[max_accuracy_index_logistic])
	best_weight_logistic = final_weights_logistic[max_accuracy_index_logistic]
	print("\nWeight vector logistic regression_type--->", best_weight_logistic)
	print("\nThe accuracy on the test data for this model comes to ", get_test_data_accuracy_rate_logistic_regression(test_data, best_weight_logistic), "\n")
	print('Weight vector components for \nBMI : {} \n2-hour serum insulin level : {}\nplasma glucose concentration : {} \nBias : {}\n'.format(best_weight_logistic[weight_vector_index_bmi], best_weight_logistic[weight_vector_index_insulin], best_weight_logistic[weight_vector_index_pgc], best_weight_logistic[-1]))

	print("\nLinear Regression:\n")
	dev_data_accuracy_list_linear_regression = []

	for i in learning_rate_list:
		dev_data_accuracy_list_linear_regression.append((i, implement_sgd_linear_regression(train_data, test_data, i, maxIterations)))	

	learning_rate_linear, final_weights_accuracy_values_linear = zip(*dev_data_accuracy_list_linear_regression)
	final_weights_linear, dev_data_accuracy_values_linear = zip(*final_weights_accuracy_values_linear)

	max_accuracy_index_linear = dev_data_accuracy_values_linear.index(max(dev_data_accuracy_values_linear))
	print("The linear regression model performs the best for a learning rate of ", learning_rate_linear[max_accuracy_index_linear], "\n")
	best_weight_linear = final_weights_linear[max_accuracy_index_linear]
	print("Weight vector for linear regression ---> ", best_weight_linear)
	print("The accuracy on the test data for this model comes to ", get_test_data_accuracy_rate_linear_regression(test_data, best_weight_linear), "\n")
	print('Weight vector components for\nBMI : {}\n2-hour serum insulin level : {}\nplasma glucose concentration : {}\nBias : {}\n'.format(best_weight_linear[weight_vector_index_bmi], best_weight_linear[weight_vector_index_insulin], best_weight_linear[weight_vector_index_pgc], best_weight_linear[-1]))


if __name__ == '__main__':
	main()
