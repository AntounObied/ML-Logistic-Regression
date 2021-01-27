import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

"""
This class holds the optimal parameters that will be trained using logistic regression.
"""

class Logistic_Regression:
    __slots__ = "parameters"

    def __init__(self, parameters):
        self.parameters = parameters

    def predict_class(self, features, parameters):
        """

        :param features: The features collected from reading the CSV file
        :param parameters: The optimal parameters from the model training
        :return: A list of predictions based on the provided features and trained parameters, where 0 corresponds to
                    Hylaminuta, and 1 corresponds to HypsiboaScinerascens
        """
        result = sigmoid(features, parameters)

        # Results below 0.5 are rounded down to 0, otherwise rounded up to 1
        return np.round(result)


def generate_data(filename):
    """
    Read in a CSV file using numpy, and create a matrix of attribute values
    :param filename: Name of file to read
    :return: Matrix of attribute values, and the actual class descriptions to be used for training
    """

    # Read file using numpy
    data_file = np.genfromtxt(filename, delimiter=",", encoding="utf-8-sig", dtype=None)

    # Create empty arrays to populate with attribute data and class number
    data = [[], [], []]
    class_list = []

    # Iterate through file data, ignoring the data header
    for i in range(1, len(data_file)):
        data[0].append(float(data_file[i][0]))  # Add MFCC 10 attribute value
        data[1].append(float(data_file[i][1]))  # Add MFCC 17 attribute value
        data[2].append(1)  # Add one to account for bias term

        # Change class identifier from Hylaminuta to 0, and HypsiboaScinerascens to 1
        if data_file[i][2] == "HylaMinuta":
            class_list.append(0)
        else:
            class_list.append(1)

    # Change the lists to numpy arrays for type compatibility, and easier transpose operation
    class_list = np.array(class_list)

    # Transpose attribute data array to have one row per entry
    data = np.array(data).T

    return data, class_list


def sigmoid(x, y):
    """
    Calculates the sigmoid value for logistic regression
    :param x: First matrix for the multiplication operation
    :param y: Second matrix for the multiplication operation
    :return: Values on the sigmoid equation
    """
    return 1 / (1 + np.exp(-(x @ y)))


def gradient_descent(features, class_list, current_params, learning_rate, max_iterations):
    """
    Runs iterations and uses gradient descent search to adjust initial parameters every iteration to find optimal
    parameters
    :param features: Matrix of MFCC 10 and MFCC 17 values for each sample
    :param class_list: List of corresponding class identifiers for each sample
    :param current_params: The initial parameters passed into the function
    :param learning_rate: Learning rate of the logistic regression
    :param max_iterations: Number of iterations to train the program
    :return: Adjusted parameters after training, which should be the estimated optimal parameters
    """

    # Dampening factor
    alpha = learning_rate / len(class_list)

    for ix in range(max_iterations):
        # Dampening the parameter values, with the class list subtracted to adjust for the sigmoid function
        current_params -= alpha * (features.T @ (sigmoid(features, current_params) - class_list))

    return current_params


def calculate_accuracy(predictions, actual_results):
    """
    Calculate the accuracy of the model based on the known class list
    :param predictions: Prediction list estimated by the model.
    :param actual_results: The known class list
    :return: Accuracy of the model
    """
    # Number of correct predictions
    correct = 0

    # Number of wrong predictions
    wrong = 0

    # For every results
    for i in range(len(actual_results)):
        # If the prediction matches the result, increment correct count
        if predictions[i] == actual_results[i]:
            correct += 1
        # Otherwise, increment wrong count
        else:
            wrong += 1

    # The accuracy is the number of correct predictions divided by the total predictions
    accuracy = correct / (correct + wrong) * 100

    return accuracy


def plot_decision_boundary(filename1, filename2, parameters):
    """
    Plot the decision boundary on a scatter plot of the two files, Frogs.csv and Frogs-subsample.csv
    :param filename1: The name of the first file
    :param filename2: The name of the second file
    :param parameters: The estimated optimal parameters calculated by the regression model
    :return:
    """

    # Read the files using numpy
    data_file = np.genfromtxt(filename1, delimiter=",", encoding="utf-8-sig", dtype=None)
    data_file_subsample = np.genfromtxt(filename2, delimiter=",", encoding="utf-8-sig", dtype=None)

    # Separate the classes in the file for differentiation in the scatter plot
    hylaminuta = [[], []]
    hypsiboascinerascens = [[], []]

    # Iterate through the data of the first file, and populate each class list with the corresponding attribute values
    for i in range(1, len(data_file)):
        if data_file[i][2] == "HylaMinuta":
            hylaminuta[0].append(float(data_file[i][0]))
            hylaminuta[1].append(float(data_file[i][1]))
        else:
            hypsiboascinerascens[0].append(float(data_file[i][0]))
            hypsiboascinerascens[1].append(float(data_file[i][1]))

    # Create a subplot of two plots, and plot the data of the first file in a scatter plot
    plt.figure(figsize=(18, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(hylaminuta[0], hylaminuta[1], marker=4, color="green")
    plt.scatter(hypsiboascinerascens[0], hypsiboascinerascens[1], marker=4, color="blue")
    plt.xlabel("MFCC 10")
    plt.ylabel("MFCC 17")
    plt.title("Frogs.csv Scatter Plot")
    plt.legend(["HylaMinuta", "HypsiboasCinerascens"])

    # Get the current axis
    axis = plt.gca()

    # Create the x and y values and equation for the decision boundary
    x_vals = np.array(axis.get_xlim())
    y_vals = (-1 / parameters[0]) * (parameters[1] * x_vals + parameters[2])

    # Plot the decision boundary
    plt.plot(x_vals, y_vals, c="r")

    # Separate the classes in the file for differentiation in the scatter plot
    hylaminuta_subsample = [[], []]
    hypsiboascinerascens_subsample = [[], []]

    # Iterate through the data of the second file, and populate each class list with the corresponding attribute values
    for i in range(1, len(data_file_subsample)):
        if data_file_subsample[i][2] == "HylaMinuta":
            hylaminuta_subsample[0].append(float(data_file_subsample[i][0]))
            hylaminuta_subsample[1].append(float(data_file_subsample[i][1]))
        else:
            hypsiboascinerascens_subsample[0].append(float(data_file_subsample[i][0]))
            hypsiboascinerascens_subsample[1].append(float(data_file_subsample[i][1]))

    # Plot the data of the second file in a scatter plot as the second subplot
    plt.subplot(1, 2, 2)
    plt.scatter(hylaminuta_subsample[0], hylaminuta_subsample[1], marker=4, color="green")
    plt.scatter(hypsiboascinerascens_subsample[0], hypsiboascinerascens_subsample[1], marker=4, color="blue")
    plt.xlabel("MFCC 10")
    plt.ylabel("MFCC 17")
    plt.title("Frogs-subsample.csv Scatter Plot")
    plt.legend(["HylaMinuta", "HypsiboasCinerascens"])

    # Get the current axis
    axis = plt.gca()

    # Create the x and y values and equation for the decision boundary
    x_vals = np.array(axis.get_xlim())
    y_vals = (-1 / parameters[0]) * (parameters[1] * x_vals + parameters[2])

    # Plot the decision boundary
    plt.plot(x_vals, y_vals, c="r")

    # Display the two created plots
    plt.show()


def main():

    # Command is the first command line argument that specifies if the user would like to train or predict
    command = sys.argv[1]

    if command == "train":

        # The file name used to train is the second command line argument
        filename = sys.argv[2]

        # Get matrix of features for each training sample and their corresponding
        features, class_list = generate_data(filename)

        # Specify the learning rate and the number of iterations to use for training
        learning_rate = 0.1
        max_iterations = 100000

        # Random parameters are initialized for training
        initial_params = np.random.rand(3)

        # Perform gradient descent on the parameters to adjust them to the estimated optimal parameters
        optimal_params = gradient_descent(features, class_list, initial_params, learning_rate, max_iterations)

        # Create the model object to store the calculated parameters
        model = Logistic_Regression(optimal_params)

        # The name of the file to save the model is the third command line argument
        model_name = sys.argv[3]

        # Create the file and save the model
        model_save = open(model_name, "wb")
        pickle.dump(model, model_save)

    elif command == "predict":
        # If predict is chosen, the name of the model is the third command line argument
        model_name = sys.argv[2]

        # The file to calculate the predictions for is the third command line argument
        file_to_predict = sys.argv[3]

        # Load the saved model to get the parameters from it
        hypothesis = open(model_name, "rb")
        model = pickle.load(hypothesis)

        # Create the matrix of features to predict the class from them, and the actual results to test the accuracy
        features, class_list = generate_data(file_to_predict)

        # Calculate the list of predictions
        predictions = model.predict_class(features, model.parameters)

        # Calculate the accuracy of the predictions
        accuracy = calculate_accuracy(predictions, class_list)

        # Display accuracy of the predictions
        print("The accuracy of this model was: {}%".format(accuracy))

        # Plot the decision boundary from the model parameters for visual representation
        plot_decision_boundary("Frogs.csv", "Frogs-subsample.csv", model.parameters)

    # If the command is not train or predict, display error message with correct syntax prompt
    else:
        print("Invalid argument provided.")
        print("Syntax for training: train <training_file_name> <model_name_to_save>")
        print("Syntax for predicting: predict <model_name> <file_to_predict>")

if __name__ == "__main__":
    main()
