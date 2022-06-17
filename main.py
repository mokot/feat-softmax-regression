import gzip
import csv
import warnings
import time
import numpy as np
from scipy.optimize import minimize

# Suppress warnings
warnings.filterwarnings("ignore")

UNDEFINED_CONSTANT = 1.0e-15
NUMBER_OF_CLASSES = None
NUMBER_OF_FEATURES = None


def read_data(filename):
    """
    Reads the data from the given file and returns a tuple of two numpy arrays:
    the first one contains the legend, the second one the data.
    """
    file = gzip.open(filename, "rt", encoding="UTF-8")
    reader = list(csv.reader(file, delimiter=","))

    return np.asanyarray(reader[0]), np.asanyarray(reader[1:])


def write_data(filename, data):
    """
    Writes the data to the given file.
    """
    np.savetxt(filename, data, fmt="%s", delimiter=",")

    return


def one_hot_encoding(data):
    """
    Encode the class labels (y) into a format that we can more easily work with.
    """
    return (np.arange(np.max(data) + 1) == data[:, None]).astype(float)


def one_hot_decoding(data):
    """
    Decode the class labels (y) into a format that we can more easily work with.
    """
    return np.argmax(data, axis=1)


def softmax(parameters, X):
    """
    Softmax function.

    wiki: https://en.wikipedia.org/wiki/Softmax_function
    http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
    """
    z = X.dot(parameters)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T


def cost(parameters, X, y, lambda_):
    """
    Cost function of logistic likelihood.

    wiki: https://en.wikipedia.org/wiki/Likelihood_function
    """
    parameters = parameters.reshape((NUMBER_OF_FEATURES, NUMBER_OF_CLASSES))
    y = one_hot_decoding(y)

    # L2 regularization
    regularization = lambda_ * np.square(np.linalg.norm(parameters))

    predictions = softmax(parameters, X)
    return np.sum(np.log(predictions[(np.arange(len(y)), y)])) - regularization


def grad(parameters, X, y, lambda_):
    """
    Gradient of logistic likelihood.

    wiki: https://en.wikipedia.org/wiki/Likelihood_function
    """
    parameters = parameters.reshape((NUMBER_OF_FEATURES, NUMBER_OF_CLASSES))

    gradient = X.T.dot(y - softmax(parameters, X)) - 2 * lambda_ * parameters
    return gradient.reshape(-1)


def bfgs(X, y, lambda_):
    """
    Implements logistic regression using BFGS algorithm and returns the
    parameters of fitted model.

    wiki: https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    """
    y = one_hot_encoding(y)

    # X: 0 = number of training samples 1 = number of features
    # y: 0 = number of classes 1 = number of training samples
    global NUMBER_OF_FEATURES, NUMBER_OF_CLASSES
    NUMBER_OF_FEATURES = X.shape[1]
    NUMBER_OF_CLASSES = y.shape[1]

    x0 = np.zeros(NUMBER_OF_FEATURES * NUMBER_OF_CLASSES)

    res = minimize(
        lambda pars, X=X, y=y, lambda_=lambda_: -cost(pars, X, y, lambda_),
        x0,
        method="L-BFGS-B",
        jac=lambda pars, X=X, y=y, lambda_=lambda_: -grad(pars, X, y, lambda_),
        tol=0.00001,
    )

    # print("Iterations:", res.njev)
    # print("BFGS:\n", res.x.reshape((NUMBER_OF_FEATURES, NUMBER_OF_CLASSES)))
    return res.x.reshape((NUMBER_OF_FEATURES, NUMBER_OF_CLASSES))


class SoftMaxLearner:
    def __init__(self, lambda_=0, intercept=True):
        self.intercept = intercept
        self.lambda_ = lambda_

    def __call__(self, X, y):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        pars = bfgs(X, y, self.lambda_)
        return SoftMaxClassifier(pars, self.intercept)


class SoftMaxClassifier:
    def __init__(self, parameters, intercept):
        self.parameters = parameters
        self.intercept = intercept

    def __call__(self, X):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        ypred = softmax(self.parameters, X)
        return ypred


def test_learning(learner, X, y):
    """
    Calculates the prediction on the learning data.
    """
    classifier = learner(X, y)
    results = classifier(X)
    return results


def test_cv(learner, X, y, k=5):
    """
    Cross validation.

    wiki: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
    """
    data = np.hstack([X, np.reshape(y, (len(y), 1))])
    number_of_classes = len(set(y))
    number_of_samples = data.shape[0]

    # Create permutation and inverse permutation
    np.random.seed()
    permutation = np.random.permutation(len(data))
    inverse_permutation = np.argsort(permutation)

    # Shuffle the data
    data = data[permutation]

    predictions = []
    num_of_validations = number_of_samples // k
    for i in range(0, number_of_samples, num_of_validations):
        # Split data into training and testing data
        train_data = np.vstack([data[:i], data[i + num_of_validations :]])
        test_data = data[i : i + num_of_validations]

        # Learn model on training data and test it on testing data
        model = learner(train_data[:, :-1], train_data[:, -1])
        temp_predictions = model(test_data[:, :-1])
        number_of_classes = temp_predictions.shape[1]
        for j in range(len(temp_predictions)):
            predictions.append(temp_predictions[j])

    # Unshuffle the data
    predictions = np.reshape(predictions, (number_of_samples, number_of_classes))
    predictions = predictions[inverse_permutation]
    return predictions


def CA(real, predictions):
    """
    Calculates the classification accuracy.
    """
    # Take the class with largest probability as the predicted class
    return np.mean(real == one_hot_decoding(predictions))


def log_loss(real, predictions):
    """
    Binary Cross Entropy / Log Loss.

    https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
    """
    real = one_hot_encoding(real)

    # Log Loss is undefined for p = 0 or p = 1
    predictions[predictions <= 0] = UNDEFINED_CONSTANT
    predictions[predictions >= 1] = 1 - UNDEFINED_CONSTANT

    # return sk_log_loss(real, predictions)
    return -np.sum(real * np.log(predictions)) / len(real)


def find_parameters(X, y, specific=False, save=True):
    """
    Find the best parameters for softmax regression.
    """
    # Find the best regularization parameter
    lambdas1 = np.array(
        [
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
            1000,
            10000,
        ]
    )
    lambdas2 = np.array(
        [
            0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
            0.0007,
            0.0008,
            0.0009,
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
        ]
    )
    lambdas = lambdas1 if not specific else lambdas2

    result = [["lambda", "log_loss", "CA", "intercept"]]
    for temp_lambda in lambdas:
        # Calculate log loss for intercept = true
        softmax_lerner = SoftMaxLearner(lambda_=temp_lambda, intercept=True)
        predictions = test_cv(softmax_lerner, X, y)
        predictions_ll = log_loss(y, predictions)
        predictions_ca = CA(y, predictions)
        result.append([temp_lambda, predictions_ll, predictions_ca, True])

        # Calculate log loss for intercept = false
        softmax_lerner = SoftMaxLearner(lambda_=temp_lambda, intercept=False)
        predictions = test_cv(softmax_lerner, X, y)
        predictions_ll = log_loss(y, predictions)
        predictions_ca = CA(y, predictions)
        result.append([temp_lambda, predictions_ll, predictions_ca, False])

    result = np.asarray(result, dtype=object)
    np.save("parameters.npy", result) if save else print("Parameters:\n", result)
    return result


def create_final_predictions():
    # Read the data
    _, train_data = read_data("train.csv.gz")
    _, test_data = read_data("test.csv.gz")

    X_train = train_data[:, 1:-1].astype(int)  # Remove id and target
    Y_train = np.asarray(
        [prediction.split("_")[-1] for prediction in train_data[:, -1]]
    ).astype(int)
    Y_train -= 0 if len(np.where(Y_train == 0)[0]) else 1  # Convert to 0-based indexing
    test_data = test_data[:, 1:].astype(int)  # Remove id

    # Get lambda for the best classification accuracy
    # parameters = find_parameters(X_train, Y_train)
    # parameters = np.load("parameters.npy", allow_pickle=True)
    # parameters = parameters[1:]
    # Best parameters
    parameters = np.asarray([[0.0002, 0.6425910637484694, 0.76298, True]])
    lambda_, intercept = parameters[np.argmin(parameters[:, 1]), (0, -1)]

    # Calculate log_loss and ca for training data
    softmax_learner = SoftMaxLearner(lambda_=lambda_, intercept=intercept)
    predictions = test_cv(softmax_learner, X_train, Y_train)
    ca_training = CA(Y_train, predictions)
    log_loss_training = log_loss(Y_train, predictions)
    print("Classification accuracy for training data: %.5f" % ca_training)
    print("Log loss for training data: %.5f" % log_loss_training)

    # Start timing build process of the softmax learner
    start_time = time.time()

    # Build the softmax learner and predict the test data
    softmax_learner = SoftMaxLearner(lambda_=lambda_, intercept=intercept)
    softmax_classifier = softmax_learner(X_train, Y_train)
    predictions = softmax_classifier(test_data)

    # End timing
    end_time = time.time()

    # Format predictions back into classes
    predictions_classified = one_hot_decoding(predictions) + (
        1 if len(np.where(Y_train == 0)[0]) else 0
    )  # Convert to 1-based indexing
    result = np.asarray(
        ["Class_%d" % prediction for prediction in predictions_classified]
    )
    write_data("class.txt", result)

    # Create legend for the output file
    legend = ["Class_%d" % class_id for class_id in range(1, predictions.shape[1] + 1)]
    legend.insert(0, "id")

    # Predictions for each class
    predictions_final = np.empty(np.asarray(predictions.shape) + 1, dtype=object)
    predictions_final[:, 0] = np.arange(len(predictions_final))
    predictions_final[0] = legend
    predictions_final[1:, 1:] = predictions
    write_data("final.txt", predictions_final)

    # Time of building the softmax learner
    print(
        "Time of building and predicting with the softmax learner: %.2f seconds"
        % (end_time - start_time)
    )
    return


if __name__ == "__main__":
    create_final_predictions()
