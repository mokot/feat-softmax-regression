# Softmax Regression

Softmax regression (or multinomial logistic regression) is a generalization of 
logistic regression to the case where we want to handle multiple classes.

The multinomial logistic regression technique was used to predict the feat rate. 
The best result was achieved using L2 regularization (regularization rate was 
*0.02*) and a learning rate equal to *0.00001*. The classification accuracy of 
the given data was 0.76 and the log loss was 0.63.

The final predictions of the test data model, which have been fully learned from 
the learning data, are available in the [final.txt](./final.txt) and 
[class.txt](./class.txt) files, with the first file being the probability of 
each class occurring and the second the final predictions of the model.

You can read more about **logistic regression** [here](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/) 
and about **softmax regression** [here](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/).