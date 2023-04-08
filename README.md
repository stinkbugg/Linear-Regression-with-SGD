# Linear-Regression-with-SGD
Defining functions to implement three functions to apply to weather data.
In this assignment, you will implement three functions train, use, and rmse and apply them to some weather data. Here are the specifications for these functions, which you must satisfy.

model = train(X, T, learning_rate, n_epochs, verbose)

X: is an 𝑁
 x 𝐷
 matrix of input data samples, one per row. 𝑁
 is the number of samples and 𝐷
 is the number of variable values in each sample.
T: is an 𝑁
 x 𝐾
 matrix of desired target values for each sample. 𝐾
 is the number of output values you want to predict for each sample.
learning_rate: is a scalar that controls the step size of each update to the weight values.
n_epochs: is the number of epochs, or passes, through all 𝑁
 samples, to take while updating the weight values.
verbose: is True or False (default value) to control whether or not occasional text is printed to show the training progress.
model: is the returned value, which must be a dictionary with the keys 'w', 'Xmeans', 'Xstds', 'Tmeans' and 'Tstds'.
Y = use(X, model)

X: is an 𝑁
 x 𝐷
 matrix of input data samples, one per row, for which you want to predict the target values.
model: is the dictionary returned by train.
Y: is the returned 𝑁
 x 𝐾
 matrix of predicted values, one for each sample in X.
result = rmse(Y, T)

Y: is an 𝑁
 x 𝐾
 matrix of predictions produced by use.
T: is the 𝑁
 x 𝐾
 matrix of target values.
result: is a scalar calculated as the square root of the mean of the squared differences between each sample (row) in Y and T.
