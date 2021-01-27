# Linear Regression

## Introduction

Linear regression can help answer questions such as "How can x be used to predict y". Where x is information and y is information we want to know. 
Simplest Example is finding the price of your house. No. of bedrooms 2 (this is x) and know how much the estate is worth on the market (this is y)

Linear regression creates an equation in which we input the given number (x) and outputs target variable that you want to find (y). We can obtain the equation by training it on pairs of (x,y) values. A dataset can be used containing historic records of house purchases in the form of ("Number of Bedrooms", "Selling Price")

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Dataset_1.png"> 
</p>

We can visualize the data points on a scatter plot to see if there are any new trends. A Scatter plot is 2D plot with each point representating a house. 

On the x-axis "Number of bedrooms" and on the y-axis "Selling Price" for the same houses

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Dataset1_Scatter.png"> 
</p>

We can see that there is a trend in the image above, more bedrooms result in a higher selling price. Considering a linear regression model is trained to get an equation of form:

<p align="center">
  Selling Price = $77,143 * (Number Of Bedrooms) - $74,286 = $80,000
</p>
 
We can also visualize graphiically what woud be the price for houses with different number of bedrooms 

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Dataset1_Linear.png"> 
</p>

## Linear Regression Model Representation withh Regression Equation

Once a linear regression mode is trainned the model forms a linear regression equation of the type

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Linear_Regression_Equation.png"> 
</p>

In the above equation:

- y is the <<b> output variable </b>. It is also called the <b> target variable </b> in machine learning or the <b> dependednt variable </b> in statistical modeling. It represents the continous value that we are trying to predict.

- x is the <b> input variable </b>. In machine learning referred to as <b> feature variable </b> or the <b> independent variable </b> in statistical modeling. Represents the information given to us at any time.

- w0 is the <b> bias term </b> or <b> y-axis intercept </b>

- w1 is the regression coefficient or scale factor. In classical statistics, it is thhe equivalent of the slope on the best-fit straight line that is produced after the model has been fitted.

- wi are called <i> weights </i> in general.

The goal of linear regrerssion can be defined as finding the unknown parameters of the equation; that is to find the values for the weights w0 and w1. 

## Simple Linear Regression vs. Multiple Linear Regression 

Both simple and multiple linear regressions assume that there is a linear relationship between the input variable(s) and the output target variable.

The main difference is the number of independent variables that they take as inputs. Simple linear regression just takes a single feature, while multiple linear regression takes multiple x values. The above formula can be rewritten for a model with n-input variables as:

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Multiple_Linear_Regression_Equation"> 
</p>
 
Despite their differences, both the simple and multiple regression models are linear models - they adopt the form of a linear equation. This is called the linear assumption. Quite simply, it means that we assume that the type of relationship between the set of independent variables and independent features is linear.

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Simple_Multiple_Diagram.png"> 
</p>
 
 ## Training A Lnear Regression Model
 
 We train the linear regression algorithm with a method named Ordinary Least Squares (or just Least Squares). The goal of training is to find the weights wi in the linear equation y = wo + w1x.

The Ordinary Least Squares procedure has four main steps in machine learning:

1. . Random weight initialization. In practice, w0 and w1 are unknown at the beginning. The goal of the procedure is to find the appropriate values for these model parameters. To start the process, we set the values of the weights at random.

2. Input the initialized weights into the linear equation and generate a prediction for each observation point. To continue with the example that we’ve already used:

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Residuals_Dataset.png"> 
</p>

3. Calculate the Residual Sum of Squares (RSS).

a) Residuals, or error terms, are the difference between each actual output and the  predicted output. They are a point-by-point estimate of how well our regression function predicts outputs in comparison to true values. We obtain residuals by calculating *actual values - predicted values* for each observation.

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Residuals_Dataset_1.png"> 
</p>

b) We square the residuals (in other words, we compute residual2 for each observation point).

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Residuals_Dataset_Squared.png"> 
</p>

c) We sum the residuals to reach our RSS: 1,600,000,000 + 293,882,449 + 2,946,969,796 + 987,719,184 = 5,828,571,429.

d) The basis here is that a lower RSS means that our line of best fit comes closer to each data point. The further away the trend line is from actual observations, the higher the RSS. So, the closer the actual values are (blue points) to the regression line (red line), the better (the green lines representing residuals will be shorter).

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Residuals_Dataset_Plot.png"> 
</p>
    
4. Model parameter selection to minimize RSS. Machine learning approaches find the best parameters for the linear model by defining a cost function and minimizing it via gradient descent. By doing so, we obtain the best possible values for the weights.

## Cost Function

The cost function is a formal representation of an objective that the algorithm is trying to achieve. 
In the case of linear regression, the cost function is the same as the residual sum of errors. The algorithm solves the minimization problem it tries to minimize the cost function in order to achieve the best fitting line with the lowest residual errors.

This is achieved through gradient descent.

## Gradient Descent

Gradient descent is a method of changing weights based on the loss function for each data point. We calculate the sum of squared errors at each input-output data point.

We take a partial derivative of the weight and bias to get the slope of the cost function at each point. (No need to brush up on linear algebra and calculus right now. 

Based on the slope, gradient descent updates the values for the set of weights and the bias and re-iterates the training loop over new values (moving a step closer to the desired goal).

This iterative approach is repeated until a minimum error is reached, and gradient descent cannot minimize the cost function any further.

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/gradient_Descent.jpg"> 
</p>
    
The results are optimal weights for the problem at hand. There is, however, one consideration to bear in mind when using gradient descent: the hyperparameter learning rate. The learning rate refers to how much the parameters are changed at each iteration. If the learning rate is too high, the model fails to converge and jumps from good to bad cost optimizations. If the learning rate is too low, the model will take too long to converge to the minimum error.

<p align="center" width="100%">
    <img src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/gradient_descent_2.png"> 
</p>
    
## Model Evaluation

How do we evaluate the accuracy of our model?

First of  all, you need to make sure that you train the model on the training dataset and build evaluation metrics on the test set to avoid overfitting. Afterward, you can check several evaluation metrics to determine how well your model performed.

There are various metrics to evaluate the goodness of fit:

1. **Mean Squared Error (MSE)**. MSE is computed as RSS divided by the total number of data points, i.e. the total number of observations or examples in our given dataset. MSE tells us what the average RSS is per data point.

2. **Root Mean Squared Error (RMSE)**. RMSE takes the MSE value and applies a square root over it. It is similar to MSE, but much more intuitive for error interpretation. It is equivalent to the absolute error between our linear regression line and any hypothetical observation point. Unlike MSE and RSS (which use
squared values), RMSE can be directly used to interpret the ‘average error’ that our prediction model makes.

3. **R2 or R-squared or R2 score**. R-squared is a measure of how much variance in the dependent variable that our linear function accounts for. This measure is more technical than the other two, so it’s less intuitive for a non-statistician. As a rule of thumb, an R-squared value that is closer to 1 is better, because it accounts for more variance.

Once we have trained and evaluated our model, we improve it to make more accurate predictions.

## Model Improvement

There are multiple methods to improve your linear regression model.

### Data Preprocessing

The biggest improvement in your modeling will result from properly cleaning your data. Linear regression has several assumptions about the structure of underlying data, which, when violated, skews or even impedes the model from making accurate predictions. Make sure to:

1. **Remove outliers**. Outliers in the quantitative response y skew the slope of the line disproportionately. Remove them to have a better-fitted line.

2. **Remove multicollinearity**. Linear regression assumes that there is little or no correlation between the input values - otherwise, it overfits the data. Create a
correlation matrix for all of your features to check which pairs of features suffer from high correlation. Remove these features to keep just one.

3. **Assert normal distribution**. The model assumes that the independent variables follow a Gaussian distribution. Transform your variables with log transform or BoxCox if they are not normally distributed.

4. **Assert linear assumption**. If your independent variables do not have a linear relationship with your predictor variable, log transform them to reshape polynomial relationships into linear.

### Feature scaling

Features can come in different orders of magnitude. Using our example of the housing price prediction, the number of bedrooms would be on a scale from 1 - 10 (approximately), while the housing area in square feet would be 100-1000x bigger (1000-10,000 square feet).

Features of different scales convert slower (or not at all) with gradient descent.

Normalize and standardize your features to speed up and improve model training.

### Regularization

Regularization is not useful for the simple regression problem with one input variable. Instead, it is commonly used in multiple regression settings to lower the complexity of the model. The complexity relates to the number of coefficients or weights (or features) that a model uses for its predictions.

Regularization can be thought of as a feature selection method, whereby features with lower contributions to the goodness of fit are removed and/or diminished in their effects, while the important features are emphasized.

There are two regularization techniques that are frequently used in linear regression settings:

1. Lasso L1 Regression - uses a penalty term to remove predictor variables, which have low contributions to overall model performance
2. Ridge L2 Regression - uses a penalty term to lower the influence of predictor variables (but does not remove features)
