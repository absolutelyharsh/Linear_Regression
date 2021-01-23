# Linear Regression

## Introduction

Linear regression can help answer questions such as "How can x be used to predict y". Where x is information and y is information we want to know. 
Simplest Example is finding the price of your house. No. of bedrooms 2 (this is x) and know how much the estate is worth on the market (this is y)

Linear regression creates an equation in which we input the given number (x) and outputs target variable that you want to find (y). We can obtain the equation by training it on pairs of (x,y) values. A dataset can be used containing historic records of house purchases in the form of ("Number of Bedrooms", "Selling Price")

<p align="center" width="100%">
    <img width="33%" src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Dataset_1.png"> 
</p>

We can visualize the data points on a scatter plot to see if there are any new trends. A Scatter plot is 2D plot with each point representating a house. 

On the x-axis "Number of bedrooms" and on the y-axis "Selling Price" for the same houses

<p align="center" width="100%">
    <img width="33%" src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Dataset1_Scatter.png"> 
</p>

We can see that there is a trend in the image above, more bedrooms result in a higher selling price. Considering a linear regression model is trained to get an equation of form:

<p align="center">
  Selling Price = $77,143 * (Number Of Bedrooms) - $74,286 = $80,000
</p>
 
We can also visualize graphiically what woud be the price for houses with different number of bedrooms 

<p align="center" width="100%">
    <img width="33%" src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Dataset1_Linear.png"> 
</p>

## Linear Regression Model Representation withh Regression Equation

Once a linear regression mode is trainned the model forms a linear regression equation of the type

<p align="center" width="100%">
    <img width="33%" src="https://github.com/absolutelyharsh/Linear_Regression/blob/main/Images/Linear_Regression_Equation.png"> 
</p>

In the above equation:

- y is the <<b> output variable </b>. It is also called the <b> target variable </b> in machine learning or the <b> dependednt variable </b> in statistical modeling. It represents the continous value that we are trying to predict.

- x is the <b> input variable </b>. In machine learning referred to as <b> feature variable </b> or the <b> independent variable </b> in statistical modeling. Represents the information given to us at any time.

- w0 is the <b> bias term </b> or <b> y-axis intercept </b>

- w1 is the regression coefficient or scale factor. In classical statistics, it is thhe equivalent of the slope on the best-fit straight line that is produced after the model has been fitted.

- wi are called <i> weights </i> in general.

The goal of linear regrerssion can be defined as finding the unknown parameters of the equation; that is to find the values for the weights w0 and w1. 
