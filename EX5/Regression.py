#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# # Exercise 5: Generalising regression

# In this exercise you will generalise regression to $N$th order polynomials and use it to predict the price of a house (in Canadian dollars) based on its lot size (in square feet).
# 
# Suppose you want to buy a house in the City of Windsor, Canada. You contact a real-estate salesperson to get information about current house prices and receive details on 546 properties sold in Windsor in the last two years. You would like to figure out what the expected cost of a house might be given only the lot size of the house you want to buy. Fortunately, his dataset has only one independent variable (i.e. `lotsize`, the lot size of a property) and one dependent variable (i.e. `price`, the sale price of a house). You will train the dataset using polynomial regression to predict the house prices.
# 
# A polynomial model of order $N$ is defined by:
# $$
# 	y = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_N x^N,
# $$
# in which, the coefficients $\theta_i$ are the parameters of the model. Notice how the function is linear in the parameters, i.e. if $x$ is fixed, the function is linear. To estimate the parameters $\theta_i$, you can therefore setup a linear equation and solve for $\theta$:
# $$
# \begin{bmatrix}
#     1 & x_1 & x_1^2 & x_1^3 & \dots & x_1^N \\
#     1 & x_2 & x_2^2 & x_2^3 & \dots & x_2^N \\
#     1 & x_3 & x_3^2 & x_3^3 & \dots & x_3^N \\
#     \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
#     1 & x_m & x_m^2 & x_m^3 & \dots & x_m^N
# \end{bmatrix}
# \times
# \begin{bmatrix}
#     \theta_0 \\
#     \theta_1 \\
#     \theta_2 \\
#     \theta_3 \\
#     \vdots \\
#     \theta_N
# \end{bmatrix}
# =
# \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     y_3 \\
#     \vdots \\
#     y_m
# \end{bmatrix},
# $$
# or more compactly: $A \theta = y$.
#  
# The  *cost function* $J(\theta)$ for linear regression is the mean squared error between the known outputs $y_i$ and the predicted outputs $M_{\theta}(x_i)$ of the model:
# \begin{equation}
# J(\theta) = \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x_{i})-y_{i})^2
# \label{eq:CostFunction}
# \end{equation}

# ## 5.1 Data exploration
# 
# We start by loading the dataset described above:

# In[ ]:


filename = "./inputs/simple_windsor.csv"
names = ["lotsize", "price"]
dataset = np.loadtxt(filename, delimiter=',', dtype=np.int32)

X_full, y_full = dataset.T


# Let us visualise the data:

# In[ ]:


plt.scatter(X_full, y_full)
plt.xlabel('Lot size')
plt.ylabel('House price');


# This visualisation already tells us a lot about the usefulnes of the data. Try to answer the following questions to the best of your abilities:
# 
# ### Task (A)
# 1. Notice the large spread in house prices for relatively similar lot sizes. Can you relate this to a real-world phenomenon? In other words, is it realistic to expect that the price of a house is determined solely by the lot size?
# 2. Can you imagine other factors that might be useful for modelling house prices?
# 3. Can you think of a method to evaluate how useful a given factor is in predicting the house price?

# ### Splitting into train and test data
# We use a helper function from the *Scikit Learn* library to split the dataset into $80\%$ training data and $20\%$ test data:

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42) # TODO


# ## 5.2 Generalising regression
# First, we need to be able to generate design matrices as described above from a single input vector `X`. We provide the function below for creating design matrices for polynomials of arbitrary order:

# In[ ]:


def get_design_matrix(x, order=1):
    """
    Get the coefficients of polynomial in a least square sense of order N.
    """
    if order < 1 or x.ndim != 1:
        return "fail"

    count = x.shape[0] # size
    matrix = np.ones((count, order + 1), np.float64)

    for i in range(1, order+1):
        matrix[:, i] = np.power(x, i)

    return matrix


# ### Task (B)
# 1. **Estimate parameters:** Implement the function `estimate(X, y, order)` below. The function should use `np.linalg.lstsq()` to estimate parameters for the model. Use `get_design_matrix(X, order)` to generate an appropriate design matrix.

# In[ ]:


def estimate(X, y, order):
    coefficient = get_design_matrix(X, order)
    return np.linalg.lstsq(coefficient, y, rcond=None)[0]


# 2. **Model:** Implement the function `linear_model(X, params)` below. It should compute the house price `y` for a given 

# In[ ]:


esti = estimate(X_train, y_train, len(X_train))

X_train @ esti               
                
                
# remember we estimate. since there is no a curce which all our opint lais on that

# def linear_model(X, params):
#     ...


# 3. **Prediction:** The variable `values` contains the integer values between the minimum and maximum lot sizes from the dataset. (A) Estimate parameters from `X_train` and `y_train`. (B) Then calculate the predicted `y`-values for `values` using the estimated parameters. (C) Plot the predicted values as a line-plot.

# In[ ]:


values = np.linspace(X_full.min(), X_full.max(), 50)

# (A) Estimate parameters

# (B) Evaluate model

# Plot training data
plt.scatter(X_train, y_train)

# (C) Plot predicted values


# 4. **Increasing orders:** Try to use higher order polynomials in the estimation and prediction code above. You should see the predictions starting to deviate drastically for orders above 3 or 4. Do you have any idea why this happens? *Hint: It has to do with the behavior of floating point numbers at extreme values.*

# The above problem can be solved simply by normalizing the *lot sizes*. We provide the following functions for easy normalization and unnormalization:

# In[ ]:


def normalized(X):
    n = (X - np.min(X_full))/np.max(X_full)
    return n

def unnormalized(X):
    return X*np.max(X_full) + np.min(X_full)


# 5. **Normalization:** Redo the estimation from task B3 above but with normalized values for `X_train`. Plot the results and experiment with different values for the order. 

# In[ ]:


values = np.linspace(X_full.min(), X_full.max(), 50)

# Estimate parameters and predict y-values


plt.scatter(X_train, y_train, c="g")

# Plot predicted values


# ## 5.3 Evaluation
# We now want to evaluate the model using the test data. You will calculate the *root mean squarred error* for various orders of polynomials and use the error to decide which order has the best tradeoff between bias and variance (underfitting/overfitting).
# 
# The *root mean squared error* is simply the square root of the *mean squared error*: 
# $$
#  \sqrt{\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x_{i})-y_{i})^2}
# $$
# We use it because it represents the average error in the same units as the data, i.e. house prices in our case.
# 

# ### Task (C)
# 1. **Error function:** Implement the `rmse` function below. Remember to normalize the X-values.

# In[ ]:


def rmse(theta, X, y):
    ...


# 2. **Test models:** Finish the implementation of `test_models` below. It should estimate parameters for a given order polynomial using the training data and then record the train and test losses using the `rmse` function.

# In[ ]:


def test_models():
    losses_train = []
    losses_test = []
    for order in range(1, 20):
        # Add code here
        
        rmse_train = ...
        rmse_test = ...
        
        losses_train.append(rmse_train)
        losses_test.append(rmse_test)
    return losses_train, losses_test


# 3. **Plot both losses:** Plot the losses. Are the results what you expected? How does this relate to the dilemma of underfitting and overfitting? 

# In[ ]:





# ### Task (D)
# 1. What does the loss plots tell you about the precision of the models in general?
# 2. Would you be able to improve the test loss to an arbitrary low value using a different model? If yes, explain what model you might use. If no, explain why not.
