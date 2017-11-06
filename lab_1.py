
# coding: utf-8

# ## L1 - Linear models and gradient descent
# 
# ### Books
# 1. [Deep Learning, I. Goodfellow, Y. Bengio and A. Courville](http://www.deeplearningbook.org/)
# 2. [Neural networks for pattern recognition, C. Bishop](http://cs.du.edu/~mitchell/mario_books/Neural_Networks_for_Pattern_Recognition_-_Christopher_Bishop.pdf)
# 3. [Machine learning: a probabilistic perspective, K. Murphy](http://dsd.future-lab.cn/members/2015nlp/Machine_Learning.pdf)

# ### 0. Basic classification
# 
# Here you can see basic (possible, non standard) classification of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) tasks.
# 1. [Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)
#  1. [Regression](https://en.wikipedia.org/wiki/Regression_analysis)
#  2. [Classification](https://en.wikipedia.org/wiki/Statistical_classification)
#  3. [Ranking](https://en.wikipedia.org/wiki/Learning_to_rank)
# 2. [Reinforcment learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
# 3. [Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
#  1. Clustering 
#  2. Manifold learning
#  3. Matrix decompostion (factorization)
#  4. Dimension reduction
#  
# In this lab we consider only supervised learning. Namely, linear regression and binary linear classification, as simple methods for beginning.

# ### 1. Supervised learning basics
# A supervised learning algorithm is an algorithm that is able to learn from data. Now we need only to clarify what is data and what it means to learn?
# 
# Let $\{x_i\}_{i=1}^{\mathcal{l}} \subset \mathbb{X} = \mathbb{R}^{n}$ and $\{y_i\}_{i=1}^{\mathcal{l}} \subset \mathbb{Y}$. Here $\mathbb{X}$ is the whole set of objects and $\mathbb{Y}$ is all possible labels of objects, so $\{x_i\}_{i=1}^{\mathcal{l}}$ is subset with known labels $\{y_i\}_{i=1}^{\mathcal{l}}$. We want to find algorithm, that can predict $y$ for any $x \in \mathbb{X}$. Actually, $x = (x^1, \dots, x^n)$ is some vector of features (formal description), but $x^k$ can have different nature. 
# 
# * $x^k \in \{0, 1\}$ – binary feature, boolean flag
# * $x^k \in \{1,\dots, m\}$ – categorical (nominal), classification of entities into particular categories.
# * $x^k \in \{1,\dots, m\}^<$ – ordinal, classification of entities in some kind of ordered relationship.
# * $x^k \in \mathbb{R}$ – cardinal, classification based on a numerical value.
# 
# Categorical features are commonly encoded in some way (for exaple [one-hot encoding](https://en.wikipedia.org/wiki/One-hot)) to ignore false ordering (important for metric algorithms). Moreover it's possible to cast any type of feature to $\mathbb{R}$, that's why we suppouse that $\mathbb{X} = \mathbb{R}^{n}$ further.
# 
# Process of finding algorithm, that can predict labels, is called training. Usually, it is reduced to minimization problem of the empirical risk.
# $$\arg \min_{\theta} Q(\theta) = \arg \min_{\theta} \frac{1}{\mathcal{l}}\sum_{i=1}^{\mathcal{l}} \mathcal{L}(f(x_i | \theta), y_i).$$
# Here $\mathcal{L}$ – some loss function that shows how good we predict $y$, and $f(x|\theta)$ is parametric function, where $\theta \in \Theta$.

# ### 2. Linear regression
# For regression task $\mathbb{Y} = \mathbb{R}$. In case of linear model we have learning vector of parameters $w \in \mathbb{R}^n$ and predict $y$ as 
# $$y = w^Tx + b.$$
# 
# For simplicity, let the last element of $x$ is always $1$ and $w$ is concatenation of $[w, b]$. So, we can rewrite model as $y = w^Tx$. For MSE (mean square error) we have following optimization problem
# $$\arg \min_{w} Q(w) = \arg \min_{w} \frac{1}{\mathcal{l}} \sum_{i=1}^{\mathcal{l}}\big(w^Tx_i - y_i\big)^2.$$
# 
# Let $X$ is matrix, where $i$-th row is feature vector of $i$-th object and $Y$ – vector of labels. In this case our expression can be rewritten in matrix form
# $$\arg\min_{w}||Xw - Y ||_{2}.$$
# But this problem is already well studied and has the analytical solution
# $$w = (X^TX)^{-1}X^TY.$$
# 
# #### Exercises
# 1. Let $y = sin(x) + \varepsilon$, where $x \in [0, 2\pi]$ and $\varepsilon \sim \mathcal{N}(0, 0.1)$. Generate 20 train samples and try to learn regression model.
# 2. Plot train data and model's predictions.
# 3. As you see, model has no enough capacity to fit train data. Let's add polynomial features, namely $x^2$ and $x^3$.
# 4. Train linear model one more time and plot results again.
# 5. What happens if you add more features, for example full range $x^{0},\dots,x^{7}$? 

# In[199]:


get_ipython().magic('matplotlib inline')

import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt

PI = np.pi


# In[200]:


def lin_regression(X, y):
    return inv(X.T.dot(X)).dot(X.T.dot(y))


X_train = X = np.linspace(0, 2 * PI, 20).reshape(-1, 1)
y_train = np.sin(X_train) + np.random.normal(0, 0.1, size=(20, 1))

w = lin_regression(X_train, y_train)
y_pred = w * X_train

plt.plot(X, y_train, label='train')
plt.plot(X, y_pred, label='predicted')
plt.legend()
plt.show()


# In[201]:


def poly(X, deg_list):
    deg = max(deg_list)
    num_deg = len(deg_list)
    deg_list.sort()
    
    X_cp = np.copy(X)
    X_cp = np.concatenate((X_cp, np.ones((X_cp.shape[0], 1))), axis=1)
    
    for i in range(2, deg + 1):
        X_cp = np.concatenate((X**i, X_cp), axis=1)
    return (X_cp[:, deg * np.ones(num_deg, dtype=int) - np.asarray(deg_list)])[:, ::-1]

X_train = poly(X, list(range(1, 4)))
w = lin_regression(X_train, y_train)
y_pred = X_train.dot(w)

plt.plot(X, y_train, label='train')
plt.plot(X, y_pred, label='predicted')
plt.legend()
plt.show()


# In[202]:


X_train = poly(X, list(range(7)))
w = lin_regression(X_train, y_train)
y_pred = X_train.dot(w)

plt.plot(X, y_train, label='train')
plt.plot(X, y_pred, label='predicted')
plt.legend()
plt.show()


# As can be seen, adding more features gives us higher precision. (However, it can't last forever. Also out tests are based on training set.)

# ### 3. Validation
# The data used to build the final model usually comes from multiple datasets. In particular, three data sets are commonly used in different stages of the creation of the model.
# 
# 1. We initially fit our parameters on a __training dataset__, that consists of pairs of a feature vector and the corresponding answer. The current model is run with the training dataset and produces a result, which is then compared with the target, for each input vector in the training dataset. Based on the result of the comparison and the specific learning algorithm being used, the parameters of the model are adjusted. The model fitting can include both variable selection and parameter estimation.
# 
# 2. Second one called the __validation dataset__. The validation dataset provides an unbiased evaluation of a model fit on the training dataset while tuning the model's hyperparameters (e.g. regularization coefficient or number of hidden units in a neural network). Validation datasets can be used for regularization by early stopping: stop training when the error on the validation dataset increases, as this is a sign of overfitting to the training dataset. This simple procedure is complicated in practice by the fact that the validation dataset's error may fluctuate during training. This complication has led to the creation of many ad-hoc rules for deciding when overfitting has truly begun.
# 
# 3. Finally, the __test dataset__ is a dataset used to provide an unbiased evaluation of a final trained model.
# 
# Cross-validation is a validation technique for estimating how accurately a predictive model will perform in practice. The goal of cross validation is to limit problems like overfitting, give an insight on how the model will generalize to an independent dataset.
# 
# Cross-validation involves partitioning a sample of data into complementary subsets, performing the analysis on one subset and making validation on the other. To reduce variability, multiple rounds of cross-validation are performed using different partitions, and the validation results are caveraged over the rounds to estimate a final predictive model.
# 
# There are following types:
# 1. Leave-p-out cross-validation - using p observations as the validation set with all possible ways.
# 2. k-fold cross-validation - split data into k folds and using each one as validation set.
# 3. Holdout validation - randomly split data into training and validation set
# 4. Repeated random sub-sampling validation - repeatedly make random splits of data into training and validation set
# 
# #### Exercises
# 1. Generate 20 validation samples
# 2. Check quality of your model on train set and validation set.
# 3. Have you experienced [overfitting](https://en.wikipedia.org/wiki/Overfitting)?
# 4. Please, read [this article](https://en.wikipedia.org/wiki/VC_dimension) to learn more about model capacity and VC-dimension.

# In[204]:


X_valid = poly(np.random.uniform(0, 2 * PI, size=(20, 1)), list(range(7)))
y_pred_val = X_valid.dot(w)

plt.plot(X, y_train, label='train')
plt.plot(X, y_pred, label='prediction on train')
plt.plot(X, y_pred_val, label='prediction on validation')
plt.legend()
plt.show()


# It's clear, that our model was overfitted, because precision level is almost zero.

# ### 4. Binary linear classification
# Let $\mathbb{Y} = \{-1, +1\}$ for binary classification. So linear model looks like
# $$sign(w^Tx + b),$$
# where $w$ is normal to the separating plane, which is defined parametrically $w^Tx+b=0$. In the half-space, which normal is directed, all points has class +1, otherwise -1. Let's assume that all points of hyperplane has class +1 to resolve the ambiguity. Also we rewrite model in the short variant $sign(w^Tx)$.
# 
# As with regression, training of linear classifier may be reduced to an optimization problem. We only have to specify the loss function. The most nature option is
# $$\mathcal{L}(y_{pred}, y_{true}) = [y_{pred} \neq y_{true}] = [M < 0],$$
# where $M$ is the margin value $yw^Tx$, which indicates how far the classifier puts a point in its class. But this loss has one drawback, it's not differentiable. That's why the optimization problem becomes very complex. However we can use any other function, which majorizes this loss. You can find some popular options below
# 
# 1. MSE has one big advantage, we optimize convex function with a local minimum. Moreover analytic solution exists.
# $$\big(w^Tx - y \big)^2$$
# 
# 2. Hinge loss function makes our linear classifier [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) (support vector machine).
# $$max \big(0, 1 - yw^Tx \big)$$
# 
# 3. Logistic loss function has a probabilistic meaning. In particular, this loss leads us to the optimal [Bayesian classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) under certain assumptions on the distribution of features. But it's a different story. So it is often used in practice.
# $$\ln \big( 1 + \exp(-yw^Tx) \big)$$

# #### Exercises
# 1. Let $\mathbb{P}\{y=1|x\} = \sigma(wx)$, where $\sigma(z) = \frac{1}{1 + \exp(-z)}$. Show that problem below it is nothing like the maximization of the likelihood.
# $$\arg\min_{w}Q(w) = \arg\min_{w} \sum_{x, y} \ln \big(1 + \exp(-yw^Tx )) \big)$$
# 2. Plot all loss functions in the axes $M \times L$.
# 3. Generate two normally distributed sets of points on the plane.
# 4. Let points of 1th set (red color) have class +1 and point of 2d set (blue color) have -1.
# 5. Train linear classifier with MSE (use analytical solution), which splits these sets.
# 6. Plot points and separating line of trained classifier.
# 7. What is time comlexity of your solution?

# ### 5. Gradient descent
# Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point. Gradient descent is based on the observation that if function $Q(x)$ is defined and differentiable in a neighborhood of a point $x$, then $Q(x)$ decreases fastest if one goes from $x$  in the direction of the negative gradient.
# 
# $$x^{k+1} = x^{k} - \lambda \cdot \triangledown Q(x)$$
# 
# Here $\lambda$ is step of descent and  $k$ – step number. If $\lambda$ is too large then algorithm may not converge, otherwise training can last a long time. Also there is rather popular hack to slowly decrease $\lambda$ with each step. You need to understand that gradient descent finds exactly local minimum. The easiest way to fight this problem is make several runs of algorithm or have good initialization.
# 
# #### Exercises
# 1. Suggest some quadratic function $Q: \mathbb{R}^2 \rightarrow \mathbb{R}$ with global minimum.
# 2. Find minimum with gradient descent method.
# 3. Plot contour lines.
# 4. Trace the path of gradient descent.
# 5. How do you choose $\lambda$?
# 6. Evaluate time complexity of solution.

# There is category of function which naive gradient descent works poorly for, e.g. [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
# $$f(x, y) = (1-x)^2 + 100(y-x^2)^2.$$
# 
# #### Exercises
# 1. Repeat previous steps for Rosenbrock function.
# 2. What problem do you face?
# 3. Is there any solution?

# There are some variations of the method, for example steepest descent, where we find optimal $\lambda$ for each step.
# $$\lambda^{k} = \arg\min_{\lambda}Q(x_k - \lambda\triangledown Q(x_k)).$$
# 
# #### Exercises
# 1. Split red and blue sets of points again. Train linear model using gradient descent and MSE.
# 2. Plot your splitting line. Compare with analytical solution.
# 3. Try steepest descent.
# 4. Comare gradient descent methods and show its convergence in axes $[step \times Q]$.

# ### 6. Stochastic gradient descent

# Sometimes you have so huge amount of data, that usual gradient descent becomes too slow. One more option, we have deal with data flow. In this case stochastic gradient method appears on the stage. The idea is simple. You can do a descent step, calculating error and gradient not for all samples, but for some small batch only.
# 
# #### Еxercises
# 1. Download [mnist](https://www.kaggle.com/c/digit-recognizer).
# 2. Train linear classificator for digits 0 and 1, using logistic loss function and stochastic gradient descent.
# 3. Use holdout to check [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) of classification.
# 4. How do accuracy and training time depend on bathch size?
# 5. Plot graphic that proves your words.
# 6. How many epochs you use? Why?
# 7. Plot value of loss function for each step (try use [exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing)).

# #### Momentum method
# Stochastic gradient descent with momentum remembers the update of $x$ at each iteration, and determines the next update as a linear combination of the gradient and the previous update
# $$x^{k+1} = x^{k} - s^{k},$$ where $s^k = \gamma s^{k-1} + \lambda\triangledown Q(x^k)$, $0 <\gamma < 1$ – smoothing ratio and $s^{-1} = 0$.
# 
# #### Еxercises
# 1. Find minimum for $Q(x,y)=10x^2+y^2$ with descent method.
# 2. Use momentum method and compare pathes.
# 3. How do you choose $\gamma$?

# #### Nesterov accelerated gradient
# And the logical development of this approach leads to the accelerated Nesterov's gradient. The descent step is calculated a little differently
# $$s^k = \gamma s^{k-1} + \lambda\triangledown Q(x^k - \gamma s^{k-1}),$$
# so we find gradient at the point which moment will move us.
# 
# #### Еxercises
# 1. Compare this method and previous with Rosenbrock function.
# 2. Plot traces of both algorithms.

# #### Adagrad (2011)
# Adaptive gradient finds lambda for each dimension of the input vector x. Informally speaking, for sparce features it makes a bigger step, but for regular ones smaller step.
# $$x_{i}^{k + 1} = x_{i}^{k} - \frac{\lambda}{\sqrt{G_{i, i}^k } + \varepsilon} \cdot \frac{\partial Q}{\partial x_i}(x^k),$$
# * $G^{k} = \sum_{t=1}^{k}g_t g_t^{T}$, где $g_t = \triangledown Q(x^t)$.
# * $\varepsilon$ - epsilon to avoid division by zero.
# It improves convergence of the learning process (e.g. when using neural networks for text).
# 
# #### RMSprop
# To avoid growth of the denominator we can use the following modification. Let's calculate the matrix $G^k$ only for a small number of latest steps, it can be done for example using exponential smoothing.
# $$G^{k+1} = \gamma G^{k} + (1 - \gamma)g_{k+1}g_{k+1}^{T},$$
# where $0< \gamma < 1$ - smoothing factor
# 
# #### Еxercises
# 1. Read about adadelta and adam (links below).
# 2. Give an example of a function that can show the difference in the studied stohastic gradient methods.
# 3. Show animation step by step how methods work.
# 4. Use your favorite method on mnist dataset again.
# 5. Show convergence of alrotigthm.
# 6. Check quality, using holdout.
# 
# #### Papers
# 1. [Adadelta (2012)](https://arxiv.org/pdf/1212.5701.pdf)
# 2. [Adam (2015)](https://arxiv.org/pdf/1412.6980.pdf)
