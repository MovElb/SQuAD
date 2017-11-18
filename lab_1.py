
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

# In[668]:


get_ipython().magic('matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = '10,8'
from matplotlib import animation

import numpy as np
from numpy.linalg import inv

from scipy.optimize import minimize

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy

PI = np.pi


# In[575]:


def lin_regression(X, y):
    return inv(X.T.dot(X)).dot(X.T.dot(y))

def poly(X, deg_list):
    deg = max(deg_list)
    num_deg = len(deg_list)
    deg_list.sort()
    
    X_cp = np.copy(X)
    X_cp = np.concatenate((X_cp, np.ones((X_cp.shape[0], 1))), axis=1)
    
    for i in range(2, deg + 1):
        X_cp = np.concatenate((X**i, X_cp), axis=1)
    return (X_cp[:, deg * np.ones(num_deg, dtype=int) - np.asarray(deg_list)])[:, ::-1]


X = np.linspace(0, 2 * PI, 20).reshape(-1, 1)
X_train = poly(X, list(range(2)))
noise = np.random.normal(0, 0.1, size=(20, 1))
y_train = np.sin(X) + noise

w = lin_regression(X_train, y_train)
y_pred = X_train.dot(w)

plt.scatter(X, y_train, label='train')
plt.plot(X, y_pred, 'r', label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[576]:


X_train = poly(X, list(range(4)))
w = lin_regression(X_train, y_train)
y_pred = X_train.dot(w)

plt.scatter(X, y_train, label='train')
plt.plot(X, y_pred, 'r', label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[577]:


X_train = poly(X, list(range(7)))
w = lin_regression(X_train, y_train)
y_pred = X_train.dot(w)

plt.scatter(X, y_train, label='train')
plt.plot(X, y_pred, 'r', label='predicted')
plt.legend()
plt.grid()
plt.show()


# As can be seen, adding more features gives us higher precision. (However, it can't last forever. Also our tests are based only on the training set, so they can't be really unbiased.)

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

# In[578]:


def MSE(y_pred, y_stand):
    return ((y_pred - y_stand)**2).mean()

X_valid = np.random.uniform(0, 2 * PI, size=(20, 1))
X_valid_poly = poly(X_valid, list(range(7)))
y_valid = np.sin(X_valid) + noise
y_pred_val = X_valid_poly.dot(w)

print("MSE on train set is %f"%MSE(y_pred, y_train))
print("MSE on validation set is %f"%MSE(y_pred_val, y_valid))


# It's clear, that our model is overfitted, because MSE is much higher than MSE on train set.

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

# # Exercises
# 1. Let $\mathbb{P}\{y=1|x\} = \sigma(w^T x)$, where $\sigma(z) = \frac{1}{1 + \exp(-z)}$. Show that problem below it is nothing like the maximization of the likelihood.
# $$\arg\min_{w}Q(w) = \arg\min_{w} \sum_{x, y} \ln \big(1 + \exp(-yw^Tx )) \big)$$
# 2. Plot all loss functions in the axes $M \times L$.
# 3. Generate two normally distributed sets of points on the plane.
# 4. Let points of 1th set (red color) have class +1 and point of 2d set (blue color) have -1.
# 5. Train linear classifier with MSE (use analytical solution), which splits these sets.
# 6. Plot points and separating line of trained classifier.
# 7. What is time comlexity of your solution?

# First of all, it's pretty clear that $\mathbb{P}(y = -1 | x) = \sigma(-w^Tx)$(it is just 1 - $\mathbb{P}(y = 1 | x))$. We assume that $\mathcal{L}(w\ |\ X) = \mathbb{P}(y\ |\ X)$, where $y \in \mathbb{R}^n, X \in \mathbb{R}^{m \times n}$, moreover, all of $X_i$ are indepedent. Hence,
# $$\arg \max \limits_{w} \mathcal{L}(w\ |\ X) = \arg \max \limits_{w} \mathbb{P}(y\ |\ X) = \arg \max \limits_{w} \prod \limits_{i = 1}^{n} \mathbb{P}(y_i\ |\ X_i) = \arg \max \limits_{w} \ln \prod \limits_{i = 1}^{n} \mathbb{P}(y_i\ |\ X_i) =$$
# Last equality follows from monotone of logarithm.
# $$= \arg \max \limits_{w} \sum \limits_{y_i, X_i} \ln \mathbb{P}(y_i\ |\ X_i) = \arg \max \limits_{w} \sum \limits_{y_i, X_i} \ln ((1 + \exp(-yw^Tx))^{-1}) = -\arg \max \limits_{w} \sum \limits_{y_i, X_i} \ln (1 + \exp(-yw^Tx)) = \arg \min \limits_{w} \sum \limits_{y_i, X_i} \ln (1 + \exp(-yw^Tx))$$

# In[579]:


M = np.linspace(-2, 2, 800)
plt.plot(M, 400 * [1] + 400 * [0], label="$M < 0$")
plt.plot(M, (1 - M)**2, label="$(1 - M)^2$")
plt.plot(M, np.max([np.zeros(800), 1 - M], axis=0), label="$\max(0, 1-M)$")
plt.plot(M, np.log(1 + np.exp(-M)) / np.log(2), label="$\ln(1 + \exp(-M)) / \ln2$")
plt.legend()
plt.grid()
plt.show()


# In[580]:


blob_1 = np.random.normal([10, 10], 7, size=(100, 2))
blob_2 = np.random.normal([-10, -10], 7, size=(100, 2))
plt.scatter(blob_1[:, 0], blob_1[:, 1], color='r')
plt.scatter(blob_2[:, 0], blob_2[:, 1], color='b')
plt.grid()
plt.show()


# In[581]:


X_train = np.concatenate((blob_1, blob_2), axis=0)
X_train = np.concatenate((X_train, np.ones((200, 1))), axis=1)
y_train = np.concatenate((np.ones((100, 1)), -1 * np.ones((100, 1))), axis=0)

w = lin_regression(X_train, y_train)
a, b, c = w[0], w[1], w[2]  # ax + by + c = 0
x = np.linspace(-25, 25, 100)
y = (-a * x - c) / b

plt.plot(x, y, 'g')
plt.scatter(blob_1[:, 0], blob_1[:, 1], color='r')
plt.scatter(blob_2[:, 0], blob_2[:, 1], color='b')
plt.grid()
plt.show()


# Let's find the complexity of the solution. We assume $m$ is the number of points, $n$ is the dimension of vector 
# space.
# For finding the solution we need to compute following expression: $(X^T X)^{-1} X^T Y$.
# Computing $X^TX$ takes $O(m^2n)$. Inverting $X^TX$ using Gauss algorithm is $O(n^3)$, because this is a square matrix.
# Multiplying $X^TX \cdot X$ is $O(mn^2)$ and, finally, multiplying by Y takes $O(nm)$. In summary, $$O(m^2n + n^3 + mn^2 + mn) = O(mn^2 + n^3 + m^2n) = O(mn^2 + n^3)$$

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

# We will use  $f(x, y) = 5(x - 2)^2 + 4(y + 5)^2$ with minumum in $(2, 5)$.

# In[654]:


def apl(pnt, func, vectorwise=False, *args):
    if vectorwise:
        return func(pnt, *args)
    return func(*pnt, *args)


# In[655]:


def naive_grad_descent(func, 
                       deriv, 
                       start_pnt, 
                       lmbda,
                       vecwise=False,
                       max_step=1e5,
                       eps=10e-15):
    
    points = []
    pnt = np.copy(start_pnt)
    points.append(np.copy(pnt))

    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        l = lmbda(step)
        pnt -= l * apl(pnt, deriv, vecwise)
        points.append(np.copy(pnt))

    return pnt, np.asarray(points)


# In[719]:


f = lambda x, y: 5 * (x - 2)**2 + 4 * (y + 5)**2
df = lambda x, y: np.asarray([10 * (x - 2), 8 * (y + 5)])

min_pnt, trace = naive_grad_descent(f, 
                                    df, 
                                    start_pnt=np.random.uniform(0, 5, size=(2,)), 
                                    lmbda=lambda step: 0.05)

print("f_min = %f in (%f, %f)"%(f(*min_pnt), *min_pnt))

x = np.arange(-7, 7, 0.25)
y = np.arange(-7, 7, 0.25)
x, y = np.meshgrid(x, y)
z = f(x, y)
plt.contourf(x, y, z, 30)
plt.plot(trace[:, 0], trace[:, 1], 'r-o')
plt.show()


# $\lambda$ was chosen as small constant $0.02$, because other methods of choosing(e.g. $2^{-step}, \frac{1}{step}$) showed lower results, moreover, constant method converged in all tests.

# There is category of function which naive gradient descent works poorly for, e.g. [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
# $$f(x, y) = (1-x)^2 + 100(y-x^2)^2.$$
# 
# #### Exercises
# 1. Repeat previous steps for Rosenbrock function.
# 2. What problem do you face?
# 3. Is there any solution?

# In[617]:


f_rosen = lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2
df_rosen = lambda x, y: np.asarray([-2 * (1 - x) - 400 * (y - x**2) * x, 200 * (y - x**2)])


# In[618]:


min_pnt_rosen, trace_rosen = naive_grad_descent(f_rosen, 
                                                df_rosen, 
                                                start_pnt=np.random.uniform(-2, 2, size=(2,)), 
                                                lmbda=lambda step: 0.001, 
                                                max_step=2 * 1e5,
                                                eps=1e-8)

print("f_min = %f in (%f, %f)"%(f_rosen(*min_pnt_rosen), *min_pnt_rosen))


# In[620]:


x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
z = f_rosen(x, y)
plt.contourf(x, y, z, 30)
plt.plot(trace_rosen[:, 0], trace_rosen[:, 1], 'r-o')
plt.show()

print("It took algorithm %d steps to reach point (%f, %f)"%(trace_rosen.shape[0], *min_pnt_rosen))


# Main feature of this function is very gentle slope in the neighbourhood of $(1, 1)$ and steep slope outside of it. That's why it takes a lot of iteration to get high precision. Probable solution may be changing $\lambda$ depending on the slope of gradient vector. 

# There are some variations of the method, for example steepest descent, where we find optimal $\lambda$ for each step.
# $$\lambda^{k} = \arg\min_{\lambda}Q(x_k - \lambda\triangledown Q(x_k)).$$
# 
# #### Exercises
# 1. Split red and blue sets of points again. Train linear model using gradient descent and MSE.
# 2. Plot your splitting line. Compare with analytical solution.
# 3. Try steepest descent.
# 4. Comare gradient descent methods and show its convergence in axes $[step \times Q]$.

# In[625]:


N_POINTS = 500
blob_1 = np.random.normal(-1, 1, size=(N_POINTS, 2))
blob_2 = np.random.normal(2, 1, size=(N_POINTS, 2))

X_train = np.concatenate((blob_1, blob_2), axis=0)
X_train = np.concatenate((X_train, np.ones((2 * N_POINTS, 1))), axis=1)
y_train = np.concatenate((np.ones((N_POINTS, 1)), -1 * np.ones((N_POINTS, 1))), axis=0)


# In[626]:


plt.scatter(blob_1[:, 0], blob_1[:, 1], color='r')
plt.scatter(blob_2[:, 0], blob_2[:, 1], color='b')
plt.grid()
plt.show()


# In[657]:


w = lin_regression(X_train, y_train)
a, b, c = w[0], w[1], w[2]  # ax + by + c = 0
x = np.linspace(-3, 3, 100)
y_an_mse = (-a * x - c) / b


# In[658]:


def gen_mse(X_train, y_train):
    return lambda a, b, c: ((X_train.dot(np.asarray([a, b, c])).reshape(2 * N_POINTS, 1) - y_train)**2).mean()


def gen_gr_mse(X, y):
    return lambda a, b, c: 2 * X.T.dot((X.dot(np.asarray([a, b, c])).reshape(2 * N_POINTS, 1) - y)).reshape(3,)


f_mse = gen_mse(X_train, y_train)
df_mse = gen_gr_mse(X_train, y_train)


# In[659]:


st_pnt = 10 * np.random.randn(3)
min_pnt_mse, trace_mse = naive_grad_descent(f_mse, 
                                            df_mse, 
                                            start_pnt=st_pnt, 
                                            lmbda=lambda step: 1e-4, 
                                            max_step=150,
                                            eps=1e-3)

print("f_min = %f in (%f, %f, %f)"%(f_mse(*min_pnt_mse), *min_pnt_mse))


# In[660]:


a_gr, b_gr, c_gr, = min_pnt_mse
y_gr_mse = (-a_gr * x - c_gr) / b_gr

plt.plot(x, y_an_mse, 'y', label='Analytical solution')
plt.plot(x, y_gr_mse, 'g', label='Naive gradient solution')

plt.scatter(blob_1[:, 0], blob_1[:, 1], color='r')
plt.scatter(blob_2[:, 0], blob_2[:, 1], color='b')

plt.legend()
plt.grid()
plt.show()


# In[661]:


def fast_grad(func, 
              deriv, 
              start_pnt, 
              max_step=1e5,
              vecwise=False,
              eps=10e-15):
    
    points = []
    pnt = np.copy(start_pnt)
    points.append(np.copy(pnt))

    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        drv = apl(pnt, deriv, vecwise)
        lmb = minimize(lambda l: apl(pnt - l * drv, func, vecwise), x0=0).x
        pnt -= lmb * drv
        points.append(np.copy(pnt))

    return pnt, np.asarray(points)


# In[662]:


min_pnt_fast, trace_fast = fast_grad(f_mse, 
                                     df_mse, 
                                     start_pnt=st_pnt, 
                                     max_step=150,
                                     eps=1e-8)


# In[663]:


a_fast, b_fast, c_fast = min_pnt_fast
y_gr_fast = (-a_fast * x - c_fast) / b_fast

plt.plot(x, y_an_mse, 'y', label='Analytical')
plt.plot(x, y_gr_mse, 'g', label='Naive gradient')
plt.plot(x, y_gr_fast, 'b', label='Steepest gradient')

plt.scatter(blob_1[:, 0], blob_1[:, 1], color='r')
plt.scatter(blob_2[:, 0], blob_2[:, 1], color='b')

plt.legend()
plt.grid()
plt.show()


# As we see perfomance of steepest descent has the same precision as analytical solution, however naive gradient descent is less precise.

# In[667]:


f_mse_val = list(map(lambda pnt: f_mse(*pnt), trace_mse))
f_fast_val = list(map(lambda pnt: f_mse(*pnt), trace_fast))

plt.plot(range(trace_mse.shape[0]), f_mse_val, label='Naive gradient')
plt.plot(range(trace_fast.shape[0]), f_fast_val, label='Steepest gradient')
plt.legend()
plt.xlabel('Step')
plt.ylabel(;)
plt.grid()
plt.show()


# Not only is steepest gradient is more precise, moreover, its convergence takes less steps.

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

# In[228]:


data = np.loadtxt("../train.csv", delimiter=',', skiprows=1)


# In[229]:


data_0_1 = data[data[:, 0] < 2, :]
data_0_1[data_0_1[:, 0] < 1, 0] = -1
data_0_1 = np.concatenate((data_0_1, np.ones((data_0_1.shape[0], 1))), axis=1)
data_0_1 = data_0_1.astype(np.float128)
dig_X_train, dig_X_test, dig_y_train, dig_y_test = train_test_split(data_0_1[:, 1:],
                                                                    data_0_1[:, 0],
                                                                    test_size=0.15, 
                                                                    random_state=42)


# In[504]:


def log_loss(w, X, y):
    return sum(np.log(1 + np.exp(-y[i] * X[i].dot(w))) for i in range(X.shape[0])) / X.shape[0]


def log_deriv(w, X, y):
    grad = np.zeros(X.shape)
    for i in range(X.shape[0]):
        deg = -y[i] * np.dot(X[i], w)
        grad[i] = -y[i] * X[i] * (np.exp(deg) / (1 + np.exp(deg)))
    grad = np.sum(grad, axis=0) / X.shape[0]
    return grad.reshape(-1, 1)


def sgd(func, 
        grad, 
        start_pnt,
        X,
        y,
        lmbda,
        batch_size=1,
        vecwise=False, 
        max_step=1e3, 
        eps=1e-12):
    
    pnt = np.copy(start_pnt)
    points = []
    points.append(np.copy(start_pnt))
    it_per_batch = X.shape[0] // batch_size
    
    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise, X, y) - apl(points[-2], func, vecwise, X, y)) >= eps)     and step <= max_step:
        for j in range(it_per_batch):
            if not ((step == 0 or abs(apl(points[-1], func, vecwise, X, y) - apl(points[-2], func, vecwise, X, y)) >= eps)     and step <= max_step):
                break
            step += 1
            X_tmp = X[j * batch_size: (j + 1) * batch_size]
            y_tmp = y[j * batch_size: (j + 1) * batch_size]
            gr = apl(pnt, grad, vecwise, X_tmp, y_tmp)
            pnt -= lmbda * gr
            points.append(np.copy(pnt))
            
    return pnt, np.array(points)


def classify(X, w):
    return np.sign(X.dot(w))


# In[505]:


min_sgd, trace_sgd = sgd(log_loss, 
                         log_deriv,
                         np.zeros((dig_X_train.shape[1], 1)),
                         dig_X_train,
                         dig_y_train,
                         1e-2,
                         vecwise=True,
                         batch_size=1000,
                         max_step=30)


# In[506]:


print("Accuracy on train %f"%accuracy(classify(dig_X_train, min_sgd), dig_y_train))
print("Accuracy on test %f"%accuracy(classify(dig_X_test, min_sgd), dig_y_test))


# In[235]:


step_num = []
accur_p_batch = []
for sz in range(1, dig_X_train.shape[0], 100):
    min_sgd, trace_sgd = sgd(log_loss, 
                             log_deriv, 
                             np.zeros((dig_X_train.shape[1], 1)),
                             dig_X_train,
                             dig_y_train,
                             1e-2,
                             vecwise=True,
                             batch_size=sz,
                             max_step=10)
    step_num.append(trace_sgd.shape[0])
    accur_p_batch.append(accuracy(classify(dig_X_test, min_sgd), dig_y_test))


# In[236]:


plt.plot(range(101, dig_X_train.shape[0], 100), step_num[1:], 'r-')
plt.xlabel("Batch size", fontsize=13)
plt.ylabel("Steps", fontsize=13)
plt.grid()
plt.show()


# In[237]:


plt.plot(range(1, dig_X_train.shape[0], 100), accur_p_batch, 'b')
plt.xlabel("Batch size", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.grid()
plt.show()


# In[238]:


min_sgd, trace_sgd = sgd(log_loss, 
                         log_deriv,
                         np.zeros((dig_X_train.shape[1], 1)),
                         dig_X_train,
                         dig_y_train,
                         1e-2,
                         vecwise=True,
                         batch_size=200,
                         max_step=10)

log_vals = list(map(lambda pnt: log_loss(dig_X_train, dig_y_train, pnt), trace_sgd))


# In[239]:


plt.plot(log_vals, 'r')
plt.xlabel("Steps", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.grid()


# #### Momentum method
# Stochastic gradient descent with momentum remembers the update of $x$ at each iteration, and determines the next update as a linear combination of the gradient and the previous update
# $$x^{k+1} = x^{k} - s^{k},$$ where $s^k = \gamma s^{k-1} + \lambda\triangledown Q(x^k)$, $0 <\gamma < 1$ – smoothing ratio and $s^{-1} = 0$.
# 
# #### Еxercises
# 1. Find minimum for $Q(x,y)=10x^2+y^2$ with descent method.
# 2. Use momentum method and compare pathes.
# 3. How do you choose $\gamma$?

# In[316]:


def grad_momentum(func, 
                  deriv, 
                  start_pnt, 
                  lmbda, 
                  gamma,
                  vecwise=False,
                  max_step=1e5, 
                  eps=10e-15):
    pnt = np.copy(start_pnt)
    points = [np.copy(start_pnt)]
    
    step = 0
    s = np.zeros(start_point.shape[0])
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        s = gamma * s + lmbda(step) * apl(pnt, deriv, vecwise)
        pnt -= s
        points.append(np.copy(pnt))
    return pnt, np.asarray(points)


# In[317]:


def quadratic(x, y):
    return 10 * x**2 + y**2


def grad_quadr(x, y):
    return np.asarray([20 * x, 2 * y])


# In[318]:


x = y = np.linspace(-15, 15, 1000)
xx, yy = np.meshgrid(x, y)
f = quadratic(xx, yy)

start_point = np.asarray([-7.35532, 9.34234])

min_moment, trace_moment = grad_momentum(quadratic,
                                         grad_quadr,
                                         np.copy(start_point),
                                         lambda step: 1e-2,
                                         0.8)

min_nve, trace_nve = naive_grad_descent(quadratic, 
                                        grad_quadr, 
                                        np.copy(start_point),
                                        lambda step: 1e-2)

plt.contourf(xx, yy, f, 30)
plt.plot(trace_nve[:, 0], trace_nve[:, 1], 'r-o', label='Naive gradient')
plt.plot(trace_moment[:, 0], trace_moment[:, 1], 'g-o', label='Momentum method')
plt.legend()
plt.show()


# I looked at all $\gamma$ in $(0, 1)$ with step $0.1$, then depending on number of steps found minimal $\gamma$. Then I searched for more accurate value in the neighbourhood of minimal $\gamma$ from previous step.

# ## Nesterov accelerated gradient
# And the logical development of this approach leads to the accelerated Nesterov's gradient. The descent step is calculated a little differently
# $$s^k = \gamma s^{k-1} + \lambda\triangledown Q(x^k - \gamma s^{k-1}),$$
# so we find gradient at the point which moment will move us.
# 
# #### Еxercises
# 1. Compare this method and previous with Rosenbrock function.
# 2. Plot traces of both algorithms.

# In[319]:


def grad_nesterov(func, 
                  deriv, 
                  start_pnt, 
                  lmbda, 
                  gamma,
                  vecwise=False,
                  max_step=1e5, 
                  eps=10e-15):
    pnt = np.copy(start_pnt)
    points = [np.copy(start_pnt)]
    
    step = 0
    s = np.zeros(start_point.shape[0])
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        s = gamma * s + lmbda(step) * apl(pnt - gamma * s, deriv, vecwise)
        pnt -= s
        points.append(np.copy(pnt))
    return pnt, np.asarray(points)


# In[322]:


start_point = np.random.randn(2) * 10

min_mom_ros, trace_mom_ros = grad_momentum(f_rosen,
                                           df_rosen,
                                           np.copy(start_point),
                                           lambda step: 0.00002,
                                           0.91)

min_nest_ros, trace_nest_ros = grad_nesterov(f_rosen,
                                             df_rosen,
                                             np.copy(start_point),
                                             lambda step: 0.00002,
                                             0.91)

x = y = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x, y)
f = f_rosen(xx, yy)

plt.contourf(xx, yy, f, 30)
plt.plot(trace_mom_ros[:, 0], trace_mom_ros[:, 1], 'r-o', label='Momentum')
plt.plot(trace_nest_ros[:, 0], trace_nest_ros[:, 1], 'g-o', label='Nesterov')
plt.legend()
plt.show()


# In[323]:


OFFSET = 1000

plt.plot(range(trace_mom_ros.shape[0])[OFFSET:],          list(map(lambda p: f_rosen(*p), trace_mom_ros))[OFFSET:], 'r', label='Momentum')

plt.plot(range(trace_nest_ros.shape[0])[OFFSET:],          list(map(lambda p: f_rosen(*p), trace_nest_ros))[OFFSET:], 'b', label='Nesterov')
plt.grid()
plt.legend()
plt.show()


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

# In[753]:


def adagrad(func, 
            deriv, 
            start_pnt, 
            lmbda,
            vecwise=False,
            max_step=1e5,
            eps=10e-8):
    
    pnt = np.copy(start_pnt)
    points = [np.copy(pnt)]
    g = np.empty(start_pnt.shape)
    
    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        drv = apl(pnt, deriv, vecwise)
        g += drv**2
        pnt -= lmbda(step) * drv / (np.sqrt(g) + eps)
        points.append(np.copy(pnt))
    return pnt, np.asarray(points)


def RMSprop(func, 
            deriv, 
            start_pnt, 
            lmbda,
            gamma,
            vecwise=False,
            max_step=1e5, 
            eps=10e-6):
    
    pnt = np.copy(start_pnt)
    points = [np.copy(pnt)]
    g = np.empty(start_pnt.shape)
    
    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        drv = apl(pnt, deriv, vecwise)
        g = gamma * g  + (1 - gamma) * drv**2
        pnt -= lmbda(step) * drv / (np.sqrt(g) + eps)
        points.append(np.copy(pnt))
    return pnt, np.asarray(points)


def adadelta(func, 
             deriv, 
             start_pnt, 
             gamma,
             vecwise=False,
             max_step=1e5, 
             eps=10e-6):
    
    pnt = np.copy(start_pnt)
    points = [np.copy(pnt)]
    g = np.empty(start_pnt.shape)
    dx = np.empty(start_pnt.shape)
    
    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        drv = apl(pnt, deriv, vecwise)
        g = gamma * g  + (1 - gamma) * drv**2
        lmbda_drv = (np.sqrt(dx) + eps) * drv / (np.sqrt(g) + eps)
        dx = gamma * dx  + (1 - gamma) * lmbda_drv**2
        pnt -= lmbda_drv
        points.append(np.copy(pnt))
    return pnt, np.asarray(points)

def adam(func, 
         deriv, 
         start_pnt,
         vecwise=False,
         alpha=0.001,
         beta_1=0.9,
         beta_2=0.999,
         max_step=30, 
         eps=10e-6):
    pnt = np.copy(start_pnt)
    points = [np.copy(pnt)]
    momentum = np.empty(start_pnt.shape)
    velocity = np.empty(start_pnt.shape)
    
    step = 0
    while (step == 0 or abs(apl(points[-1], func, vecwise) - apl(points[-2], func, vecwise)) >= eps)                     and step <= max_step:
        step += 1
        drv = apl(pnt, deriv, vecwise)
        momentum = beta_1 * momentum + (1 - beta_1) * drv
        velocity = beta_2 * velocity + (1 - beta_2) * drv**2
        update = alpha * (momentum / (1 - beta_1**step)) / (np.sqrt(velocity / (1 - beta_2**step)) + eps)
        pnt -= update
        points.append(np.copy(pnt))
    return pnt, np.asarray(points)


# In[742]:


start = np.random.randn(2) * 10


# In[745]:


print(start)


# In[754]:


_, momentum_tr = grad_momentum(f,
                               df,
                               np.copy(start),
                               lambda step: 0.00002,
                               0.91)
print('Momentum is done Steps %d. Point (%f, %f)'%(momentum_tr.shape[0], *momentum_tr[-1]))
print('='*30)

_, nesterov_tr = grad_nesterov(f,
                               df,
                               np.copy(start),
                               lambda step: 0.00002,
                               0.91)

print('Nesterov is done. Steps %d. Point (%f, %f)'%(nesterov_tr.shape[0], *nesterov_tr[-1]))
print('='*30)

_, adagrad_tr = adagrad(f,
                        df,
                        np.copy(start),
                        lambda step: 0.0000002)

print('Adagrad is done. Steps %d. Point (%f, %f)'%(adagrad_tr.shape[0], *adagrad_tr[-1]))
print('='*30)

_, rmsprop_tr = RMSprop(f,
                        df,
                        np.copy(start),
                        lambda step: 0.0000002,
                        0.8)

print('RMSprop is done. Steps %d. Point (%f, %f)'%(rmsprop_tr.shape[0], *rmsprop_tr[-1]))
print('='*30)

_, adadelta_tr = adadelta(f,
                          df,
                          np.copy(start),
                          0.5)

print('Adadelta is done. Steps %d. Point (%f, %f)'%(adadelta_tr.shape[0], *adadelta_tr[-1]))
print('='*30)

_, adam_tr = adam(f,
                  df,
                  np.copy(start),
                  beta_1=0.8,
                  beta_2=0.8,
                  max_step=10e5)

print('Adam is done. Steps %d. Point (%f, %f)'%(adam_tr.shape[0], *adam_tr[-1]))
print('='*30)


# In[774]:


fig = plt.figure()
ax = plt.axes(xlim=(0, 12), ylim=(-7, 5))

x = np.linspace(0, 12)
y = np.linspace(-7, 5, 1000)
xx, yy = np.meshgrid(x, y)
f_val = f(xx, yy)
plt.contourf(xx, yy, f_val, 30)

momentum_l, = ax.plot([], [], 'r', lw=2, label='Momentum')
nesterov_l, = ax.plot([], [], 'g', lw=2, label='Nesterov')
adagrad_l, = ax.plot([], [], 'b', lw=2, label='Adagrad')
rmsprop_l, = ax.plot([], [], 'y', lw=2, label='RMSprop')
adadelta_l, = ax.plot([], [], 'm', lw=2, label='Adadelta')
adam_l, = ax.plot([], [], 'w', lw=2, label='Adam')
plt.legend()

lines = [momentum_l,
         nesterov_l,
         adagrad_l,
         rmsprop_l,
         adadelta_l,
         adam_l]

traces = [momentum_tr,
          nesterov_tr,
          adagrad_tr,
          rmsprop_tr,
          adadelta_tr,
          adam_tr]


# initialization function: plot the background of each frame
def init():
    for l in lines:
        l.set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    for l, tr in zip(lines, traces):
        l.set_data(tr[0:min(10 * i, tr.shape[0]), 0], tr[0:min(10 * i, tr.shape[0]), 1])
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=20, blit=True)

anim.save('animation.mp4', fps=120)
plt.show()


# In[528]:


full_log_loss = lambda pnt : log_loss(pnt, dig_X_train, dig_y_train)
full_log_deriv = lambda pnt : log_deriv(pnt, dig_X_train, dig_y_train)

min_adam_mnist, trace_adam_mnist = adam(full_log_loss,
                                        full_log_deriv,
                                        np.zeros((dig_X_train.shape[1], 1)),
                                        vecwise=True)


# In[529]:


print("Accuracy on train %f"%accuracy(classify(dig_X_train, min_adam_mnist), dig_y_train))
print("Accuracy on test %f"%accuracy(classify(dig_X_test, min_adam_mnist), dig_y_test))


# In[531]:


adam_loss = list(map(lambda pnt: full_log_loss(pnt), trace_adam_mnist))


# In[535]:


plt.plot(adam_loss)
plt.title('Adam\'s Logloss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid()
plt.show()

