#!/usr/bin/env python
# coding: utf-8

# # Practical introduction to Python and Numpy
# This part of the exercises aims to introduce you to a bit more of the Python and Numpy syntax and give you some simple guidelines for working effectively with arrays (which is what images are).
# 
# Tasks you have to perform are marked as **Task (x)**.
# 
# *Note: Run each cell as you read along. If a cell is incomplete, it is part of some exercise as described in the text.*

# ## 2.1.1 Using Python to implement basic linear algebra functions
# We start by defining vectors $v_a$, $v_b$ as Python lists:

# In[ ]:


va = [2, 2]
vb = [3, 4]

# The euclidean length of a vector is defined 
# $$||v|| = \sqrt{\sum_{i=1}^N v_i^2}.$$
# 
# ### Task (A)
# 1. Implement this as a Python function in the code below.
# 2. Test the result on vectors $v_a$ and $v_b$ and verify by hand.
# 
# **Hints:** 
# * For-loops in python work like for-each loops in Java, i.e. they loop through the elements of an iterator and takes the current iterator value as the iteration variable.
# * Python has a `range(x)` function which returns an iterator of integers from $0,\dots, x$.
# * The size of a list can be found using the `len(l)` function.
# * Exponentiation in python works using the `**` operator. For square root, use `x**(1/2)`.
# * Remember to use Python's built in `help(<function/class/method>)` function for additional documentation. In Jupyter Lab, you can also open a documentation popover by pressing placing the text cursor on the desired symbol and pressing **Shift + Tab**.

# In[ ]:


def norm(v):
    res = 0
    for i in range(len(v)):
        res = res + v[i] ** 2
            
    return res ** (1/2)

print('a', norm(va))
print('b', norm(vb))


# Using loops for list iteration requires quite a lot of boilerplate code. Luckily, Python's *list comprehensions* are created exactly for making list iteration more expressive and easier to understand.
# 
# A list comprehension has the following form
# ```python
# [f(e) for e in list]
# ```
# where $f$ is an arbitrary function applied to each element $e$. For people familiar with functional programming, this is equivalent to the `map` function. *Note: List comprehensions can also include guard rules. You can read more about list comprehensions [here](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions).*
# 
# Python also provides a wealth of utility functions for performing common list operations. One such function is
# ```python
# sum(l)
# ```
# which sums all elements in the list argument.
# 

# ### Task (B)
# 1. Implement the euclidean length function below by using a list comprehension and sum function.
#     - First exponentiate each element in the list comprehension, resulting in a new list of exponentiated values.
#     - Then use the sum function to add all elements together and finally remember the square root.
# 2. Test the result as before.

# In[ ]:


def norm2(v):
    return sum([vi ** 2 for vi in v]) ** (1/2)
    
print('a', norm2(va))
print('b', norm2(vb))


# If execution “falls off the end” of a Python function without a return statement, it's essentially treated as if the function ended with: return None.

# Your next task is to produce a dot product given two vectors. For reference, the dot product for vectors is defined as:
# $$
# dot(a, b) = a\bullet b = \sum_{i=1}^N a_ib_i.
# $$
# 
# The dot product is an algebraic operation which takes two equal-sized vectors and returns a single scalar (which is why it is sometimes referred to as the scalar product).
# 
# ### Task (C)
# 1. Finish the function "dot" below by implementing the equation for dot product using either for-loops or list comprehensions.
#     - *Note: If you want to use list comprehensions you need the function `zip` which interleaves two iterators. It is equivalent to most functional language zip functions. Documentation can be found [here](https://docs.python.org/3/library/functions.html#zip)*
# 2. Test the implementation on $v_a$ and $v_b$. Verify by hand!

# In[ ]:


def dot(a, b):
    if len(a) == len(b):
        res = 0
        for i in range(len(a)):
            res = res + a[i] * b[i] 
            
        return res    
    else:
        return "The vectors must have equal-length"
            
dot(va, vb)


# In[ ]:


def dot2(a,b):
    if len(a) == len(b):
        return sum(a_i*b_i for a_i, b_i in zip(a, b))
    else:
        return "The vectors must have equal-length"


dot2(va, vb)


# Finally, we need to implement matrix multiplication. For an $N\times D$ matrix $A$ and a $D\times K$ matrix $B$, the matrix multiplication (or matrix product) is a new $N\times K$ matrix $R$. Elements $R_{ij}$ of $R$ can be calculated using the following formula
# $$
# R_{ij} = \sum_{d=1}^D A_{id}B_{dj}.
# $$
# In other words, it is the dot product of the $i$'th row vector of $A$ and the $j$'th column vector of $B$.
# 
# ### Task (D)
# 1. We provided a possible implementation of matrix multiplication. Make sure that you understand what's going on, especially concerning the actual result calculation (for loops).
# 2. Create sample matrices $m_a$ and $m_b$ by filling out the code stubs below. The sizes aren't important as long as they are valid for multiplication. *Hint: You can simply nest lists in Python*.
# 3. Verify the implementation result by hand calculation.

# In[ ]:


ma = [ [1,5],
       [45,87],
       [475,8]
     ]

mb = [ [1,5,9,5,22],
       [45,87,10,2,0]
     ]
 
maa = [ [1,5],
       [45,87]
     ]

mbb = [ [1,5],
       [45,87]
     ]
 
outtest = [[0 for _ in range(len(mb[0]))] for _ in range(len(ma))]
outtest


# In[ ]:


def matmul(a, b):
    # Check for valid matrix sizes
    if len(a[0]) != len(b):
        raise ValueError(f'Matrices of size ({len(a), len(a[0])}) and ({len(b), len(b[0])}) are not compatible')
        
    # D = inner_size
    inner_size = len(b)
    
    
    
    # The NxK output matrix, initialised to 0s
    # It works only for square matrix
    #out = [[0 for _ in range(inner_size)] for _ in range(inner_size)] 
    
    out = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    
    for row_a in range(len(a)):
        for col_b in range(len(b[0])):
            
            # Application of the formula
            out[row_a][col_b] = 0
            for i in range(inner_size):
                
                out[row_a][col_b] += a[row_a][i] * b[i][col_b]
    return out
    
matmul(maa, mbb)


# ## Introducing Numpy
# 
# Numpy makes it way easier to work with multidimensional arrays and provides a significant performance increase as well. We start by showing how Numpy is used to do simple array operations.
# 
# The following code imports the `numpy` package and creates a $3\times 3$ matrix:
# 
# *Note the import statement renaming `numpy` to `np`. This is commonly done in Python to avoid namespace confusion.*

# In[ ]:


import numpy as np

vec = np.array([
    [1, 2, 3],
    [3, 4, 9],
    [5, 7, 3]
])


# To check the size of an array use `a.shape`. This works on all numpy arrays, e.g. `(vec*2).shape` works as well (we will return to array operations later in this exercise). 
# 
# Print the shape of `vec` below and verify that it corresponds to the expected shape.

# In[ ]:


vec.shape


# Slicing in Python enables the selection of more than one array element. Instead of specifying a single number, e.g. `0` a range can be specified using the notation `<start>:<stop>`, e.g. `0:2`. Check the code below for a few examples:
# 
#   https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
# 

# ***Note*** that element in built-in array can be accessed ***only*** as follow:
# 
# vec = [....]
# 
#   vec[***row***] [column] ***Not*** vec[row,column]
#    

# In[ ]:


single = vec[0]
print('single', single)

vector = vec[:2, 1] # 0's can be ommitted.
print('vector', vector)

matrix = vec[:, :2]
print('matrix\n', matrix)


# It is also possible to use negative indices. These are equivalent to counting from the end of the array, i.e. `-<idx>` $\equiv$ `len(a)-<idx>`. A few examples:

# In[ ]:


single = vec[-1, -1]
print('single', single)

arange = vec[0:-2, 0:-1]
print('arange', arange)


# ### Task (E)
# 1. Create variable `ur` as a 2x2 matrix of the upper right corner of `vec` using slicing.
# 2. Create variable `row` as the 2nd row of `vec`
# 3. Create variable `col` as the 1st column of `vec`

# In[ ]:


ur = vec[:2, :2]
row = vec[1,:]
col = vec[:, 0]

print('upper right', ur)
print('row', row)
print('column', col)


# ## 2.1.2 Using Numpy for array operations
# While these implementations seem fine for our sample inputs, they quickly become unbearingly slow when increasing the size of the operands. For example, the matrix multiplication runs in time $O(NKD)$ which is $O(N^3)$ if $N=K=D$. 
# 
# Let's try an example. The code below uses numpy to generate $100\times 100$ matrices of random numbers:

# In[ ]:


ta = np.random.randint(100, size=(100, 100))
tb = np.random.randint(100, size=(100, 100))


# Jupyter Lab provides special command `%timeit <statement>` which runs a performance test on the supplied statement. This way, we can test how fast our matrix multiplication code is:

# In[ ]:


get_ipython().run_line_magic('timeit', 'matmul(ta, tb)')


# Not very fast huh? Now let's try using numpy's built in matrix multiplication function `np.dot` instead.

# In[ ]:


get_ipython().run_line_magic('timeit', 'np.dot(ta, tb)')


# That is approximately 1500 times faster than the native version (on the test computer, anyway)!!. What about other list operations? Lets try the sum function:

# In[ ]:


get_ipython().run_line_magic('timeit', 'sum(ta)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'np.sum(ta)')


# Also a clear improvement. Because of its speed, Numpy should always be used instead of native Python wherever possible.

# ## 2.1.3 Adapting Python code to Numpy
# 
# Although simple multiplications and sums are easier to perform using Numpy functions than native Python, the other is true in a lot of situations where more specific processing is needed.
# 
# Let's start by adapting our `norm` function to numpy. Numpy supports many elementwise operations, all implemented as overloads of the traditional operators. To lift all elements of a numpy array to the $n$'th power, simply use the `**` operator.
# 
# ### Task (F)
# 1. Implement norm_np without using loops or list comprehensions. Numpy has a sum function `np.sum` you can use.
# 2. Test it on the provided vector

# In[ ]:


def norm_np(v):
    return np.sum(np.power(v,2)) ** (1/2)
        


# In[ ]:


vec = np.array([2, 3, 4, 5])
norm_np(vec)


# Again, we compare the Python and Numpy implementations using an array of random numbers:

# In[ ]:


vr = np.random.randint(100, size=10000)


# In[ ]:


get_ipython().run_line_magic('timeit', 'norm_np(vr)')
get_ipython().run_line_magic('timeit', 'norm(vr)')
get_ipython().run_line_magic('timeit', 'norm2(vr)')


# Once again a huge difference between Numpy and Python (with the Python loop and list-comprehension versions being approximately equal). 

# ## 2.1.4 Useful Numpy tricks
# We end this exercise by introducing a few essential Numpy tricks that will make coding with Numpy much easier (and faster).
# 
# ### Comparison operators
# Just as the elementwise arithmetic operators, Numpy implements elementwise comparison operators. For example, if we wanted to find elements of `vr` larger than $98$ we can use the following code:

# In[ ]:


vr > 98


# In itself this isn't super useful. Luckely, Numpy supports a special indexing mode using an array of booleans to indicate which elements to keep. 
# 
# To get the actual values we simply insert the comparison code into the index operator for the array:

# In[ ]:


vr[vr > 98]


# Finally, we can combine boolean arrays by using the logical operators `&` and `|`

# In[ ]:


vr[(vr < 2) | (vr > 98)]


# These tricks also work for assignment:

# In[ ]:


vr[vr > 50] = 0
vr[:10]


# ### Operate along dimensions
# You will often create operations on arrays that contain multiple instances grouped together. For example, in machine learning, you often concatenate input vectors into matrices. In these instances you may want to perform operations along only one or some of the array axes.
# 
# As an example, let's try to calculate the average of $N$ random vectors. We define an $N\times K$ matrix of random values:

# In[ ]:


N, K = 20, 10
r = np.random.uniform(size=(N, K))


# Numpy provides a function `np.mean` for calculating averages. Using the `axis` argument, we specify that the average should be calculated over the rows:

# In[ ]:


np.mean(r, axis=0)


# The `axis` argument is supported by most of Numpy's functions, including `sum` and `sqrt`.
