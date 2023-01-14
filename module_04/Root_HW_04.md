---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Homework
## Problems [Part 1](./01_Linear-Algebra.md)

1. Consider the following pistons connected in series. You want to find
   the distance each piston moves when equal and opposite forces are
   applied to each side. In case 1, you consider the motion of all five
   pistons, but in case 2 you set the motion of piston 5 to 0. 

![Springs-masses](../images/series-springs_01-02.png)

Using a FBD for each piston, in case 1:

$k_1x_{1}-k_2x_{2}=F$

$(k_1 +k_2)x_2 -k_1x_1 - k_2x_3 = 0$

$(k_2 +k_3)x_3 -k_2x_2 - k_3x_4 = 0$

$(k_3 +k_4)x_4 -k_3x_3 - k_4x_5 = 0$

$k_4x_5 - k_4x_4 = -F$

in matrix form:

$\left[ \begin{array}{ccccc}
k_1 & -k_1 & 0 & 0 & 0\\
-k_1 & k_1+k_2 & -k_2 & 0 & 0 \\
0 & -k_2 & k_2+k_3 & -k_3 & 0\\
0 & 0 & -k_3 & k_3+k_4 & -k_4\\
0 & 0 & 0 & -k_4 & k_4  
\end{array} \right]
\left[ \begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \\
x_{5}\end{array} \right]=
\left[ \begin{array}{c}
F \\
0 \\
0 \\
0 \\
-F \end{array} \right]$

Try to use `np.linalg.solve` to find the piston x-positions. Do you get
any warnings or errors?

```{code-cell} ipython3
import numpy as np

k_1 = 100
k_2 = 100
k_3 = 100
k_4 = 100
F = 100

A=np.array([[k_1, -k_1, 0, 0, 0],
            [-k_1, k_1+k_2, -k_2, 0, 0],
            [0, -k_2, k_2+k_3, -k_3, 0], 
            [0, 0, -k_3, k_3+k_4, -k_4], 
            [0, 0, 0, -k_4, k_4]])

b = np.array([F, 0, 0, 0, -F])

x = np.linalg.solve(A,b)
```

For case 1, you do get a linear algebra error called, 'Singular Matrix'. This means that the matrix has multiple solutions. This means that there is an infinite number of number of values for x and y. 

+++

Now, consider case 2, 

Using a FBD for each piston, in case 2:

$k_1x_{1}-k_2x_{2}=F$

$(k_1 +k_2)x_2 -k_1x_1 - k_2x_3 = 0$

$(k_2 +k_3)x_3 -k_2x_2 - k_3x_4 = 0$

$(k_3 +k_4)x_4 -k_3x_3 = 0$

in matrix form:

$\left[ \begin{array}{cccc}
k_1 & -k_1 & 0 & 0 \\
-k_1 & k_1+k_2 & -k_2 & 0 \\
0 & -k_2 & k_2+k_3 & -k_3 \\
0 & 0 & -k_3 & k_3+k_4 \\
\end{array} \right]
\left[ \begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \end{array} \right]=
\left[ \begin{array}{c}
F \\
0 \\
0 \\
0 \end{array} \right]$

Try to use `np.linalg.solve` to find the piston x-positions. Do you get
any warnings or errors? Why does this solution work better [hint: check
condition
numbers](./01_Linear-Algebra.md#Singular-and-ill-conditioned-matrices)

```{code-cell} ipython3
A=np.array([[k_1, -k_1, 0, 0],
            [-k_1, k_1+k_2, -k_2, 0],
            [0, -k_2, k_2+k_3, -k_3], 
            [0, 0, -k_3, k_3+k_4]])

b = np.array([F, 0, 0, 0])

x = np.linalg.solve(A,b)


for i in range(0,4):
    print('[{:7.1f} {:7.1f} {:7.1f}]'.format(*A[i]),'*', '[{:3.1f}]'.format(x[i]), '=', '[{:5.1f}]'.format(b[i]))
```

Case 2 does work because matrix A is not singular anymore. This allow the code to determine all of the x and y values. There is no longer infinite many solutions to the set of equations, therefore the x values can be found.

+++

![HVAC diagram showing the flow rates and connections between floors](../images/hvac.png)

2. In the figure above you have an idealized Heating, Ventilation and Air conditioning (HVAC) system. In the current configuration, the three-room building is being cooled off by $12^oC$ air fed into the building at 0.1 kg/s. Our goal is to determine the steady-state temperatures of the rooms given the following information

* $\dot{m}_1=0.1~kg/s$
* $\dot{m}_2=0.12~kg/s$
* $\dot{m}_3=0.12~kg/s$
* $\dot{m}_4=0.1~kg/s$
* $\dot{m}_5=0.02~kg/s$
* $\dot{m}_6=0.02~kg/s$
* $C_p=1000~\frac{J}{kg-K}$
* $\dot{Q}_{in} = 300~W$
* $T_{in} = 12^{o} C$

The energy-balance equations for rooms 1-3 create three equations:

1. $\dot{m}_1 C_p T_{in}+\dot{Q}_{in}-\dot{m}_2 C_p T_{1}+\dot{m}_6 C_p T_{2} = 0$

2. $\dot{m}_2 C_p T_{1}+\dot{Q}_{in}+\dot{m}_5 C_p T_{3}-\dot{m}_3 C_p T_{2}-\dot{m}_6 C_p T_{2} = 0$

3. $\dot{m}_3 C_p T_{2}+\dot{Q}_{in}-\dot{m}_5 C_p T_{3}-\dot{m}_4 C_p T_{3} = 0$

Identify the unknown variables and constants to create a linear algebra problem in the form of $\mathbf{Ax}=\mathbf{b}$.

a. Create the matrix $\mathbf{A}$

b. Create the known vector $\mathbf{b}$

c. Solve for the unknown variables, $\mathbf{x}$

d. What are the warmest and coldest rooms? What are their temperatures?

```{code-cell} ipython3
m_1 = 0.1
m_2 = 0.12
m_3 = 0.12
m_4 = 0.1
m_5 = 0.02
m_6 = 0.02
C_p = 1000
Q_in = 300
T_in = 12
```

```{code-cell} ipython3
print('1.2a')

A = np.array([[-m_2*C_p, m_6*C_p, 0],
              [m_2*C_p, (-m_3*C_p)+(-m_6*C_p), m_5*C_p],
              [0, m_3*C_p, (-m_5*C_p)+(-m_4*C_p)]])
print(A)
```

```{code-cell} ipython3
print('1.2b')

b = np.array([(-m_1*C_p*T_in) - Q_in, -Q_in, -Q_in])
print(b)
```

```{code-cell} ipython3
print('1.2c')

x = np.linalg.solve(A, b)
print('The unknown x variables are:\nx1 = {:.2f}\nx2 = {:.2f}\nx3 = {:.2f}'.format(x[0], x[1], x[2]))
```

```{code-cell} ipython3
print('1.2d')

x_max = np.max(x)
x_min = np.min(x)

print('The warmest room is {} degrees C and the coldest room is {:.2f} degrees C.'.format(x_max, x_min))
```

3. The [Hilbert Matrix](https://en.wikipedia.org/wiki/Hilbert_matrix) has a high condition number and as the matrix increases dimensions, the condition number increases. Find the condition number of a 

a. $1 \times 1$ Hilbert matrix

b. $5 \times 5$ Hilbert matrix

c. $10 \times 10$ Hilbert matrix

d. $15 \times 15$ Hilbert matrix

e. $20 \times 20$ Hilbert matrix

If the accuracy of each matrix element is $\approx 10^{-16}$, what is the expected rounding error in the solution $\mathbf{Ax} = \mathbf{b}$, where $\mathbf{A}$ is the Hilbert matrix.

```{code-cell} ipython3
print('1.3a\n')

N=1
A_1=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        A_1[i,j]=1/(i+j+1)
print(A_1[-3:,-3:])

np.linalg.cond(A_1)
```

```{code-cell} ipython3
error_1 = 10**(0 - 16) # 0 because the 1 has a power of 0
print('Error for 1x1 Hibert Matrix:', error_1)
```

```{code-cell} ipython3
print('1.3b\n')

N=5
A_5=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        A_5[i,j]=1/(i+j+1)

print(A_5)

np.linalg.cond(A_5)
```

```{code-cell} ipython3
error_5 = 10**(5 - 16)
print('Error for 5x5 Hibert Matrix:', error_5)
```

```{code-cell} ipython3
print('1.3c\n')

N=10
A_10=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        A_10[i,j]=1/(i+j+1)
print(A_10)

np.linalg.cond(A_10)
```

```{code-cell} ipython3
error_10 = 10**(13 - 16) 
print('Error for 10x10 Hibert Matrix:', error_10)
```

```{code-cell} ipython3
print('1.3d\n')

N=15
A_15=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        A_15[i,j]=1/(i+j+1)
print(A_15)

np.linalg.cond(A_15)
```

```{code-cell} ipython3
error_15 = 10**(17 - 16) 
print('Error for 15x15 Hibert Matrix:', error_15)
```

```{code-cell} ipython3
print('1.3d\n')

N=20
A_20=np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        A_20[i,j]=1/(i+j+1)
print(A_20)

np.linalg.cond(A_20)
```

```{code-cell} ipython3
error_20 = 10**(18 - 16) 
print('Error for 20x20 Hibert Matrix:', error_20)
```

## Problems [Part 2](./02_Gauss_elimination.md)

1. 4 masses are connected in series to 4 springs with K=100N/m. What are the final positions of the masses? 

![Springs-masses](../images/mass_springs.png)

The masses haves the following amounts, 1, 2, 3, and 4 kg for masses 1-4. Using a FBD for each mass:

$m_{1}g+k(x_{2}-x_{1})-kx_{1}=0$

$m_{2}g+k(x_{3}-x_{2})-k(x_{2}-x_{1})=0$

$m_{3}g+k(x_{4}-x_{3})-k(x_{3}-x_{2})=0$

$m_{4}g-k(x_{4}-x_{3})=0$

in matrix form K=100 N/m:

$\left[ \begin{array}{cccc}
2k & -k & 0 & 0 \\
-k & 2k & -k & 0 \\
0 & -k & 2k & -k \\
0 & 0 & -k & k \end{array} \right]
\left[ \begin{array}{c}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \end{array} \right]=
\left[ \begin{array}{c}
m_{1}g \\
m_{2}g \\
m_{3}g \\
m_{4}g \end{array} \right]$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
k = 100
m1 = 1
m2 = 2
m3 = 3
m4 = 4
g = 9.81

K = np.array([[2*k, -k, 0, 0], 
              [-k, 2*k, -k, 0],
              [0, -k, 2*k, -k],
              [0, 0, -k, k]])

m = g*np.array([m1, m2, m3, m4])
```

```{code-cell} ipython3
def GaussNaive(A,y):
    '''GaussNaive: naive Gauss elimination
    x = GaussNaive(A,b): Gauss elimination without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    y = right hand side vector
    returns:
    ---------
    x = solution vector
    Aug = augmented matrix (used for back substitution)'''
    [m,n] = np.shape(A)
    Aug = np.block([A,y.reshape(n,1)])
    Aug = Aug.astype(float)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination 
    for k in range(0,n-1):
        for i in range(k+1,n):
            if Aug[i,k] != 0.0:
                factor = Aug[i,k]/Aug[k,k]
                Aug[i,:] = Aug[i,:] - factor*Aug[k,:]
    # Back substitution
    x=np.zeros(n)
    for k in range(n-1,-1,-1):
        x[k] = (Aug[k,-1] - Aug[k,k+1:n]@x[k+1:n])/Aug[k,k]
    return x,Aug
```

```{code-cell} ipython3
print('2.1\n')

x, Aug = GaussNaive(K,m)
print(Aug[0:4, 0:4])
print('Final positions for each mass:\nx1 = {:.3f} m\nx2 = {:.3f} m\nx3 = {:.3f} m\nx4 = {:.3f} m'.format(x[0], x[1], x[2], x[3]))
```

![Triangular truss](../images/truss.png)

For problems __2-3__, consider the simple 3-element triangular truss, shown above, with a point load applied at the tip. The goal is to understand what tension is in the horizontal element, $P_1$. In problem __2__, the applied force is verical $(\theta=0)$ and in problem __3__ the applied force varies in angle $(\theta \neq 0)$. 

2. In the truss shown above, calculate the tension in bar 1, $P_1$, when $\theta=0$. When $\theta=0$, the $\sum F=0$ at each corner creates 3 equations and 3 unknowns as such (here, you reduce the number of equations with symmetry, $P_2=P_3,~R_2=R_3,~and~R_1=0$ ). 

$\left[ \begin{array}{ccc}
1 & \cos\alpha & 0 \\
0 & 2\sin\alpha & 0 \\
0 & -\sin\alpha &  1 \\
 \end{array} \right]
\left[ \begin{array}{c}
P_{1} \\
P_{2} \\
R_{2} \end{array} \right]=
\left[ \begin{array}{c}
0 \\
F \\
0 \end{array} \right]$

a. Create the system of equations, $\mathbf{Ax}=\mathbf{b}$, when $\alpha=75^o$, $\beta=30^o$, and $F=1~kN$. Use __Gauss elimination__ to solve for $P_1,~P_2,~and~R_2$. What is the resulting augmented matrix, $\mathbf{A|y}$ after Gauss elimination?

b. Solve for the $\mathbf{LU}$ decomposition of $\mathbf{A}$. 

c. Use the $\mathbf{LU}$ solution to solve for the tension in bar 1 $(P_1)$ every 10 N values of force, F, between 100 N and 1100 N. Plot $P_1~vs~F$.

```{code-cell} ipython3
def GaussNaive(A,y):
    '''GaussNaive: naive Gauss elimination
    x = GaussNaive(A,b): Gauss elimination without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    y = right hand side vector
    returns:
    ---------
    x = solution vector
    Aug = augmented matrix (used for back substitution)'''
    [m,n] = np.shape(A)
    Aug = np.block([A,y.reshape(n,1)])
    Aug = Aug.astype(float)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination 
    for k in range(0,n-1):
        for i in range(k+1,n):
            if Aug[i,k] != 0.0:
                factor = Aug[i,k]/Aug[k,k]
                Aug[i,:] = Aug[i,:] - factor*Aug[k,:]
    # Back substitution
    x=np.zeros(n)
    for k in range(n-1,-1,-1):
        x[k] = (Aug[k,-1] - Aug[k,k+1:n]@x[k+1:n])/Aug[k,k]
    return x,Aug
```

```{code-cell} ipython3
alpha = 75 * np.pi/180
beta = 30 * np.pi/180
F = 1000 #N

LHS = np.array([[1, np.cos(alpha), 0],
              [0, 2*np.cos(alpha), 0],
              [0, -np.sin(alpha), 1]])

RHS = np.array([0, F, 0])

values, A_y = GaussNaive(LHS, RHS)

print('P1 = {:.3f}\nP2 = {:.3f}\nR2 = {:.3f}'.format(values[0], values[1], values[2]))

print('The resulting augmented matrix, A|y after Gauss elimination is:\n', A_y)
```

```{code-cell} ipython3
def LUNaive(A):
    '''LUNaive: naive LU decomposition
    L,U = LUNaive(A): LU decomposition without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    returns:
    ---------
    L = Lower triangular matrix
    U = Upper triangular matrix
    '''
    [m,n] = np.shape(A)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination
    U = A.astype(float)
    L = np.eye(n)

    for k in range(0,n-1):
        for i in range(k+1,n):
            if U[k,k] != 0.0:
                factor = U[i,k]/U[k,k]
                L[i,k]=factor
                U[i,:] = U[i,:] - factor*U[k,:]
    return L,U
```

```{code-cell} ipython3
print('2.2b\n')

L, U = LUNaive(LHS)

print('L = {}\n\nU = {}'.format(L, U))
```

```{code-cell} ipython3
print('2.2c\n')

from numpy.random import default_rng
rng = default_rng()

F_space = np.zeros(1000)
x1_space = np.zeros(1000)

for i in range(1000):
    F = 1000*rng.random(3)
    y = np.linalg.solve(L, F)
    x = np.linalg.solve(U, y)
    F_space[i] = F[1]
    x1_space[i] = x[0]
    
plt.plot(x1_space, F_space, 's')  
plt.ylabel('F (N)')
plt.xlabel('P1 (N)')
plt.title('F vs P1')
```

3. Using the same truss as shown above, let's calculate the tension in bar 1, $P_1$, when $\theta=[0...90^o]$ and $F=[100...1100]~kN$. When $\theta\neq 0$, the resulting 6 equations and 6 unknowns are given in the following matrix

$\left[ \begin{array}{ccc}
0 & \sin\alpha & 0 & 1 & 0 & 0 \\
1 & \cos\alpha & 0 & 0 & 1 & 0 \\
0 & \cos\beta/2 & \cos\beta/2 & 0 & 0 & 0 \\
0 & -\sin\beta/2 & \sin\beta/2 & 0 & 0 & 0 \\
-1 & 0 & \cos\alpha & 0 & 0 & 0 \\
0 & 0 & \sin\alpha & 0 & 0 & 1 \\
 \end{array} \right]
\left[ \begin{array}{c}
P_{1} \\
P_{2} \\
P_{3} \\
R_{1} \\
R_{2} \\
R_{3}\end{array} \right]=
\left[ \begin{array}{c}
0 \\
0 \\
F\cos\theta \\
-F\sin\theta \\
0 \\
0 \end{array} \right]$

a. Create the system of equations, $\mathbf{Ax}=\mathbf{b}$, when $\alpha=75^o$, $\beta=30^o$, $\theta=45^o=\pi/4$, and $F=1~kN$. Use __Gauss elimination__ to solve for $P_1,~P_2,~P_3,~R_1,~R_2,~and~R_3$. What is the resulting augmented matrix, $\mathbf{A|y}$ after Gauss elimination? _Hint: do you see a problem with the order of the equations? How can you __pivot__ to fix it?_

b. Solve for the $\mathbf{PLU}$ decomposition of $\mathbf{A}$. 

c. Use the $\mathbf{PLU}$ solution to solve for the tension in bar 1 $(P_1)$ every 10 N values of force, F, between 100 N and 1100 N. Plot $P_1~vs~F$.

```{code-cell} ipython3
print('2.3a.')

a = 75 * np.pi/180 # radians
b = 30 * np.pi/180
theta = 45 * np.pi/180
F = 1000 # N

A = np.array([[0, np.sin(a), 0, 1, 0, 0],
             [1, np.cos(a), 0, 0, 1, 0],
             [0, np.cos(b/2), np.cos(b/2), 0, 0, 0],
             [0, -np.sin(b/2), np.sin(b/2), 0, 0, 0],
             [-1, 0, np.cos(a), 0, 0, 0],
             [0, 0, np.sin(a), 0, 0, 1]])

b = np.array([[0, 0, F*np.cos(theta), -F*np.sin(theta), 0, 0]])

GaussNaive(A, b)
```

You can't use the Gauss Naive method to solve for the unknown variables in this equation because it does not use pivoting, which means that it doesn't look for cases where you divide by zero. To fix this, we can swap the first row with some other row that does not have 0 in the first position. I switched the first row with the second row.

```{code-cell} ipython3
a = 75 * np.pi/180 # radians
b = 30 * np.pi/180
theta = np.pi/4
F = 1000 # N

A = np.array([[1, np.cos(a), 0, 0, 1, 0],
              [0, np.sin(a), 0, 1, 0, 0],
              [0, np.cos(b/2), np.cos(b/2), 0, 0, 0],
              [0, -np.sin(b/2), np.sin(b/2), 0, 0, 0],
              [-1, 0, np.cos(a), 0, 0, 0],
              [0, 0, np.sin(a), 0, 0, 1]])

b = np.array([[0, 0, F*np.cos(theta), -F*np.sin(theta), 0, 0]])

values, A_y = GaussNaive(A, b)

print('P1 = {:.3f}\nP2 = {:.3f}\nP3 = {:.3f}\nR1 = {:.3f}\nR2 = {:.3f}\nR3 = {:.3f}'.format(values[0], values[1], values[2], values[3], values[4], values[5]))

print('The resulting augmented matrix, A|y after Gauss elimination is:\n', A_y)
```

```{code-cell} ipython3
from scipy.linalg import lu, solve
```

```{code-cell} ipython3
def solveLU(L,U,b):
    '''solveLU: solve for x when LUx = b
    x = solveLU(L,U,b): solves for x given the lower and upper 
    triangular matrix storage
    uses forward substitution for 
    1. Ly = b
    then backward substitution for
    2. Ux = y
    
    Arguments:
    ----------
    L = Lower triangular matrix
    U = Upper triangular matrix
    b = output vector
    
    returns:
    ---------
    x = solution of LUx=b '''
    n=len(b)
    x=np.zeros(n)
    y=np.zeros(n)
        
    # forward substitution
    for k in range(0,n):
        y[k] = b[k] - L[k,0:k]@y[0:k]
    # backward substitution
    for k in range(n-1,-1,-1):
        x[k] = (y[k] - U[k,k+1:n]@x[k+1:n])/U[k,k]
    return x
```

```{code-cell} ipython3
print('2.3b\n')

P,L,U = lu(A) # a built-in partial-pivoting LU decomposition function
print('P=\n',P)
print('L=\n',L)
print('U=\n',U)
```

```{code-cell} ipython3
print('2.3c\n')

from numpy.random import default_rng
rng = default_rng()

Fspace = np.zeros(1000)
xspace = np.zeros(1000)

for i in range(1000):
    F = 1000*rng.random(6)
    y = np.linalg.solve(L, F)
    x = np.linalg.solve(U, y)
    Fspace[i] = F[1]
    xspace[i] = x[0]
    
plt.plot(xspace, Fspace, 's')  
plt.ylabel('F (N)')
plt.xlabel('P1 (N)')
plt.title('P1 vs. F')
```

## Problems [Part 3](./03_Linear-regression-algebra.md)

<img
src="https://i.imgur.com/LoBbHaM.png" alt="prony series diagram"
style="width: 300px;"/> <img src="https://i.imgur.com/8i140Zu.png" alt
= "stress relax data" style="width: 400px;"/> 

Viscoelastic Prony series model and stress-vs-time relaxation curve of wheat kernels [[3]](https://www.cerealsgrains.org/publications/plexus/cfw/pastissues/2013/Documents/CFW-58-3-0139.pdf). Stress relaxation curve of a wheat kernel from regressed equation data that illustrate where to locate relaxation times (vertical dotted lines) and stresses (horizontal black marks). $\sigma$ = stress; t = time.

2. [Viscoelasticity](https://en.wikipedia.org/wiki/Viscoelasticity) is a property of materials that exhibit stiffness, but also tend to flow slowly. One example is [Silly Putty](https://en.wikipedia.org/wiki/Silly_Putty), when you throw a lump it bounces, but if you leave it on a table it _creeps_, slowly flowing downwards. In the stress-vs-time plot above, a wheat kernel was placed under constant strain and the stress was recorded. In a purely elastic material, the stress would be constant. In a purely viscous material, the stress would decay to 0 MPa. 

Here, you have a viscoelastic material, so there is some residual elastic stress as $t\rightarrow \infty$. The researchers used a 4-part [Prony series](https://en.wikipedia.org/wiki/Prony%27s_method) to model viscoelasticity. The function they fit was

$\sigma(t) = a_1 e^{-t/1.78}+a_2 e^{-t/11}+a_3e^{-t/53}+a_4e^{-t/411}+a_5$

a. Load the data from the graph shown above in the file `../data/stress_relax.dat`. 

b. Create a $\mathbf{Z}$-matrix to perform the least-squares regression for the given Prony series equation $\mathbf{y} = \mathbf{Za}$.

c. Solve for the constants, $a_1,~a_2,~a_3,~a_4~,a_5$

d. Plot the best-fit function and the data from `../data/stress_relax.dat` _Use at least 50 points in time to get a smooth best-fit line._

```{code-cell} ipython3
print('3.2a\n')

! head ../data/stress_relax.dat

data = np.loadtxt('../data/stress_relax.dat', skiprows = 1, delimiter = ',')
```

```{code-cell} ipython3
print('3.2b\n')
 
time = data[:, 0]
stress = data[:, 1]

Z = np.block([[np.exp(-time/1.78)], [np.exp(-time/11)], [np.exp(-time/53)], [np.exp(-time/411)], [time**0]]).T

print(Z)
```

```{code-cell} ipython3
print('3.2c\n')

C = np.linalg.solve(Z.T@Z, Z.T@stress)
for i in range(0, 5):
    print('a{} = {:.4f}'.format(i, C[i]))
```

```{code-cell} ipython3
print('3.2d\n')

plt.plot(time, stress, 's')
plt.plot(time, Z@C)
plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.title('Stress vs. Time')
```

3. Load the '../data/primary-energy-consumption-by-region.csv' that has the energy consumption of different regions of the world from 1965 until 2018 [Our world in Data](https://ourworldindata.org/energy). 
You are going to compare the energy consumption of the United States to all of Europe. Load the data into a pandas dataframe. *Note: you can get certain rows of the data frame by specifying what you're looking for e.g. 
`EUR = dataframe[dataframe['Entity']=='Europe']` will give us all the rows from Europe's energy consumption.*

a. Use a piecewise least-squares regression to find a function for the energy consumption as a function of year

energy consumed = $f(t) = At+B+C(t-1970)H(t-1970)$

c. What is your prediction for US energy use in 2025? How about European energy use in 2025?

```{code-cell} ipython3
import pandas as pd

dataframe = pd.read_csv('../data/primary-energy-consumption-by-region.csv')

US = dataframe[dataframe['Entity']=='United States']
yUS = US['Year'].values
EUS = US['Primary Energy Consumption (terawatt-hours)'].values

EUR = dataframe[dataframe['Entity']=='Europe']
yEUR = EUR['Year'].values
EEUR = EUR['Primary Energy Consumption (terawatt-hours)'].values
```

```{code-cell} ipython3
ZUS = np.block([[yUS], [yUS**0], [(yUS >= 1970)*(yUS - 1970)]]).T
ZUS
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
plt.plot(yUS, EUS, 's')
coeffs_US = np.linalg.solve(ZUS.T@Z, ZUS.T@EUS)
text_US = np.arange(1960, 2030)
Zext_US = np.block([[text_US], [text_US**0], [(text_US >= 1970)*(text_US - 1970)]]).T
plt.plot(text, Zext_US@coeffs)
plt.title('US Energy Use per Year')

coeffs_US

#print('f(t) = {}t + {} + {}(t - 1970){}(t - 1970))'.format())
```

```{code-cell} ipython3
ZEUR = np.block([[yEUR], [yEUR**0], [(yEUR >= 1970)*(yEUR - 1970)]]).T

plt.plot(yEUR, EEUR, 's')

coeffs_EUR = np.linalg.solve(ZEUR.T@ZEUR, ZEUR.T@EEUR)
text_EUR = np.arange(1960, 2030)
Zext_EUR = np.block([[text_EUR], [text_EUR**0], [(text_EUR >= 1970)*(text_EUR - 1970)]]).T
plt.plot(text_EUR, Zext_EUR@coeffs_EUR)
plt.title('EUR Energy Use per Year')
```

```{code-cell} ipython3
print('3.3c')

US_2025 = np.array([2025, 1, 2025-1970])@coeffs_US
EUR_2025 = np.array([2025, 1, 2025-1970])@coeffs_EUR

print('Using the data above, I predict the US energy use in 2025 will be {:.2f} terawatt-hours.'.format(US_2025))
print('Using the data above, I predict the EUR energy use in 2025 will be {:.2f} terawatt-hours.'.format(EUR_2025))
```
