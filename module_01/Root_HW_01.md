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

> __Content modified under Creative Commons Attribution license CC-BY
> 4.0, code under BSD 3-Clause License © 2020 R.C. Cooper__

+++

# Homework

## Problems [Part 1](./01_Interacting_with_Python.md)

1. Calculate some properties of a rectangular box that is 12.5"$\times$11"$\times$14" and weighs 31 lbs

    a. What is the volume of the box?
    
    b. What is the average density of the box?
    
    c. What is the result of the following logical operation, `volume>1000` (in inches^3)

```{code-cell} ipython3
#a - Volume
length = 12.5 #in
width = 11 #in
height = 14 #in
volume = length * width * height
print('a. The volume of the box is', volume, 'cubic inches.')

#b - Average Density
mass = 31 #lb
density = mass/volume #lb/in^3
print('b. The average density of the box is', density, 'lb/in^3.')

#c - Volume > 1000
print('c.', volume > 1000)
```

2. Use the variables given below, `str1` and `str2`, and check the following 

    a. `str1<str2`
    
    b. `str1==str2`
    
    c. `str1>str2`
    
    d. How could you force (b) to be true? [Hint](https://docs.python.org/3/library/stdtypes.html?highlight=str.lower#str.lower) or [Hint](https://docs.python.org/3/library/stdtypes.html?highlight=str.lower#str.upper)

```{code-cell} ipython3
str1 = 'Python'
str2 = 'python'
```

```{code-cell} ipython3
print('a.', str1 < str2)
print('b.', str1 == str2)
print('c.', str1 > str2)
```

```{code-cell} ipython3
#d
str1 > str.upper(str2)
```

3. The following code has an error, fix the error so that the correct result is returned:

```y is 20 and x is less than y```

```python
x="1"
y=20

if x<y and y==20:
    print('y is 20 and x is less than y')
else:
    print('x is not less than y')
```

```{code-cell} ipython3
x="1"
y=20

if int(x)<y and y==20:                       # x was changed from a string to an integer so x and y can be compared.
    print('y is 20 and x is less than y')
else:
    print('x is not less than y')
```

4. There is a commonly-used programming question that asks interviewees
   to build a [fizz-buzz](https://en.wikipedia.org/wiki/Fizz_buzz) result. 
   
   Here, you will build a similar program, but use the numbers from the
   class, **3255:** $3,~2,~5\rightarrow$ "computational", "mechanics",
   "rocks!". You should print out a list of numbers, if the number is
   divisible by 3, replace the 3 with "computational". If the number is
   divisible by 2, replace with "mechanics". If the number is divisible
   by 5, replace the number with "rocks!". If the number is divisible by
   a combination, then add both words e.g. 6 is divisible by 3 and 2, so
   you would print out "computational mechanics". 
   
   Here are the first 20 outputs your program should print, 
   
| index | printed output |
| ---   | ---            |
0 | Computational Mechanics Rocks!
1 | 1
2 | Mechanics 
3 | Computational 
4 | Mechanics 
5 | Rocks!
6 | Computational Mechanics
7 | 7
8 | Mechanics 
9 | Computational 
10 | Mechanics Rocks!
11 | 11
12 | Computational Mechanics
13 | 13
14 | Mechanics 
15 | Computational Rocks!
16 | Mechanics 
17 | 17
18 | Computational Mechanics
19 | 19

```{code-cell} ipython3
def fizz_buzz(num):
    word = str(num)
    lst = []
    for i in word:
        lst.append(int(i))

    digits = []
    for t in lst:
        if t not in digits:
            digits.append(t)
        
    sentence = []
    for j in digits:
        if j%2 == 0:
            sentence.append('Mechanics')
        elif j%3 == 0:
            sentence.append('Computational')
        elif j%4 == 0:
            sentence.append('Mechanics')
        elif j%5 == 0:
            sentence.append('Rocks!')
        elif j%6 == 0:
            sentence.append('Computational Mechanics')
        elif j%8 == 0:
            sentence.append('Mechanics')
        elif j%9 == 0:
            sentence.append('Computational')
        elif j%10 == 0:
            sentence.append('Mechanics Rocks!')
        elif j%12 == 0:
            sentence.append('Computational Mechanics')
        elif j%14 == 0:
            sentence.append('Mechanics')
        elif j%15 == 0:
            sentence.append('Computational Rocks!')
        elif j%16 == 0:
            sentence.append('Mechanics')
        elif j%18 == 0:
            sentence.append('Computational Mechanics')
            
    print(*sentence)
    
fizz_buzz(3255)
```

## Problems [Part 2](./02_Working_with_Python.md)

1. Create a function called `sincos(x)` that returns two arrays, `sinx` and `cosx` that return the sine and cosine of the input array, `x`. 

    a. Document your function with a help file in `'''help'''`
    
    b. Use your function to plot sin(x) and cos(x) for x=$0..2\pi$

```{code-cell} ipython3
import math
import numpy as np
import matplotlib.pyplot as plt

def sincos(x):
    
    '''
    sincos(x): returns the sine and cosine of the array, x
    
    Arguments
    ---------
    x: an array, on the interval from 0 to 2pi
    
    Returns
    -------
    sincos_output: the sine and cosine of the array, x'''
    
    x = np.linspace(0, 2*np.pi)
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    plt.plot(x, sin_x, color='red', linestyle='-', label='sin')
    plt.plot(x, cos_x, color='blue', linestyle='--', label='cos')
    plt.legend(loc = 'best')
```

```{code-cell} ipython3
print('2.1a.')
help(sincos)
```

```{code-cell} ipython3
print('2.1b.')
sincos(x)
```

2. Use a for-loop to create a variable called `A_99`, where every element is the product
of the two indices from 0 to 9 e.g. A_99[3,2]=6 and A_99[4,4]=16. 

    a. time your script using `%%time`    
    
    b. Calculate the mean of `A_99`

    c. Calculate the standard deviation of `A_99`

```{code-cell} ipython3
import math

print('2.2a.')
%time
A_99 = []
for i in range(10):
    for j in range(10):
        num = i * j
        A_99.append(num)
total = sum(A_99)
length = len(A_99)
mean = total/length
print('2.2b.', mean)

sd = 0
sum_A_99 = 0
for i in A_99:
    temp_sum = (i - mean)**2
    sum_A_99 += temp_sum

stand_dev = math.sqrt(sum_A_99/(len(A_99)))
    
print('2.2c.', stand_dev)
```

3. Use the two arrays, X and Y, given below to create A_99 using numpy array math rather than a for-loop.

```{code-cell} ipython3
X, Y = np.meshgrid(np.arange(10), np.arange(10))
```

    a. time your script using `%%time`    
    
    b. Calculate the mean of `A_99`

    c. Calculate the standard deviation of `A_99`
        
    d. create a filled contour plot of X, Y, A_99 [contourf plot documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html)

```{code-cell} ipython3
print('3.3a.')
%time

A_99 = X * Y

mean = np.mean(A_99)
print('3.3b.', mean)

stand_dev = np.sqrt(np.sum((A_99 - mean)**2)/A_99.size)
print('3.3c.', stand_dev)

print('3.3d.')
Z=(X**2+Y**2)**0.5
plt.contourf(X,Y,Z)
```

4. The following linear interpolation function has an error. It is supposed to return y(x) given the the two points $p_1=[x_1,~y_1]$ and $p_2=[x_2,~y_2]$. Currently, it just returns and error.

```python
def linInterp(x,p1,p2):
    '''linear interplation function
    return y(x) given the two endpoints 
    p1=np.array([x1,y1])
    and
    p2=np.array([x2,y2])'''
    slope = (p2[2]-p1[2])/(p2[1]-p1[1])
    
    return p1[2]+slope*(x - p1[1])
```

```{code-cell} ipython3
def linInterp(x,p1,p2):
    '''linear interplation function
    return y(x) given the two endpoints 
    p1=np.array([x1,y1])
    and
    p2=np.array([x2,y2])'''
    slope = (p2[1]-p1[1])/(p2[0]-p1[0])

    return p1[1]+slope*(int(x) - p1[0])

p1 = [2,3]
p2 = [5,4]

linInterp(x,p1,p2)
```

The original function had the list being indexed at the 2 position, however the given information, p1 and p2, are lists that only include 2 elements. When indexing in Python, the first element is given the position 0 rather than 1. This means that a list of 2 elements only include a 0th element and a 1st element, so indexing at 2 would be out of range for indexing.

+++

## Problems [Part 3](03_Numerical_error.md)

1. The growth of populations of organisms has many engineering and scientific applications. One of the simplest
models assumes that the rate of change of the population p is proportional to the existing population at any time t:

$\frac{dp}{dt} = k_g p$

where $t$ is time in years, and $k_g$ is growth rate in \[1/years\]. 

The world population has been increasing dramatically, let's make a prediction based upon the [following data](https://worldpopulationhistory.org/map/2020/mercator/1/0/25/) saved in [world_population_1900-2020.csv](../data/world_population_1900-2020.csv):


|year| world population |
|---|---|
|1900|1,578,000,000|
|1950|2,526,000,000|
|2000|6,127,000,000|
|2020|7,795,482,000|

a. Use a growth rate of $k_g=0.013$ [1/years] and compare the analytical solution (use initial condition p(1900) = 1578000000) to the Euler integration for time steps of 20 years from 1900 to 2020 (Hint: use method (1)- plot the two solutions together with the given data) 

b. Discussion question: If you decrease the time steps further and the solution converges, will it converge to the actual world population? Why or why not? 

**Note: We have used a new function `np.loadtxt` here. Use the `help` or `?` to learn about what this function does and how the arguments can change the output. In the next module, we will go into more details on how to load data, plot data, and present trends.**

```{code-cell} ipython3
import numpy as np
year, pop = np.loadtxt('../data/world_population_1900-2020.csv',skiprows=1,delimiter=',',unpack=True)
print('years=',year)
print('population =', pop)
```

```{code-cell} ipython3
print('average population changes 1900-1950, 1950-2000, 2000-2020')
print((pop[1:] - pop[0:-1])/(year[1:] - year[0:-1]))
print('average growth of 1900 - 2020')
print(np.mean((pop[1:] - pop[0:-1])/(year[1:] - year[0:-1])))
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

print('3.1a.')

t=np.linspace(1900,2020,10)
k=0.013

analytical=lambda year: pop[0]*np.exp(k*(year-1900))
numerical=np.zeros(len(t));
numerical[0]=pop[0];

for i in range(0, len(t)-1):
    numerical[i+1]=numerical[i]+k*numerical[i]*(t[1]-t[0])
    
plt.plot(t, analytical(t), label='analytical');
plt.plot(t, numerical, 'o-', label='numerical');
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc='best')
plt.show()

print('3.1b.')
t=np.linspace(1900,2020,7)
k=0.013

analytical=lambda year: pop[0]*np.exp(k*(year-1900))
numerical=np.zeros(len(t));
numerical[0]=pop[0];

for i in range(0, len(t)-1):
    numerical[i+1]=numerical[i]+k*numerical[i]*(t[1]-t[0])
    
plt.plot(t, analytical(t), label='analytical');
plt.plot(t, numerical, 'o-', label='numerical');
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc='best')
plt.show()
```

If you decrease the time steps further and the solution converges, it will not converge because the less data points we have the less data we have to work from. The code is making assumptions about what may come next if the trend was to continue in the same way. This means that the more data points included in the code, the more accurate the assumption will be. If data is taken away, the data will not converge to the actual world population. This can be seen in the graph above. The numerical line is

+++

__d.__ As the number of time steps increases, the Euler approximation approaches the analytical solution, not the measured data. The best-case scenario is that the Euler solution is the same as the analytical solution.

+++

2. In the freefall example you used smaller time steps to decrease the **truncation error** in our Euler approximation. Another way to decrease approximation error is to continue expanding the Taylor series. Consider the function f(x)

    $f(x)=e^x = 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\frac{x^4}{4!}+...$

    We can approximate $e^x$ as $1+x$ (first order), $1+x+x^2/2$ (second order), and so on each higher order results in smaller error. 
    
    a. Use the given `exptaylor` function to approximate the value of exp(1) with a second-order Taylor series expansion. What is the relative error compared to `np.exp(1)`?
    
    b. Time the solution for a second-order Taylor series and a tenth-order Taylor series. How long would a 100,000-order series take (approximate this, you don't have to run it)
    
    c. Plot the relative error as a function of the Taylor series expansion order from first order upwards. (Hint: use method (4) in the comparison methods from the "Truncation and roundoff error accumulation in log-log plot" figure)

```{code-cell} ipython3
from math import factorial
def exptaylor(x,n):
    '''Taylor series expansion about x=0 for the function e^x
    the full expansion follows the function
    e^x = 1+ x + x**2/2! + x**3/3! + x**4/4! + x**5/5! +...'''
    if n<1:
        print('lowest order expansion is 0 where e^x = 1')
        return 1
    else:
        ex = 1+x # define the first-order taylor series result
        for i in range(1,n):
            ex+=x**(i+1)/factorial(i+1) # add the nth-order result for each step in loop
        return ex
        
```

```{code-cell} ipython3
approximation = exptaylor(1, 2)
actual = np.exp(1)
relative_error = np.abs((actual - approximation)/actual) * 100
print('3.2a.', relative_error)

print()

%time
print()
second = exptaylor(1,2)
tenth = exptaylor(1, 10)
approx = (tenth-second)/(10-2)
time = approx * 100000
print('3.2b.', time)

print()

print('3.2c.')
n = np.arange(1, 20, 1)
N=len(n)

relative= np.zeros(N)

for i in range(1,N):
    t=exptaylor(1,n[i])
    relative[i]=np.abs((exptaylor(1,i)-np.exp(1))/np.exp(1))
    
plt.loglog(n, relative,'o')
plt.xlabel('Number of Timesteps')
plt.ylabel('Relative Error')
plt.show()
```
