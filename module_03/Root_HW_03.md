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

# Homework

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

## Problems [Part 1](./01_Catch_Motion.md)

1. Instead of using $\frac{\Delta v}{\Delta t}$, you can use the [numpy polyfit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) to determine the acceleration of the ball. 

    a. Use your coordinates from the saved .npz file you used above to load your projectile motion data
    
    ```python
    npz_coords = np.load('projectile_coords.npz')
    t = npz_coords['t']
    x = npz_coords['x']
    y = npz_coords['y']```
    
    b. Calculate $v_x$ and $v_y$ using a finite difference again, then do a first-order polyfit to $v_x-$ and $v_y-$ vs $t$. What is the acceleration now?
    
    c. Now, use a second-order polynomial fit for x- and y- vs t. What is acceleration now?
    
    d. Plot the polyfit lines for velocity and position (2 figures) with the finite difference velocity data points and positions. Which lines look like better e.g. which line fits the data?

```{code-cell} ipython3
print('1.1a.')
```

```{code-cell} ipython3
npz_coords = np.load('../data/projectile_coords.npz')
t = npz_coords['t']
x = npz_coords['x']
y = npz_coords['y']
```

```{code-cell} ipython3
plt.plot(x, y)
```

```{code-cell} ipython3
print('1.1b.')

delta_t = t[1] - t[0]

v_x = np.zeros(len(x))
v_y = np.zeros(len(y))

for i in range(len(t)-1):
    v_y[i] = (y[i+1] - y[i])/dt
    v_x[i] = (x[i+1] - x[i])/dt

plt.plot(t[0:-1], v_x[0:-1], label = '$v_x$ vs. Time')
plt.plot(t[0:-1], v_y[0:-1], label = '$v_y$ vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (cm/s)')
plt.legend(bbox_to_anchor = (1,1))

m_vx, b_vx = np.polyfit(t[0:-1], v_x[0:-1], 1)
f_linear_x = np.poly1d((m_vx, b_vx))

m_vy, b_vy = np.polyfit(t[0:-1], v_y[0:-1], 1)
f_linear_y = np.poly1d((m_vy, b_vy))

plt.plot(t, f_linear_x(t), '--k', linewidth=3)
plt.plot(t, f_linear_y(t), '--k', linewidth=3)

v_y = v_y[0:-1]
v_x = v_x[0:-1]

a_x = ((v_x[1:] - v_x[:-1]) / dt)/10
a_y = ((v_y[1:] - v_y[:-1]) / dt)/10

print('v_x:', v_x)
print('v_y:', v_y)
print()
print('The acceleration in the x direction is: {:.2f}'.format(a_x.mean()))
print('The acceleration in the y direction is: {:.2f}'.format(a_y.mean()))
```

```{code-cell} ipython3
print('1.1c.')

ppx = np.polyfit(t[0:-1], (x[0:-1])/10, 2)
ppy = np.polyfit(t[0:-1], (y[0:-1])/10, 2)

plt.plot(t[0:-1], x[0:-1]/10, 'ro')
plt.plot(t[0:-1], y[0:-1]/10, 'ks')

plt.plot(t[0:-1], np.polyval(ppx, t[0:-1]), 'r-')
plt.plot(t[0:-1], np.polyval(ppy, t[0:-1]), 'k-')

print('The acceleration in the x direction is: {:.2f}'.format(ppx[0]*2))
print('The acceleration in the y direction is: {:.2f}'.format(ppy[0]*2))
```

1.1d. The second order polyfit line fit the data better. The trend of the line fits the data points almost exactly, which indicates that it is a good fit for the data.

+++

### 2. Not only can you measure acceleration of objects that you track, you can look at other physical constants like [coefficient of restitution](https://en.wikipedia.org/wiki/Coefficient_of_restitution), $e$ .

+++

During a collision with the ground, the coefficient of restitution is
     
$e = -\frac{v_{y}'}{v_{y}}$ . 
     
Where $v_y'$ is y-velocity perpendicular to the ground after impact and $v_y$ is the y-velocity after impact. 
     
a. Calculate $v_y$ and plot as a function of time from the data `'../data/fallingtennisball02.txt'`
     
b. Find the locations when $v_y$ changes rapidly i.e. the impact locations. Get the maximum and minimum velocities closest to the impact location. _Hint: this can be a little tricky. Try slicing the data to include one collision at a time before using  the `np.min` and `np.max` commands._
     
c. Calculate the $e$ for each of the three collisions

```{code-cell} ipython3
tennis_ball = np.loadtxt('../data/fallingtennisball02.txt')
t = tennis_ball[:,0]
y = tennis_ball[:,1]
vy = np.diff(y)/np.diff(t)

plt.title('Time vs. $v_y$')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.plot(t, y)
```

```{code-cell} ipython3
print('1.2a.')

print('vy = ', vy)

plt.plot(t[1:], vy)
plt.plot(t[t > 1], vy[t[1:] > 1])

plt.title('Time vs. $v_y$')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
```

```{code-cell} ipython3
print('1.2b.')

np.max(vy[t[1:] > 1])/np.min(vy[t[1:] > 1])

collision1 = np.max(vy[t[1:] > 1])
collision2 = 0
collision3 = np.min(vy[t[1:] > 1])

print('The three places where the velocity changes rapidly are {}, {}, and {}, respectively.'.format(collision1, collision2, collision3))
```

These numbers seem off to me, but I wasn't really sure how to do this part. I tried to use the max and the min functions, but I'm not sure if this was the correct way to do this problem.

```{code-cell} ipython3
print('1.2c.')

e1 = -(collision1/vy.mean())
e2 = -(collision2/vy.mean())
e3 = -(collision3/vy.mean())

print('The coefficient of resitution at each of the collisions were {}, {}, and {}, respectively.'.format(e1, e2, e3))
```

## Problems [Part 2](./02_Step_Future.md)

1. Integrate the `fall_drag` equations for a tennis ball and a [lacrosse ball](https://en.wikipedia.org/wiki/Lacrosse_ball) with the same initial conditions as above. Plot the resulting height vs time. 

_Given:_ y(0) = 1.6 m, v(0) = 0 m/s

|ball| diameter | mass|
|---|---|---|
|tennis| $6.54$–$6.86 \rm{cm}$ |$56.0$–$59.4 \rm{g}$|
|lacrosse| $6.27$–$6.47 \rm{cm}$ |$140$–$147 \rm{g}$|

Is there a difference in the two solutions? At what times do the tennis ball and lacrosse balls reach the ground? Which was first?

+++

![Projectile motion with drag](../images/projectile.png)

The figure above shows the forces acting on a projectile object, like the [lacrosse ball](https://en.wikipedia.org/wiki/Lacrosse_ball) from [Flipping Physics](http://www.flippingphysics.com) that you analyzed in [lesson 01_Catch_Motion](./01_Catch_Motion.ipynb). Consider the 2D motion of the [lacrosse ball](https://en.wikipedia.org/wiki/Lacrosse_ball), now the state vector has two extra variables, 

$
\mathbf{y} = \begin{bmatrix}
x \\ v_x \\
y \\ v_y 
\end{bmatrix},
$

and its derivative is now, 

$\dot{\mathbf{y}} = \begin{bmatrix}
v_x \\ -c v_x^2 \\
v_y \\ g - cv_y^2 
\end{bmatrix},$ 

where $c= \frac{1}{2} \pi R^2 \rho C_d$.

```{code-cell} ipython3
def fall_drag(state,C_d=0.47,m=0.0577,R = 0.0661/2):
    '''Computes the right-hand side of the differential equation
    for the fall of a ball, with drag, in SI units.
    
    Arguments
    ----------    
    state : array of two dependent variables [y v]^T
    m : mass in kilograms default set to 0.0577 kg
    C_d : drag coefficient for a sphere default set to 0.47 (no units)
    R : radius of ball default in meters is 0.0661/2 m (tennis ball)
    Returns
    -------
    derivs: array of two derivatives [v (-g+a_drag)]^T
    '''
    
    rho = 1.22   # air density kg/m^3
    pi = np.pi
    
    a_drag = -1/(2*m) * pi * R**2 * rho * C_d * (state[1])**2*np.sign(state[1])
    
    derivs = np.array([state[1], -9.81 + a_drag])
    return derivs
```

```{code-cell} ipython3
def eulerstep(state, rhs, dt):
    '''Uses Euler's method to update a state to the next one. 
    
    Arguments
    ---------
    state: array of two dependent variables [y v]^T
    rhs  : function that computes the right hand side of the 
           differential equation.
    dt   : float, time increment. 
    
    Returns
    -------
    next_state: array, updated state after one time increment.       
    '''
    
    next_state = state + rhs(state) * dt
    return next_state
```

```{code-cell} ipython3
t = np.linspace(0, 1)
dt = t[1] - t[0]

state_tennis = np.zeros((len(t), 2))
state_tennis[0] = [1.6, 0]
m_tennis = 56e-3
rhs = lambda state_tennis: fall_drag(state_tennis, m = m_tennis)

for i in range(1, len(t)):
    state_tennis[i] = eulerstep(state_tennis[i-1], lambda state_tennis: fall_drag(state_tennis, m = m_tennis), dt)

plt.plot(t, state_tennis[:,0])
plt.ylabel("Height (m)")
plt.xlabel("Time (s)")
plt.title("Tennis Ball Motion")
```

```{code-cell} ipython3
t[state_tennis[:,0] < 0][0]
```

```{code-cell} ipython3
t = np.linspace(0, 1)
dt = t[1] - t[0]

state_lacrosse = np.zeros((len(t), 2))
state_lacrosse[0] = [1.6, 0]
m_lacrosse = 140e-3
rhs = lambda state_lacrosse: fall_drag(state_lacrosse, m = m_lacrosse)

for i in range(1, len(t)):
    state_lacrosse[i] = eulerstep(state_lacrosse[i-1], lambda state_lacrosse: fall_drag(state_lacrosse, m = m_lacrosse), dt)

plt.plot(t, state_lacrosse[:,0])
plt.ylabel("Height (m)")
plt.xlabel("Time (s)")
plt.title("Lacrosse Ball Motion")
```

```{code-cell} ipython3
t[state_lacrosse[:,0] < 0][0]
```

2.1

The two balls will land at the exact same time, approximately 0.5918 seconds. There is no difference in the solutions.

+++

## Problems [Part 3](./03_Get_Oscillations.md)

1. Show that the implicit Heun's method has the same second order convergence as the Modified Euler's method. _Hint: you can use the same code from above to create the log-log plot to get the error between $2\cos(\omega t)$ and the `heun_step` integration. Use the same initial conditions x(0) = 2 m and v(0)=0m/s and the same RHS function, `springmass`._

```{code-cell} ipython3
def heun_step(state,rhs,dt,etol=0.000001,maxiters = 100):
    '''Update a state to the next time increment using the implicit Heun's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    etol  : tolerance in error for each time step corrector
    maxiters: maximum number of iterations each time step can take
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    e=1
    eps=np.finfo('float64').eps
    next_state = state + rhs(state)*dt
    ################### New iterative correction #########################
    for n in range(0,maxiters):
        next_state_old = next_state
        next_state = state + (rhs(state)+rhs(next_state))/2*dt
        e=np.sum(np.abs(next_state-next_state_old)/np.abs(next_state+eps))
        if e<etol:
            break
    ############### end of iterative correction #########################
    return next_state
```

```{code-cell} ipython3
def springmass(state, w = 2):
    '''Computes the right-hand side of the spring-mass differential 
    equation, without friction.
    
    Arguments
    ---------   
    state : array of two dependent variables [x v]^T
    
    Returns 
    -------
    derivs: array of two derivatives [v - w*w*x]^T
    '''
    
    derivs = np.array([state[1], -w**2*state[0]])
    return derivs
```

```{code-cell} ipython3
w = 2
period = 2*np.pi/w
dt_values = np.array([period/50, period/100, period/200,period/400,period/1000])
T = 1*period

x0 = 2
v0 = 0

num_sol_time = np.empty_like(dt_values, dtype=np.ndarray)


for j, dt in enumerate(dt_values):

    N = int(T/dt)
    t = np.linspace(0, T, N)
    
    #initialize solution array
    num_sol = np.zeros([N,2])
    
    
    #Set intial conditions
    num_sol[0,0] = x0
    num_sol[0,1] = v0
    
    for i in range(N-1):
        num_sol[i+1] = heun_step(num_sol[i], springmass, dt)

    num_sol_time[j] = num_sol.copy()
```

```{code-cell} ipython3
def get_error(num_sol, T):
    
    x_an = x0 * np.cos(w * T) # analytical solution at final time
    
    error =  np.abs(num_sol[-1,0] - x_an)
    
    return error
```

```{code-cell} ipython3
error_values = np.empty_like(dt_values)

for j in range(len(dt_values)):
    
    error_values[j] = get_error(num_sol_time[j], T)
```

```{code-cell} ipython3
# plot the solution errors with respect to the time incremetn
fig = plt.figure(figsize=(6,6))

plt.loglog(dt_values, error_values, 'ko-')  #log-log plot
plt.loglog(dt_values, 10*dt_values**2, 'k:')
plt.grid(True)                         #turn on grid lines
plt.axis('equal')                      #make axes scale equally
plt.xlabel('$\Delta t$')
plt.ylabel('Error')
plt.title('Convergence of the Heun method (dotted line: slope 2)\n');
```

<img src="../images/damped-spring.png" style="width: 400px;"/>

+++

2. In the image above, you have a spring, mass, _and damper_. A damper is designed to slow down a moving object. These devices are typical in automobiles, mountain bikes, doors, any place where oscillations may not be desired, but motion is required. The new differential equation, if F(t)=0, that results from this addition is

$\ddot{x} = -\frac{b}{m}\dot{x} -\frac{k}{m}x$

or keeping our _natural frequency_ above, 

$\ddot{x} = -\zeta\omega\dot{x} -\omega^2x$

where $\zeta$ is a new constant called the __damping ratio__ of a system. When $\zeta\gt 1$, there are no oscillations and when $0<\zeta<1$ the system oscillates, but decays to v=0 m/s eventually. 

Create the system of equations that returns the right hand side (RHS) of the state equations, e.g. $\mathbf{\dot{y}} = f(\mathbf{y}) = RHS$

Use $\omega = 2$ rad/s and $\zeta = 0.2$.

```{code-cell} ipython3
#3.2

def smd(state, w = 2, zeta = 0.2):
    '''Computes the right-hand side of the spring-mass-damper
    differential equation, without friction.
    
    Arguments
    ---------   
    state : array of two dependent variables [x, v]^T
    
    Returns 
    -------
    derivs: array of two derivatives [v, -zeta*w*v - w*w*x]^T
    '''
    x, v = state
    derivs = np.array([state[1], -zeta*w*v - w**2*x])
    
    return derivs
```

3. Use three methods to integrate your `smd` function for 3 time periods of oscillation and initial conditions x(0)=2 m and v(0)=0 m/s. Plot the three solutions on one graph with labels. 

a. Euler integration

b. second order Runge Kutta method (modified Euler method)

c. the implicit Heun's method

How many time steps does each method need to converge to the same results? _Remember that each method has a certain convergence rate_

```{code-cell} ipython3
#3.3a - Euler Integration
```

```{code-cell} ipython3
def eulerstep(state, rhs, dt):
    '''Uses Euler's method to update a state to the next one. 
    
    Arguments
    ---------
    state: array of two dependent variables [y v]^T
    rhs  : function that computes the right hand side of the 
           differential equation.
    dt   : float, time increment. 
    
    Returns
    -------
    next_state: array, updated state after one time increment.       
    '''
    
    next_state = state + rhs(state) * dt
    return next_state
```

```{code-cell} ipython3
#3.3b - Second Order Runge Kutta method
```

```{code-cell} ipython3
def rk2_step(state, rhs, dt):
    '''Update a state to the next time increment using modified Euler's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    
    mid_state = state + rhs(state) * dt*0.5    
    next_state = state + rhs(mid_state)*dt
 
    return next_state
```

```{code-cell} ipython3
#3.3c - Implicit Heun's method
```

```{code-cell} ipython3
def heun_step(state,rhs,dt,etol=0.000001,maxiters = 100):
    '''Update a state to the next time increment using the implicit Heun's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    etol  : tolerance in error for each time step corrector
    maxiters: maximum number of iterations each time step can take
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    e=1
    eps=np.finfo('float64').eps
    next_state = state + rhs(state)*dt
    ################### New iterative correction #########################
    for n in range(0,maxiters):
        next_state_old = next_state
        next_state = state + (rhs(state)+rhs(next_state))/2*dt
        e=np.sum(np.abs(next_state-next_state_old)/np.abs(next_state+eps))
        if e<etol:
            break
    ############### end of iterative correction #########################
    return next_state
```

```{code-cell} ipython3
N = 101
t = np.linspace(0, 10, N)
dt = t[1] - t[0]

eul_sol = np.zeros((len(t), 2))
rk2_sol = np.zeros((len(t), 2))
heun_sol = np.zeros((len(t), 2))

eul_sol[0, :] = [2, 0]
rk2_sol[0, :] = [2, 0]
heun_sol[0, :] = [2, 0]

for i in range(1, len(t)):
    eul_sol[i] = eulerstep(eul_sol[i-1], smd, dt)
    rk2_sol[i] = rk2_step(rk2_sol[i-1], smd, dt)
    heun_sol[i] = heun_step(heun_sol[i-1], smd, dt)
    
plt.plot(t, eul_sol[:, 0], label = 'Euler')
plt.plot(t, rk2_sol[:, 0], label = 'Runge Kutta (RK2)')
plt.plot(t, heun_sol[:, 0], label = 'Heun')
plt.legend(bbox_to_anchor = (1,1))
plt.title('Timestep = {} s'.format(dt))
```

The RK2 and Heun functions seem to converge with each other with a timestep of 0.1 s, however the Euler function does not work as well with oscillating systems. This means that it will make a lot more, much smaller time steps to see the Euler function converge with the other two.

```{code-cell} ipython3
N = 1001
t = np.linspace(0, 10, N)
dt = t[1] - t[0]

eul_sol = np.zeros((len(t), 2))
rk2_sol = np.zeros((len(t), 2))
heun_sol = np.zeros((len(t), 2))

eul_sol[0, :] = [2, 0]
rk2_sol[0, :] = [2, 0]
heun_sol[0, :] = [2, 0]

for i in range(1, len(t)):
    eul_sol[i] = eulerstep(eul_sol[i-1], smd, dt)
    rk2_sol[i] = rk2_step(rk2_sol[i-1], smd, dt)
    heun_sol[i] = heun_step(heun_sol[i-1], smd, dt)
    
plt.plot(t, eul_sol[:, 0], label = 'Euler')
plt.plot(t, rk2_sol[:, 0], label = 'Runge Kutta (RK2)')
plt.plot(t, heun_sol[:, 0], label = 'Heun')
plt.legend(bbox_to_anchor = (1,1))
plt.title('Timestep = {} s'.format(dt))
```

It takes timesteps of around 0.01 s for the Euler function to converge with RK2 and Heun functions. The yellow and blue are very similar, and while there are is a little blue showing, they still look to be converging with each other. More, very small time steps will be most accurate to should convergence (like 3001, for example), but it can be noted that at around 1001 time steps is where the Euler function starts to show convergence.

+++

## Problems [Part 3](./04_Getting_to_the_root.md)

1. One of the main benefits of a bracketing method is the stability of solutions. Open methods are not always stable. Here is an example. One way engineers and data scientists model the probability of failure is with a [sigmoid function e.g. this Challenger O-ring case study](https://byuistats.github.io/M325_Hathaway/textbook/challengerLogisticReg.html)

$$\begin{equation}
    \sigma(T) = \frac{e^{a_0-a_1 T}}{1+e^{a_0-a_1 T}}
\end{equation}$$

The Challenger explosion was a terrible incident that occurred due to the failure of an O-ring. The post-mortem data analysis showed that at low temperatures the O-rings were brittle and more likely to fail. You can use the function $\sigma(T)$ to determine the point at which there is a 50\% chance of O-ring failure. Using the pass-fail data, the two constants are

$a_0 = 15.043$

$a_1 = 0.232$

a. Plot the function $\sigma(T)$ for $T=0-100^{o}F$. Where do you see the function cross 50\% (0.5)?

b. Create two functions `f_T` and `dfdT` where `f_T`=$f(T)=\sigma(T) - 0.5$ and `dfdT`=$\frac{df}{dT}$

c. Use the `incsearch` and `newtraph` functions to find the root of f(T). When does Newton-Raphson fail to converge? Why does it fail? _Hint: if you're stuck here, take a look at this [youtube video finding an interval of convergence for the Newton-Raphson method](https://youtu.be/zyXRo8Qjj0A). Look at the animation of how the method converges and diverges._

```{code-cell} ipython3
print('3.1a.')

T = np.linspace(0, 100)

a0 = 15.043
a1 = 0.232

def prob_fail(T, a0 = 15.043, a1 = 0.232):
    '''Determine the point at which there is a 50% chance of O-ring failure.
    
    Arguments
    ---------
    a0 : initial acceleration
    a1 : final acceleration
    T : temperature
    
    Returns
    -------
    fail_perc : failure percentage'''
    
    sigma = np.exp(a0 - a1*T)/(1 + np.exp(a0 - a1*T))
    
    return sigma

failure_probabiliites = prob_fail(T, a0 = 15.043, a1 = 0.232)

fifty_perc = np.isclose(failure_probabiliites, 0.5, atol = 0.05)
print(T[fifty_perc])
```

```{code-cell} ipython3
plt.plot(T, failure_probabiliites)
```

The function crosses 50% at approximately $65^{o}F$. A function can also be used to find the closest temperature to where the failure probablility reaches 50%. In this case, using the `np.isclose()` function, it was returned that the  temperature that would produce closest to a 50% failure probablility is $65.30612245^{o}F$.

```{code-cell} ipython3
print('3.1b.')
```

```{code-cell} ipython3
def f_T(prob_fail = 0.5, T = 0.5):
    f_T = prob_fail - T
    return f_T

def dfdT(T = 0.5):
    dfdT = -np.exp(T + 1)/(np.exp(T) + np.exp(1))**2
    return dfdT
```

```{code-cell} ipython3
import sympy
T = sympy.var('T')
f = sympy.exp(a0 - a1*T)/(1+sympy.exp(a0-a1*T)) - 0.5
dfdT = sympy.diff(f, T)
```

```{code-cell} ipython3
dfdT_sol = sympy.lambdify(T, dfdT, 'numpy')
```

```{code-cell} ipython3
f = lambda T: np.exp(a0 - a1*T)/(1 + np.exp(a0-a1*T)) - 0.5

T = np.linspace(0, 120)

plt.plot(T, f(T))
plt.plot(T, dfdT_sol(T)*10)
```

```{code-cell} ipython3
def newtraph(func,dfunc,x0,es=0.0001,maxit=50):
    '''newtraph: Newton-Raphson root location zeroes
    root,[ea,iter]=newtraph(func,dfunc,x0,es,maxit,p1,p2,...):
    uses Newton-Raphson method to find the root of func
    arguments:
    ----------
    func = name of function
    dfunc = name of derivative of function
    x0 = initial guess
    es = desired relative error (default = 0.0001 )
    maxit = maximum allowable iterations (default = 50)
    returns:
    ----------
    root = real root
    ea = approximate relative error (%)
    iter = number of iterations'''
    xr = x0
    ea=1
    for iter in range(1,maxit):
        xrold = xr
        dx = -func(xr)/dfunc(xr)
        xr = xrold+dx
        if xr!=0:
            ea= np.abs((xr-xrold)/xr)*100 # relative error in %
        if ea < es:
            break
    return xr,[func(xr),ea,iter]
```

```{code-cell} ipython3
print('3.1c.')

newtraph(f, dfdT_sol, 20)
```

The Newton-Rauphson fails to converge because it will trend towards infinity, as shown above. Because we are looking at such a small interval, it can be difficult to see convergence of a function. 

```{code-cell} ipython3
actual, deriv = newtraph(f, dfdT_sol, 1)
print('There would be a root at', deriv[2], 'degrees F.')
```

```{code-cell} ipython3
def incsearch(func,xmin,xmax,ns=50):
    '''incsearch: incremental search root locator
    xb = incsearch(func,xmin,xmax,ns):
      finds brackets of x that contain sign changes
      of a function on an interval
    arguments:
    ---------
    func = name of function
    xmin, xmax = endpoints of interval
    ns = number of subintervals (default = 50)
    returns:
    ---------
    xb(k,1) is the lower bound of the kth sign change
    xb(k,2) is the upper bound of the kth sign change
    If no brackets found, xb = [].'''
    x = np.linspace(xmin,xmax,ns)
    f = func(x)
    sign_f = np.sign(f)
    delta_sign_f = sign_f[1:]-sign_f[0:-1]
    i_zeros = np.nonzero(delta_sign_f!=0)
    nb = len(i_zeros[0])
    xb = np.block([[ x[i_zeros[0]+1]],[x[i_zeros[0]] ]] )

    
    if nb==0:
      print('no brackets found\n')
      print('check interval or increase ns\n')
    else:
      print('number of brackets:  {}\n'.format(nb))
    return xb
```

```{code-cell} ipython3
mn = 0
mx = 0.5
T = np.linspace(mn, mx)
plt.plot(T, f_T(T))

xb = incsearch(lambda T: f_T(T), mn, mx, ns=50)
plt.plot(xb, f_T(xb), 's')
plt.ylabel('f_T')
plt.xlabel('T')
plt.title('Upper bound on temperature = {:.2f} \nLower Bounds = {:.2f}'.format(*xb[0,:],*xb[1,:]));
```

This graph shows that there would be a root somewhere between 0.49 and 0.5, where the two red dots are located. This is similar to the root value that was found above, 49, however it is a decimal point off.

+++

2. In the [Shooting Method
   example](https://cooperrc.github.io/computational-mechanics/module_03/04_Getting_to_the_root.html#shooting-method), you determined the initial velocity after the first
   bounce by specifying the beginning y(0) and end y(T) for an object
   subject to gravity and drag. Repeat this analysis for the time period
   just after the second bounce and just before the third bounce. The
   indices are given below for t[1430:2051] = 1.43-2.05 seconds.

    a. What is the velocity just after the second bounce?

    b. What is the coefficient of restitution for the second bounce? _Hint: use the ratio of the last velocity from above to the initial velocity calculated here._

```{code-cell} ipython3
filename = '../data/fallingtennisball02.txt'
t, y = np.loadtxt(filename, usecols=[0,1], unpack=True)
i0=1430
ie=2051
print(t[i0],t[ie])
plt.plot(t,y)
plt.plot(t[i0:ie],y[i0:ie],'s')
```

```{code-cell} ipython3
def fall_drag(state,C_d=0.47,m=0.0577,R = 0.0661/2):
    '''Computes the right-hand side of the differential equation
    for the fall of a ball, with drag, in SI units.
    
    Arguments
    ----------    
    state : array of two dependent variables [y v]^T
    m : mass in kilograms default set to 0.0577 kg
    C_d : drag coefficient for a sphere default set to 0.47 (no units)
    R : radius of ball default in meters is 0.0661/2 m (tennis ball)
    Returns
    -------
    derivs: array of two derivatives [v (-g+a_drag)]^T
    '''
    
    rho = 1.22   # air density kg/m^3
    pi = np.pi
    
    a_drag = -1/(2*m) * pi * R**2 * rho * C_d * (state[1])**2*np.sign(state[1])
    
    derivs = np.array([state[1], -9.8 + a_drag])
    return derivs
```

```{code-cell} ipython3
from scipy.integrate import solve_ivp
```

```{code-cell} ipython3
def shooting_bounce(v):   
    sol = solve_ivp(lambda t, y: fall_drag(y), 
                    [t[i0],t[ie]], 
                    [y[i0], v], 
                    t_eval = np.linspace(t[i0], t[ie]))
    return sol.y[0, -1] - y[ie]
fy = shooting_bounce(10)
fy
```

```{code-cell} ipython3
def mod_secant(func,dx,x0,es=0.0001,maxit=50):
    '''mod_secant: Modified secant root location zeroes
    root,[fx,ea,iter]=mod_secant(func,dfunc,xr,es,maxit,p1,p2,...):
    uses modified secant method to find the root of func
    arguments:
    ----------
    func = name of function
    dx = perturbation fraction
    xr = initial guess
    es = desired relative error (default = 0.0001 )
    maxit = maximum allowable iterations (default = 50)
    p1,p2,... = additional parameters used by function
    returns:
    --------
    root = real root
    fx = func evaluated at root
    ea = approximate relative error ( )
    iter = number of iterations'''

    iter = 0;
    xr=x0
    for iter in range(0,maxit):
        xrold = xr;
        dfunc=(func(xr+dx)-func(xr))/dx;
        xr = xr - func(xr)/dfunc;
        if xr != 0:
            ea = abs((xr - xrold)/xr) * 100;
        else:
            ea = abs((xr - xrold)/1) * 100;
        if ea <= es:
            break
    return xr,[func(xr),ea,iter]
```

```{code-cell} ipython3
print('3.2a.')

xr, outputs = mod_secant(shooting_bounce, 0.001, 2)

print('The velocity just after the second bounce is {} m/s.'.format(xr))
```

```{code-cell} ipython3
sol = solve_ivp(lambda t, y: fall_drag(y), 
                    [t[i0],t[ie]], 
                    [y[i0], xr], 
                    t_eval = np.linspace(t[i0], t[ie]))


plt.plot(t, y)
plt.plot(t[i0:ie], y[i0:ie], 's', label = 'data')
plt.plot(sol.t, sol.y[0], label = 'my shooting solution')

plt.legend()
```

```{code-cell} ipython3
print('3.2b.')

e = t[ie]/xr

print('The coefficient of restitution for the second bounce is {}.'. format(e))
```
