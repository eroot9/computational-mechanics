```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.


```python
initial_temp = 85 #degrees F
initial_time = 0

final_temp = 74 #degrees F
final_time = 2

ambient_temp = 65 #degrees F

change_time = final_time - initial_time

dT_dt = (final_temp - initial_temp)/change_time

K = -(dT_dt)/(initial_temp - ambient_temp) 

print(K)
```

    0.275


2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.


```python
def emper_constant(initial_temp, final_temp, ambient_temp, time_elapsed):
    dT_dt = (final_temp - initial_temp)/time_elapsed
    K = -(dT_dt)/(initial_temp - ambient_temp)
    return K
```


```python
emper_constant(85, 74, 65, 2)
```




    0.275



3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?


```python
import numpy as np
import matplotlib.pyplot as plt

print('3a.')

t = np.linspace(0, 12, 5)
dt = t[1] - t[0]

print('Time Step =', dt)

#Analytical Solution

def analytical(ambient_temp, initial_temp, K, t):
    T_analytical = ambient_temp + (initial_temp - ambient_temp) * np.exp(-K * t)
    return T_analytical

#Euler Solution
def euler(ambient_temp, initial_temp, K, t):
    T_eul = np.zeros(len(t))
    T_eul[0] = initial_temp
    for i in range(1, len(t)):
        T_eul[i] = T_eul[i-1] - K * (T_eul[i-1] - ambient_temp) * dt
    return T_eul

plt.plot(analytical(ambient_temp, initial_temp, K, t), label = "Analytical")
plt.plot(euler(ambient_temp, initial_temp, K, t), color = 'red', label = "Euler")
plt.xlabel('time (hr)')
plt.ylabel('Temperature')
plt.legend()
```

    3a.
    Time Step = 3.0





    <matplotlib.legend.Legend at 0x7fa75b64b160>




    
![png](output_7_2.png)
    



```python
t = np.linspace(0, 4, 5)
dt = t[1] - t[0]
print('Time Step =', dt)

plt.plot(analytical(ambient_temp, initial_temp, K, t), label = "Analytical")
plt.plot(euler(ambient_temp, initial_temp, K, t), color = 'red', label = "Euler")
plt.xlabel('time (hr)')
plt.ylabel('Temperature')
plt.legend()
```

    Time Step = 1.0





    <matplotlib.legend.Legend at 0x7fa7607581c0>




    
![png](output_8_2.png)
    


As shown in the graphs above, as the time step decreases the two graphs converge with each other. The first graph has  larger time steps whereas the second graph shown has smaller time steps. While the data in the first graph has the same general trajectory, they have significant error in some places, whereas the second graph shows the convergence of the two solutions of the data with some, but much less, error. 


```python
print('3b.')

print('As t -> infinity:', analytical(ambient_temp, initial_temp, K, np.inf), 'degrees F')
```

    3b.
    As t -> infinity: 65.0 degrees F


As t approaches infinity, it makes sense that the body temperature will reach thermal equilibrium with the environment that it is left in. If the ambient temperature is 65 degrees F, then the body will eventually reach that temperature after a given amount of time.


```python
N = 50
t = np.linspace(0, 2, N)
dt = t[1] - t[0]

T_eul_f = np.zeros(len(t))
T_eul_f[0] = initial_temp
T_eul_b = np.zeros(len(t))
T_eul_b[0] = initial_temp
for i in range(1, len(t)):
    T_eul_f[i] = T_eul_f[i-1] - K * (T_eul_f[i-1] - ambient_temp) * dt
    T_eul_b[i] = T_eul_b[i-1] + K * (T_eul_b[i-1] - ambient_temp) * dt
    
plt.plot(t, T_eul_f, label = 'Forwards')
plt.plot(-t, T_eul_b, label = 'Backwards')
plt.xlabel('Time (hr)')
plt.ylabel('Temperature')
plt.legend()

Tdeath = np.isclose(T_eul_b, 98.6, atol = 0.18)
print(t[Tdeath])

print('3c.')
```

    [1.87755102]
    3c.



    
![png](output_12_1.png)
    


The corpse was 98.6 degrees F about 1.8 hours before the temperature was taken. This would mean the death occurred at approximately 9:12 am if the initial body temperature was recorded at 11:00am.
