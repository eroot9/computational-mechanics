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

# CompMech04-Linear Algebra Project
## Practical Linear Algebra for Finite Element Analysis

+++

In this project we will perform a linear-elastic finite element analysis (FEA) on a support structure made of 11 beams that are riveted in 7 locations to create a truss as shown in the image below. 

![Mesh image of truss](../images/mesh.png)

+++

The triangular truss shown above can be modeled using a [direct stiffness method [1]](https://en.wikipedia.org/wiki/Direct_stiffness_method), that is detailed in the [extra-FEA_material](./extra-FEA_material.ipynb) notebook. The end result of converting this structure to a FE model. Is that each joint, labeled $n~1-7$, short for _node 1-7_ can move in the x- and y-directions, but causes a force modeled with Hooke's law. Each beam labeled $el~1-11$, short for _element 1-11_, contributes to the stiffness of the structure. We have 14 equations where the sum of the components of forces = 0, represented by the equation

$\mathbf{F-Ku}=\mathbf{0}$

Where, $\mathbf{F}$ are externally applied forces, $\mathbf{u}$ are x- and y- displacements of nodes, and $\mathbf{K}$ is the stiffness matrix given in `fea_arrays.npz` as `K`, shown below

_note: the array shown is 1000x(`K`). You can use units of MPa (N/mm^2), N, and mm. The array `K` is in 1/mm_

$\mathbf{K}=EA*$

$  \left[ \begin{array}{cccccccccccccc}
 4.2 & 1.4 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 1.4 & 2.5 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 8.3 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 4.2 & -1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & -1.4 & 2.5 \\
\end{array}\right]~\frac{1}{m}$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
fea_arrays = np.load('./fea_arrays.npz')
K=fea_arrays['K']
K
```

```{code-cell} ipython3
print(np.linalg.cond(K))
print(np.linalg.cond(K[2:13,2:13]))


print('expected error in x=solve(K,b) is {}'.format(10**(17-16)))
print('expected error in x=solve(K[2:13,2:13],b) is {}'.format(10**(2-16)))
```

In this project we are solving the problem, $\mathbf{F}=\mathbf{Ku}$, where $\mathbf{F}$ is measured in Newtons, $\mathbf{K}$ `=E*A*K` is the stiffness in N/mm, `E` is Young's modulus measured in MPa (N/mm^2), and `A` is the cross-sectional area of the beam measured in mm^2. 

There are three constraints on the motion of the joints:

i. node 1 displacement in the x-direction is 0 = `u[0]`

ii. node 1 displacement in the y-direction is 0 = `u[1]`

iii. node 7 displacement in the y-direction is 0 = `u[13]`

We can satisfy these constraints by leaving out the first, second, and last rows and columns from our linear algebra description.

+++

### 1. Calculate the condition of `K` and the condition of `K[2:13,2:13]`. 

a. What error would you expect when you solve for `u` in `K*u = F`? 

b. Why is the condition of `K` so large? __The problem is underconstrained. It describes stiffness of structure, but not the BC's. So, we end up with sumF=0 and -sumF=0__

c. What error would you expect when you solve for `u[2:13]` in `K[2:13,2:13]*u=F[2:13]`

```{code-cell} ipython3
print('1a.')
```

When you try to solve for u in K*u = F, you would expect to find a 'ill-conditioned matrix' error. This usually occurs when you do not apply boundary conditions correctly.  

```{code-cell} ipython3
print('1b.')
```

If you just use `np.linalg.cond(K)`, you are taking away the boundary conditions at node 1 and at node 7 (the first and last node in the truss model). This means that if a force was applied, it would just move in the direction of the force, as there is nothing supporting it or keeping it attached to whatever is on either side of it.

```{code-cell} ipython3
elems = fea_arrays['elems']
nodes = fea_arrays['nodes']

ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

r = np.block([n[1:3] for n in nodes])
r
```

```{code-cell} ipython3
def Kel(node1,node2):
    '''Kel(node1,node2) returns the diagonal and off-diagonal element stiffness matrices based upon
    initial angle of a beam element and its length the full element stiffness is
    K_el = np.block([[Ke1,Ke2],[Ke2,Ke1]])
    
    Out: [Ke1 Ke2]
         [Ke2 Ke1]   
    arguments:
    ----------
    node1: is the 1st node number and coordinates from the nodes array
    node2: is the 2nd node number and coordinates from the nodes array
    outputs:
    --------
    Ke1 : the diagonal matrix of the element stiffness
    Ke2 : the off-diagonal matrix of the element stiffness
    '''
    a = np.arctan2(node2[2]-node1[2],node2[1]-node1[1])
    l = np.sqrt((node2[2]-node1[2])**2+(node2[1]-node1[1])**2)
    Ke1 = 1/l*np.array([[np.cos(a)**2,np.cos(a)*np.sin(a)],[np.cos(a)*np.sin(a),np.sin(a)**2]])
    Ke2 = 1/l*np.array([[-np.cos(a)**2,-np.cos(a)*np.sin(a)],[-np.cos(a)*np.sin(a),-np.sin(a)**2]])
    return Ke1,Ke2
```

```{code-cell} ipython3
K=np.zeros((len(nodes)*2,len(nodes)*2))
for e in elems:
    ni = nodes[e[1]-1]
    nj = nodes[e[2]-1]
    
    Ke1,Ke2 = Kel(ni,nj)
    #--> assemble K <--
    i1=int(ni[0])*2-2
    i2=int(ni[0])*2
    j1=int(nj[0])*2-2
    j2=int(nj[0])*2
    
    K[i1:i2,i1:i2]+=Ke1
    K[j1:j2,j1:j2]+=Ke1
    K[i1:i2,j1:j2]+=Ke2
    K[j1:j2,i1:i2]+=Ke2
    
np.savez('fea_arrays',nodes=nodes,elems=elems,K=K)
```

```{code-cell} ipython3
print(K*1000)
```

```{code-cell} ipython3
print('1c.\n')

print(np.linalg.cond(K))
print(np.linalg.cond(K[2:13,2:13]))


print('expected error in x=solve(K,b) is {}'.format(10**(17-16)))
print('expected error in x=solve(K[2:13,2:13],b) is {}'.format(10**(2-16)))
```

### 2. Apply a 300-N downward force to the central top node (n 4)

a. Create the LU matrix for K[2:13,2:13]

b. Use cross-sectional area of $0.1~mm^2$ and steel and almuminum moduli, $E=200~GPa~and~E=70~GPa,$ respectively. Solve the forward and backward substitution methods for 

* $\mathbf{Ly}=\mathbf{F}\frac{1}{EA}$

* $\mathbf{Uu}=\mathbf{y}$

_your array `F` is zeros, except for `F[5]=-300`, to create a -300 N load at node 4._

c. Plug in the values for $\mathbf{u}$ into the full equation, $\mathbf{Ku}=\mathbf{F}$, to solve for the reaction forces

d. Create a plot of the undeformed and deformed structure with the displacements and forces plotted as vectors (via `quiver`). Your result for aluminum should match the following result from [extra-FEA_material](./extra-FEA_material.ipynb). _note: The scale factor is applied to displacements $\mathbf{u}$, not forces._

> __Note__: Look at the [extra FEA material](./extra-FEA_material). It
> has example code that you can plug in here to make these plots.
> Including background information and the source code for this plot
> below.


![Deformed structure with loads applied](../images/deformed_truss.png)

```{code-cell} ipython3
from scipy.linalg import lu
P, L, U = lu(K[2:13,2:13])
```

```{code-cell} ipython3
print('2a. LU Matrix')

print(L, U)
```

```{code-cell} ipython3
L@U
```

```{code-cell} ipython3
print('2b.')
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
#Steel

E_st = 200e3
A = 0.1
F_st = np.zeros(11)
F_st[5] = -300/E_st/A
U_st = solveLU(L, U, P.T@F_st) #Forwards
print('Forwards:\n', U_st)


Utotal_st = np.zeros(14)
Utotal_st[2:13]= U_st #Backwards

u_st = Utotal_st
print('Backwards:\n', u_st)

F_st = E_st*A*K@Utotal_st
```

```{code-cell} ipython3
#Aluminium

E_al = 70e3
A = 0.1
F_al = np.zeros(11)
F_al[5] = -300/E_al/A
U_al = solveLU(L, U, P.T@F_al) #Forwards
print('Forwards:\n', U_al)


Utotal_al = np.zeros(14)
Utotal_al[2:13]= U_al #Backwards

u_al = Utotal_al
print('Backwards:\n', u_al)

F_al = E_al*A*K@Utotal_al
```

```{code-cell} ipython3
print('2c.')

F_st = E_st*A*K@Utotal_st
F_al = E_al*A*K@Utotal_al

print('\nSteel:\n')
for i, f in enumerate(F_st):
    print('F{}'.format(i), ':', f, 'N')

print('\nAluminium:\n')
for i, f in enumerate(F_al):
    print('F{}'.format(i), ':', f, 'N')
    
```

```{code-cell} ipython3
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

print('2d.')

nodes = fea_arrays['nodes']
elems = fea_arrays['elems']

ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

r = np.block([n[1:3] for n in nodes])

l = 300

def f_steel(s):
    plt.plot(r[ix],r[iy],'-',color=(0,0,0,1))
    plt.plot(r[ix]+u_st[ix]*s,r[iy]+u_st[iy]*s,'-',color=(1,0,0,1))
    #plt.quiver(r[ix],r[iy],u[ix],u[iy],color=(0,0,1,1),label='displacements')
    plt.quiver(r[ix],r[iy],F_st[ix],F_st[iy],color=(1,0,0,1),label='applied forces')
    plt.quiver(r[ix],r[iy],u_st[ix],u_st[iy],color=(0,0,1,1),label='displacements')
    plt.axis(300*np.array([-0.5,3.5,-0.5,2]))
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend(bbox_to_anchor=(1,0.5))
    plt.title('Deformation scale = {:.1f}x'.format(s))
```

```{code-cell} ipython3
f_steel(5)
```

```{code-cell} ipython3
nodes = fea_arrays['nodes']
elems = fea_arrays['elems']

ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

r = np.block([n[1:3] for n in nodes])

l = 300

def f_al(s):
    plt.plot(r[ix],r[iy],'-',color=(0,0,0,1))
    plt.plot(r[ix]+u_al[ix]*s,r[iy]+u_al[iy]*s,'-',color=(1,0,0,1))
    #plt.quiver(r[ix],r[iy],u[ix],u[iy],color=(0,0,1,1),label='displacements')
    plt.quiver(r[ix],r[iy],F_al[ix],F_al[iy],color=(1,0,0,1),label='applied forces')
    plt.quiver(r[ix],r[iy],u_al[ix],u_al[iy],color=(0,0,1,1),label='displacements')
    plt.axis(300*np.array([-0.5,3.5,-1,2]))
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend(bbox_to_anchor=(1,0.5))
    plt.title('Deformation scale = {:.1f}x'.format(s))
```

```{code-cell} ipython3
f_al(5)
```

### 3. Determine cross-sectional area

a. Using aluminum, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

b. Using steel, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

c. What are the weights of the aluminum and steel trusses with the
chosen cross-sectional areas?

```{code-cell} ipython3
print('3a.')

A_al = 24

F_al = np.zeros(11)
F_al[5] = -300/E_al/A
U_al = solveLU(L, U, P.T@F_al)

U_al
```

Using aluminum, the minimum cross-sectional area to keep total y-deflections  <0.2 𝑚𝑚 would be approximately $24 mm^2$.

```{code-cell} ipython3
print('3b.')

A_st = 8.25

F_st = np.zeros(11)
F_st[5] = -300/E_st/A
U_st = solveLU(L, U, P.T@F_st)

U_st
```

Using steel, the minimum cross-sectional area to keep total y-deflections  <0.2 𝑚𝑚 would be approximately $8.25 mm^2$.

```{code-cell} ipython3
print('3c.')

area_al = A_al * 11
rho_al = 2710 #kg/m3
mass_al = area_al * rho_al
weight_al = mass_al * 9.81

area_st = A_st * 11
rho_st = 7750 #kg/m3
mass_st = area_st * rho_st
weight_st = mass_st * 9.81

print('The weight of the aluminium truss would be {} N with the cross sectional area of approximately, {} mm^2.'.format(weight_al, A_al))
print('The weight of the steel truss would be {} N with the cross sectional area of approximately, {} mm^2.'.format(weight_st, A_st))
```

## References

1. <https://en.wikipedia.org/wiki/Direct_stiffness_method>
