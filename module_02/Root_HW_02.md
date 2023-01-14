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

+++

## Problems [Part 1](./01_Cheers_Stats_Beers.md)

1. Gordon Moore created an empirical prediction that the rate of
semiconductors on a computer chip would double every two years. This
prediction was known as Moore's law. Gordon Moore had originally only
expected this empirical relation to hold from 1965 - 1975
[[1](https://en.wikipedia.org/wiki/Moore%27s_law),[2](https://spectrum.ieee.org/computing/hardware/gordon-moore-the-man-whose-name-means-progress)],
but semiconductor manufacturers were able to keep up with Moore's law
until 2015. 

In the folder "../data" is a comma separated value (CSV) file,
"transistor_data.csv" [taken from wikipedia
01/2020](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors).

a. Use the `!head ../data/transistor_data.csv` command to look at
the top of the csv. What are the headings for the columns?

b. Load the csv into a pandas dataframe. How many missing values
(`NaN`) are
in the column with the number of transistors? What fraction are
missing?

```{code-cell} ipython3
!head ../data/transistor_data.csv
print()
print('1.1a.\n')
print('The headings for the columns are:\n- Processor \n- MOS transistor count \n- Date of Introduction \n- Designer \n- MOSprocess \n- Area')
```

```{code-cell} ipython3
transistor_data = pd.read_csv('../data/transistor_data.csv')
transistor_data
```

```{code-cell} ipython3
MOS_series = transistor_data['MOS transistor count']
MOS_clean = MOS_series.dropna()
MOS = MOS_clean.values

num_NaN = len(MOS_series) - len(MOS)
fraction_missing = 1 - (len(MOS)/len(MOS_series))
percent_missing = fraction_missing * 100

print('1.1b.\n')

print('There are', num_NaN, 'missing values (NaN) in the column with the number of transistors.')
print('The missing values makes up approximately {:.4f} or {:.2f}% of the data.'.format(fraction_missing, percent_missing))
```

## Problems [Part 2](./02_Seeing_Stats.md)

1. Many beers do not report the IBU of the beer because it is very
small. You may be accidentally removing whole categories of beer from
our dataset by removing rows that do not include the IBU measure. 

    a. Use the command `beers_filled = beers.fillna(0)` to clean the `beers` dataframe
    
    b. Repeat the steps above to recreate the plot "Beer ABV vs. IBU mean values by style" 
    scatter plot with `beers_filled`. What differences do you notice between the plots?

```{code-cell} ipython3
from matplotlib import cm

print('2.1a.\n')

!head ../data/beers.csv
beers = pd.read_csv('../data/beers.csv')

beers
```

```{code-cell} ipython3
beers_filled = beers.fillna(0)
beers_styles = beers_filled.drop(['Unnamed: 0','name','brewery_id','ounces','id'], axis=1)
style_counts=beers_styles['style'].value_counts()
ibu=beers_filled['ibu'].values
abv=beers_filled['abv'].values
style_means=beers_styles.groupby('style').mean()
style_counts=style_counts.sort_index()

colors=cm.viridis(style_counts.values)
ax = style_means.plot.scatter(figsize=(10,10),
x='abv', y='ibu', s=style_counts*20, color=colors,
title='Beer ABV vs. IBU mean values by style\n',
alpha=0.3);

for i, txt in enumerate(list(style_counts.index.values)):
    if style_counts.values[i] > 65:
        ax.annotate(txt, (style_means.abv.iloc[i],style_means.ibu.iloc[i]), fontsize=12)
        
print()
print('2.1b')
```

2.1b. Compared to the plot in Module 2 this scatter plot has been shifted over to the right. Instead of having the cluster of smaller of smaller purple dots centralized around the 0.02 abv area, the cluster has move to around 0.05 abv. There are also more styles of beer that have over 65 different beers being represented in this chart. The code above will show the name of the style of beer with more than 65 types of beer of the same style. For example, American IPA has 424 types beers of that style and its name is shown. In the scatter plot in Module 2, only the names of 4 different styles of beers are shown on the graph, while here, 8 are shown. This is most likely due to the fact that the scatter plot shown above does not drop the beers that don't have all the information. Instead, the values are filled in with zeros. Therefore, the graph above represents the average of more beers than in the lesson (which dropped the beers with missing information instead of filling in the missing values).

+++

2. Gordon Moore created an empirical prediction that the rate of
semiconductors on a computer chip would double every two years. This
prediction was known as Moore's law. Gordon Moore had originally only
expected this empirical relation to hold from 1965 - 1975
[[1](https://en.wikipedia.org/wiki/Moore%27s_law),[2](https://spectrum.ieee.org/computing/hardware/gordon-moore-the-man-whose-name-means-progress)],
but semiconductor manufacturers were able to keep up with Moore's law
until 2015. 

    In the folder "../data" is a comma separated value (CSV) file, "transistor_data.csv" [taken from wikipedia 01/2020](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors). 
    Load the csv into a pandas dataframe, it has the following headings:

    |Processor| MOS transistor count| Date of Introduction|Designer|MOSprocess|Area|
    |---|---|---|---|---|---|

    a. In the years 2017, what was the average MOS transistor count? 
    Make a boxplot of the transistor count in 2017 and find the first, second and third quartiles.

    b. Create a semilog y-axis scatter plot (i.e. `plt.semilogy`) for the 
    "Date of Introduction" vs "MOS transistor count". 
    Color the data according to the "Designer".

```{code-cell} ipython3
print('2.2a\n')

data_2017 = transistor_data[transistor_data['Date of Introduction'] == 2017]
average_MOS_count = data_2017['MOS transistor count'].mean()
print('The average transistor count was', average_MOS_count)

plt.boxplot(data_2017['MOS transistor count'])
plt.xlabel('2017')
plt.ylabel('# of Transistors')


transistor_count_2017 = data_2017['MOS transistor count']

quartiles = np.percentile(transistor_count_2017, q=[25, 50, 75])
print('The first quartile for the transistor count in 2017 is {}'.format(quartiles[0]))
print('The second quartile for the transistor count in 2017 is {}'.format(quartiles[1]))
print('The third quartile for the transistor count in 2017 is {}'.format(quartiles[2]))
```

```{code-cell} ipython3
print('2.2b')

len(transistor_data['Designer'].unique()), len(transistor_data)
transistor_data['Designer'].unique()
for designer in transistor_data['Designer'].unique():
    designer_data = transistor_data[transistor_data['Designer'] == designer]
    plt.semilogy(designer_data['Date of Introduction'], designer_data['MOS transistor count'],'s', label = designer)
plt.legend(bbox_to_anchor = (1, 1))
plt.title('Date of Introduction vs MOS transistor count\n')
plt.xlabel('Date of Introduction')
plt.ylabel('MOS transistor count')
```

## Problems [Part 3](03_Linear_Regression_with_Real_Data.md)

1. There is a csv file in '../data/primary-energy-consumption-by-region.csv' that has the energy consumption of different regions of the world from 1965 until 2018 [Our world in Data](https://ourworldindata.org/energy). 
Compare the energy consumption of the United States to all of Europe. Load the data into a pandas dataframe. *Note: you can get certain rows of the data frame by specifying what you're looking for e.g. 
`EUR = dataframe[dataframe['Entity']=='Europe']` will give us all the rows from Europe's energy consumption.*

    a. Plot the total energy consumption of the United States and Europe
    
    b. Use a linear least-squares regression to find a function for the energy consumption as a function of year
    
    energy consumed = $f(t) = At+B$
    
    c. At what year would you change split the data and use two lines like you did in the 
    land temperature anomoly? Split the data and perform two linear fits. 
    
    d. What is your prediction for US energy use in 2025? How about European energy use in 2025?

```{code-cell} ipython3
energy = pd.read_csv('../data/primary-energy-consumption-by-region.csv')
energy
```

```{code-cell} ipython3
print('3.1a.')
EUR = energy[energy['Entity']=='Europe']
USA = energy[energy['Entity']=='United States']
year_EUR = EUR['Year']
year_USA = USA['Year']
PEC_EUR = EUR['Primary Energy Consumption (terawatt-hours)']
PEC_USA = USA['Primary Energy Consumption (terawatt-hours)']
plt.figure(figsize=(10,7))
plt.plot(year_EUR, PEC_EUR, label='Europe')

plt.plot(year_USA, PEC_USA, label='United States')
plt.title('total energy consumption of the United States and Europe \n')
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (terawatt-hours)')
plt.legend()
plt.show()
```

```{code-cell} ipython3
print('3.1b.')

EURx=EUR['Year'].values
EURy=EUR['Primary Energy Consumption (terawatt-hours)'].values
EURx_mean=np.mean(EURx)
EURy_mean=np.mean(EURy)
USAx=USA['Year'].values
USAy=USA['Primary Energy Consumption (terawatt-hours)'].values
USAx_mean=np.mean(USAx)
USAy_mean=np.mean(USAy)

def coefficients(x, y, x_mean, y_mean):
    a_1 = np.sum(y*(x - x_mean)) / np.sum(x*(x - x_mean))
    a_0 = y_mean - a_1*x_mean
    
    return a_1, a_0

us_coeff = coefficients(USAx, USAy, USAx_mean, USAy_mean)
eu_coeff = coefficients(EURx, EURy, EURx_mean, EURy_mean)

us_a_1, us_a_0 = us_coeff
us_reg = us_a_0 + us_a_1 * USAx

eu_a_1, us_a_0 = eu_coeff
eu_reg = eu_a_0 + eu_a_1 * EURx

print('EUR: f(t) =  199.6 x - 3.76e+05')
print('USA: f(t) =  200.4 x - 3.765e+05')

EURa_1=np.sum(EURy*(EURx-EURx_mean))/np.sum(EURx*(EURx-EURx_mean))
EURa_0=EURy_mean-EURa_1*EURx_mean
EURreg=EURa_0+EURa_1*EURx
USAa_1=np.sum(USAy*(USAx-USAx_mean))/np.sum(USAx*(USAx-USAx_mean))
USAa_0=USAy_mean-USAa_1*USAx_mean
USAreg=USAa_0+USAa_1*USAx
plt.figure(figsize=(10,5))
plt.plot(yearE,PEC_E, label='Europe')
plt.plot(yearU,PEC_U, label='United States')
plt.plot(EURx, EURreg, '--', label='EUR lin reg')
plt.plot(USAx, USAreg, '-.', label='USA lin reg')
plt.title('Total Energy Consumption of the United States and Europe \n')
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (terawatt-hours)')
plt.legend()
```

```{code-cell} ipython3
print('3.1c')
#United States

a_1n, a_0n = np.polyfit(USAx, USAy, 1)
f_linear = np.poly1d((a_1n, a_0n))
f_linear = lambda x: a_1n*x+a_0n
plt.figure(figsize=(10, 5))
plt.plot(USAx, USAy,'s', color='#2929a3', linewidth=1, alpha=0.5,label='Energy Consumption')
plt.plot(USAx, f_linear(USAx), 'k--', linewidth=2, label='Linear regression')
plt.title('Energy Consumption vs. Year (USA)')
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (terawatt-hours)')
plt.legend(loc='best', fontsize=15)
plt.grid();
x = USA['Year'].values
y = USA['Primary Energy Consumption (terawatt-hours)'].values
mean_x = np.mean(x)
mean_y = np.mean(y)
```

3.1c. I would split the data at 1970 and at 2010. This would leave us with three lines, one from 1965 to 1969, 1970 to 2009, and then 2010 to 2020. At these years, the data starts showing different trends. For example, from 1965 to 1969, the slope is much steeper than from 1970 to 2009. The trend from 2010 to 2020 has a much smaller slope than the previous two.

```{code-cell} ipython3
x = USA['Year'].values
y = USA['Primary Energy Consumption (terawatt-hours)'].values

mean_x = np.mean(x)
mean_y = np.mean(y)

year = x
energy_consumed = y

year_1 , energy_consumed_1 = year[0:5], energy_consumed[0:5]
year_2 , energy_consumed_2 = year[5:45], energy_consumed[5:45]
year_3 , energy_consumed_3 = year[45:], energy_consumed[45:]
m1, b1 = np.polyfit(year_1, energy_consumed_1, 1)
m2, b2 = np.polyfit(year_2, energy_consumed_2, 1)
m3, b3 = np.polyfit(year_3, energy_consumed_3, 1)
f_linear_1 = np.poly1d((m1, b1))
f_linear_2 = np.poly1d((m2, b2))
f_linear_3 = np.poly1d((m3, b3))

plt.figure(figsize=(10, 5))
plt.plot(year, energy_consumed, color='#2929a3', linestyle='-', linewidth=1)
plt.plot(year_1, f_linear_1(year_1), 'r--', linewidth=2, label='1965-1969')
plt.plot(year_2, f_linear_2(year_2), 'g--', linewidth=2, label='1970-2009')
plt.plot(year_3, f_linear_3(year_3), 'b--', linewidth=2, label='2010-2020')
plt.title("Split Regression USA")
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (terawatt-hours)')
plt.legend(loc='best', fontsize=15)
plt.grid();
```

```{code-cell} ipython3
#Europe

a_1n, a_0n = np.polyfit(EURx, EURy, 1)
f_linear = np.poly1d((a_1n, a_0n))
f_linear = lambda x: a_1n*x+a_0n
plt.figure(figsize=(10, 5))
plt.plot(EURx, EURy,'s', color='#2929a3', linewidth=1, alpha=0.5,label='Energy Consumption')
plt.plot(EURx, f_linear(EURx), 'k--', linewidth=2, label='Linear regression')
plt.title('Linear Regression Line EUR')
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (terawatt-hours)')
plt.legend(loc='best', fontsize=15)
plt.grid();
x = USA['Year'].values
y = USA['Primary Energy Consumption (terawatt-hours)'].values
mean_x = np.mean(x)
mean_y = np.mean(y)
```

3.1c. I would split the data at 2010. These are the two spots in the data where the trajectory of the data seems to shift. This means that there would be three different trend lines: from 1965 to 1985, 1986 to 2010, and finally 2011 to 2020. The trend from 1965 to 1985 is the steepest, positive slope, the slope from 1986 to 2010 is flatter, and finally, from 2011 to 2020, the slope becomes negative.

```{code-cell} ipython3
x = EUR['Year'].values
y = EUR['Primary Energy Consumption (terawatt-hours)'].values

mean_x = np.mean(x)
mean_y = np.mean(y)

year = x
energy_consumed = y

year_1 , energy_consumed_1 = year[0:5], energy_consumed[0:5]
year_2 , energy_consumed_2 = year[5:45], energy_consumed[5:45]
year_3 , energy_consumed_3 = year[45:], energy_consumed[45:]
m1, b1 = np.polyfit(year_1, energy_consumed_1, 1)
m2, b2 = np.polyfit(year_2, energy_consumed_2, 1)
m3, b3 = np.polyfit(year_3, energy_consumed_3, 1)
f_linear_1 = np.poly1d((m1, b1))
f_linear_2 = np.poly1d((m2, b2))
f_linear_3 = np.poly1d((m3, b3))

plt.figure(figsize=(10, 5))
plt.plot(year, energy_consumed, color='#2929a3', linestyle='-', linewidth=1)
plt.plot(year_1, f_linear_1(year_1), 'r--', linewidth=2, label='1965-1969')
plt.plot(year_2, f_linear_2(year_2), 'g--', linewidth=2, label='1970-2009')
plt.plot(year_3, f_linear_3(year_3), 'b--', linewidth=2, label='2010-2020')
plt.title("Split Regression EUR")
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption (terawatt-hours)')
plt.legend(loc='best', fontsize=15)
plt.grid();
```

3.1d. As shown in the data for the USA energy consumption, the energy use is trending upwards, which means that more energy will be used in the coming years. On the other hand, the EUR data trends downwards, indicating the energy consumption will decrease. This shows that in the year 2025, the USA would be around 26000 terawatt-hours, whereas EUR would use about 22000 terawatt-hours of energy.

+++

2. You plotted Gordon Moore's empirical prediction that the rate of semiconductors on a computer chip would double every two years in [02_Seeing_Stats](./02_Seeing_Stats). This prediction was known as Moore's law. Gordon Moore had originally only expected this empirical relation to hold from 1965 - 1975 [[1](https://en.wikipedia.org/wiki/Moore%27s_law),[2](https://spectrum.ieee.org/computing/hardware/gordon-moore-the-man-whose-name-means-progress)], but semiconductor manufacuturers were able to keep up with Moore's law until 2015. 

Use a linear regression to find our own historical Moore's Law.    

Use your code from [02_Seeing_Stats](./02_Seeing_Stats) to plot the semilog y-axis scatter plot 
(i.e. `plt.semilogy`) for the "Date of Introduction" vs "MOS transistor count". 
Color the data according to the "Designer".

Create a linear regression for the data in the form of 

$log(transistor~count)= f(date) = A\cdot date+B$

rearranging

$transistor~count= e^{f(date)} = e^B e^{A\cdot date}$

You can perform a least-squares linear regression using the following assignments

$x_i=$ `dataframe['Date of Introduction'].values`

and

$y_i=$ as `np.log(dataframe['MOS transistor count'].values)`

a. Plot your function on the semilog y-axis scatter plot

b. What are the values of constants $A$ and $B$ for our Moore's law fit? How does this compare to Gordon Moore's prediction that MOS transistor count doubles every two years?

```{code-cell} ipython3
print('3.2a.')

data = pd.read_csv('../data/transistor_data.csv')
data = data.dropna()
data_file = data.drop(['Processor', 'MOSprocess', 'Area'], axis=1)
xi = data_file['Date of Introduction'].values
TC = np.log(data['MOS transistor count'].values)

for name in data_file['Designer'].unique():
    data_values = data_file[data_file['Designer'] == name]
    plt.semilogy(data_values['Date of Introduction'], data_values['MOS transistor count'], 's', label = name)
plt.title('Date of Introduction vs MOS transistor count')
plt.legend(bbox_to_anchor = (1,1))
```

```{code-cell} ipython3
print('3.2b.')

a_1n, a_0n = np.polyfit(xi, TC, 1)
f_linear = np.poly1d((a_1n, a_0n))
plt.plot(xi, np.exp(f_linear(xi)), label = 'Linear Regression')

for name in data_file['Designer'].unique():
    data_values = data_file[data_file['Designer'] == name]
    plt.semilogy(data_values['Date of Introduction'], data_values['MOS transistor count'], 's', label = name)
plt.title('Date of Introduction vs MOS transistor count')
plt.legend(bbox_to_anchor = (1,1))

print(f_linear, '\n')

print('The A and B values for our Moore\'s law fit are, A = {}, B = {}'.format(f_linear[1], f_linear[0]))
```

3.2b. This shows that Gordon Moore's prediction that MOS transistor count doubles every two years was correct. As shown in the graph above, the linear regression line seems to have the same trajectory as the rest of the MOS transistor data. 

+++

## Problems [Part 4](04_Stats_and_Montecarlo.md)

__1.__ [Buffon's needle problem](https://en.wikipedia.org/wiki/Buffon) is
another way to estimate the value of $\pi$ with random numbers. The goal
in this Monte Carlo estimate of $\pi$ is to create a ratio that is close
to [3.1415926...](http://www.math.com/tables/constants/pi.htm) _similar
to the example with darts points lying inside/outside a unit circle
inside a unit square._ 

![Buffon's needle for parallel
lines](https://upload.wikimedia.org/wikipedia/commons/f/f6/Buffon_needle.gif)

In this Monte Carlo estimation, you only need to know two values:
- the distance from line 0, $x = [0,~1]$
- the orientation of the needle, $\theta = [0,~2\pi]$

The y-location does not affect the outcome of crosses line 0 or not
crossing line 0. 

__a.__ Generate 100 random `x` and `theta` values _remember_ $\theta =
[0,~2\pi]$

__b.__ Calculate the x locations of the 100 needle ends e.g. $x_end = x
\pm \cos\theta$ _since length is unit 1. 

__c.__ Use 
[`np.logical_and`](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html)
to find the number of needles that have minimum $x_{end~min}<0$ and
maximum $x_{end~max}>0$. The ratio
$\frac{x_{end~min}<0~and~x_{end~max}>0}{number~of~needles} =
\frac{2}{\pi}$ _for large values of $number~of~needles$_.

__2.__ Build a random walk data set with steps between $dx = dy =
-1/2~to~1/2~m$. If 100 particles take 10 steps, calculate the number of
particles that move further than 0.5 m. 

_Bonus: Can you do the work without any `for`-loops? Change the size of
`dx` and `dy` to account for multiple particles._

```{code-cell} ipython3
print('4.1a\n')

from numpy.random import default_rng
rng = default_rng()

N = 100
x = rng.random(N)
theta = rng.random(N)*2*np.pi
print(x, '\n')
print(theta, '\n')
```

```{code-cell} ipython3
print('4.1b.\n')

x_end = x + np.array([np.cos(theta), -np.cos(theta)])

xleft = np.min(x_end, axis = 0)
xright = np.max(x_end, axis = 0)

print(xleft, '\n')
print(xright, '\n')

print('4.1c.\n')

predicted_value = np.sum(np.logical_and(xleft<0, xright>0))/N
print(predicted_value, 'result')
print(2/np.pi, 'predicted')

print(2/predicted_value, 'calculated pi')
```

```{code-cell} ipython3
from numpy.random import default_rng
rng = default_rng()
N_steps = 10
N_particles = 100
dx = rng.random((N_particles, N_steps)) - 0.5
dy = rng.random((N_particles, N_steps)) - 0.5
```

```{code-cell} ipython3
print('4.2.\n')

x = np.sum(dx, axis = 1)
y = np.sum(dy, axis = 1)

r = np.sqrt(x**2 + y**2)

print('{}% particles moved > 0.5 m from origin'.format(np.sum(r > 0.5)/len(r)*100))

plt.plot(x, y, 's')
plt.axis('equal')
```

__3.__ 100 steel rods are going to be used to support a 1000 kg structure. The
rods will buckle when the load in any rod exceeds the [critical buckling
load](https://en.wikipedia.org/wiki/Euler%27s_critical_load)

$P_{cr}=\frac{\pi^3 Er^4}{16L^2}$

where E=200e9 Pa, r=0.01 m +/-0.001 m, and L is the 
length of the rods supporting the structure. Create a Monte
Carlo model `montecarlo_buckle` that predicts 
the mean and standard deviation of the buckling load for 100
samples with normally distributed dimensions r and L. 

```python
mean_buckle_load,std_buckle_load=\
montecarlo_buckle(E,r_mean,r_std,L,N=100)
```

__a.__ What is the mean_buckle_load and std_buckle_load for L=5 m?

__b.__ What length, L, should the beams be so that only 2.5% will 
reach the critical buckling load?

```{code-cell} ipython3
def montecarlo_buckle(E,r_mean,r_std,L,N=100):
    '''Generate N rods of length L with radii of r=r_mean+/-r_std
    then calculate the mean and std of the buckling loads in for the
    rod population holding a 1000-kg structure
    Arguments
    ---------
    E: Young's modulus [note: keep units consistent]
    r_mean: mean radius of the N rods holding the structure
    r_std: standard deviation of the N rods holding the structure
    L: length of the rods (or the height of the structure)
    N: number of rods holding the structure, default is N=100 rods
    Returns
    -------
    mean_buckle_load: mean buckling load of N rods under 1000*9.81/N-Newton load
    std_buckle_load: std dev buckling load of N rods under 1000*9.81/N-Newton load
    '''
    r = r_mean + np.array([r_std, -r_std])
    critical_buckling_load = ((np.pi**3) * (E) * (r**4))/((16 * L**2))
    mean_buckle_load = np.mean(critical_buckling_load)
    std_buckle_load = np.std(critical_buckling_load)
    
    return mean_buckle_load, std_buckle_load
```

```{code-cell} ipython3
print('4.3a.\n')

E = 200 * 10**9
r_mean = 0.01
r_std = 0.001
L = 5
montecarlo_buckle(E, r_mean, r_std, L, N=100)
mean_buckle_load = montecarlo_buckle(E, r_mean, r_std, L, N=100)[0]
std_buckle_load = montecarlo_buckle(E, r_mean, r_std, L, N=100)[1]
print('The mean buckling load is {:.2f} N.'.format(mean_buckle_load))
print('The standard deviation of the buckling load is {:.2f}.'.format(std_buckle_load))
```

```{code-cell} ipython3
def montecarlo_buckle_length(E,r_mean,r_std,L,N=100):
    r = r_mean + np.array([r_std, -r_std])
    critical_buckling_load = ((np.pi**3) * (E) * (r**4))/((16 * L**2))
    length = np.sqrt(((np.pi**3) * (E) * (r**4))/((16 * (critical_buckling_load*0.025))))
    
    return length
```

```{code-cell} ipython3
print('4.3b.\n')
montecarlo_buckle_length(E, r_mean, r_std, L, N=100)
print('The length of the beam should be about {:.2f} m, so that only 2.5% of the beam will reach the critical buckling load.'.format(montecarlo_buckle_length(E, r_mean, r_std, L, N=100)[0]))
```
