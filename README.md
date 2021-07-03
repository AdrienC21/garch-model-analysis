# Application of GARCH model on real data

GARCH model and statistical analysis on real financial data.

## Introduction

The aim of this project is to model the return of a stock/index using a GARCH model.

All the functions have been applied to the CAC40 index and are contained in the file main.py

You will find in this README file below the code with the corresponding plots (as it if was a notebook).

## Installation

Clone this repository :

```bash
git clone https://github.com/AdrienC21/garch-model-analysis.git
```

or just follow the guide.

Please make sure that the package `arch` is installed. If not, type in a python console :

```python
pip install git+https://github.com/bashtage/arch.git

# or type :
# pip install arch
```

## I. Importation : packages & data

```python
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs
from arch import arch_model
```

```python
fname = "CAC40.csv"
stock = pd.read_csv("data/{fname}".format(fname=fname), sep=";")
stock["Date"] = pd.to_datetime(stock["Date"], format="%d/%m/%Y")
stock = stock.set_index("Date")
```

## II. Weak white noise test and looking for an ARCH effect

Compute log-return and simple & partial autocorrelograms.

```python
# Log-returns
rstock = np.log(stock["Close"]).diff().dropna()  # drop NaN, diff for derivation
rstock.plot()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/logreturn.png?raw=true)

```python
# Simple & partial autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(rstock, ax=ax1)
smt.graphics.plot_pacf(rstock, ax=ax2)
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/autocorr.png?raw=true)

Test for the Zero Mean Hypothesis : `scs.ttest_1samp`

```python
scs.ttest_1samp(rstock, 0.0)
# Ttest_1sampResult(statistic=0.27375665934174925, pvalue=0.7842962457837591)
```

**H0** : times series mean different from 0. Rejected here which implies that the mean is equal to 0.

Test the significativity of the correlations : `sm.stats.diagnostic.acorr_ljungbox`

```python
LB = sm.stats.diagnostic.acorr_ljungbox(rstock, lags=20)
plt.plot(range(1, 21, 1), LB[1], 'o')  # print p-values
plt.plot(range(1, 21, 1), 0.05 * np.ones(20))
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/ljungbox.png?raw=true)

**H0** : nullity of the autocorrelations from 1 to h : h abscissa axis.

Some p-values are below 5%, this implies that it is not a white noise.

To detect an ARCH effect, we need to do the same thing on the return squared and to perform an ARCH effect test by using `sm.stats.diagnostic.het_arch`.

```python
rstock2 = rstock ** 2
rstock2.plot()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/squared.png?raw=true)

```python
# Simple & partial autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(rstock2, ax=ax1)
smt.graphics.plot_pacf(rstock2, ax=ax2)
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/autocorr2.png?raw=true)

There are autocorrelations between the squared log-return, it's not a strong white noise Maybe a GARCH because the squared return seem to be linearly dependant from the previous ones.

```python
sm.stats.diagnostic.het_arch(rstock)

"""
(255.05934691708347,
 5.5715023993261574e-39,
 10.51063633578267,
 1.1645836743832024e-41)
"""
```

**H0** : no ARCH effect.

- LM (Lagrange multiplier)
- p-value
- Fisher statistic
- p-value of this statistic

Really low p-value, we reject **H0** so **H1** is considered true : there is an ARCH effect.

```python
from statsmodels.tsa.stattools import adfuller
adfuller(rstock2, autolag='AIC')

"""
(-5.6907015812759925,
 8.083877104180183e-07,
 25,
 2272,
 {'1%': -3.4332314716515406,
  '10%': -2.56744765684667,
  '5%': -2.8628129627258123},
 -29414.886192176207)
"""
```

**H1** is stationary : the squared return here is a stationary time series.

## III. Model estimation and selection of the best one

Estimate possible GARCH models using the function `arch_model` from the `arch` library.

```python
model = arch_model(rstock, p=1, q=1, mean='Zero')  # (1, 1) default value, mean=0 to remove any mean value and to remove one parameter in our model
result = model.fit()
result.summary()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/GARCHmodelresult.png?raw=true)

t is our statistic, P>|t| represents the p-value. The value is really low for all our parameters.

Here, **H0** : nullity of a coefficient. The null hypothesis is reject : *in this model, the coefficients are significants*

```python
# We can extract some key info
# result.loglikelihood
# result.aic
# result.bic
result.pvalues

"""
omega       0.000000e+00
alpha[1]    5.845570e-01
beta[1]     8.957777e-10
Name: pvalues, dtype: float64
"""
```

**Remark :** We could have changed the conditional distribution by adding the parameter `dist`. The default value is the normal law.

There is also the Student law (`'t'`), asymetric Student law (`'skewt'`) and the generalized normal distribution (`'ged'`). To choose the best distribution, one needs to look at the qq-plot of the standardized residuals and compare it to the chosen conditional law.

<img src="https://latex.codecogs.com/gif.latex?X_{t}=\epsilon_{t}\sqrt(h_{t}) " /> , changing the distribution equal changing the law of <img src="https://latex.codecogs.com/gif.latex?\epsilon_{t} " /> 

```python
residus = result.resid / result.conditional_volatility
sm.qqplot(residus, scs.norm, line="s")
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/qqplot1.png?raw=true)

The tail of the normal distribution is not adapted.

Let's look at the Student law :

```python
residus = result.resid / result.conditional_volatility
sm.qqplot(residus, scs.t, line="s", distargs=(8,))  # distargs=degree of freedom for the Student law (t)
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/qqplot2.png?raw=true)

Compare the models by looking at information criteria.

```python
model = arch_model(rstock, p=1, q=1, mean='Zero', dist='t')
result1 = model.fit()

model2 = arch_model(rstock, p=1, q=1, mean='Zero', dist='skewt')
result2 = model2.fit()

model3 = arch_model(rstock, p=1, q=1, mean='Zero', dist='ged')
result3 = model3.fit()

print("AIC : " + str(result1.aic) + " " + str(result2.aic) + " " + str(result3.aic))
print("BIC : " + str(result1.bic) + " " + str(result2.bic) + " " + str(result3.bic))
print("Log-L : " + str(result1.loglikelihood) + " " + str(result2.loglikelihood) + " " + str(result3.loglikelihood))

"""
AIC : -14224.243715247943 -14222.243715247943 -14220.537859243548
BIC : -14201.284537414309 -14193.544742955899 -14197.578681409914
Log-L : 7116.121857623972 7116.121857623972 7114.268929621774
"""
```

Student law seems to be better (lower AIC, BIC and higher Log-Likelihood). Here, it corresponds to the variable `result1`.

```python
result1.summary()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/GARCHmodelresult2.png?raw=true)

The parameter nu in the summary is the best degree of freedom for the Student Law (for QQ-plot).

Optional : We can plot the conditional volatility

```python
result.conditional_volatility.plot()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/condvolat.png?raw=true)

## IV. Best model : residuals analysis

Check the residuals' autocorrelograms & the associated Ljung-Box test. Check if the residuals are gaussians (histogram, qq-plot, tests).

```python
residuals = result.resid / result.conditional_volatility
residuals.plot()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/resid.png?raw=true)

```python
# Autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(residuals, ax=ax1)
smt.graphics.plot_pacf(residuals, ax=ax2)
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/autocorr3.png?raw=true)

```python
LB = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=20)
plt.plot(range(1, 21, 1), LB[1], 'o')  # print p-values
plt.plot(range(1, 21, 1), 0.05 * np.ones(20))
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/ljungbox2.png?raw=true)

Nothing below 5% for the significativity of the autocorrelograms : it seems to be a white noise.

```python
import scipy

mu = 0
sigma = 1
# Histogram
n, bins, patches = plt.hist(residuals, 100, facecolor='blue', alpha=0.75, density=True)

# Gaussian fit
y = scipy.stats.norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

# Plot
plt.xlabel('log-return')
plt.ylabel('Density')
plt.title("Are residuals gaussians ?")
plt.show()
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/gauss.png?raw=true)

Check if there is no ARCH effect in the residuals.

```python
sm.stats.diagnostic.het_arch(residuals)

"""
(16.547452727483048,
 0.9416951232737653,
 0.6097552664318716,
 0.9428554956570107)
"""
```

"High" p-value (last of the four values), we accept **H0**, there is no ARCH effect.

## V. Predictions

Calculate predictions for the conditional volatility using our model.

```python
result.forecast(horizon=20).variance.tail(1)
```

![alt text](https://github.com/AdrienC21/garch-model-analysis/blob/main/images/forecast.png?raw=true)

## License

[MIT](https://choosealicense.com/licenses/mit/)
