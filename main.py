"""
# GARCH model on real data

The aim of this project is to modelize the return of a stock using a GARCH model.

"""## I. Importation : packages & data"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs
from arch import arch_model

fname = "CAC40.csv"
stock = pd.read_csv("data/{fname}".format(fname=fname), sep=";")
stock["Date"] = pd.to_datetime(stock["Date"], format="%d/%m/%Y")
stock = stock.set_index("Date")

"""## II. Weak white noise test and looking for an ARCH effect

Compute log-return and simple & partial autocorrelograms.
"""

# Log-returns
rstock = np.log(stock["Close"]).diff().dropna()  # drop NaN, diff for derivation
rstock.plot()

# Simple & partial autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(rstock, ax=ax1)
smt.graphics.plot_pacf(rstock, ax=ax2)

"""Test for the Zero Mean Hypothesis : `scs.ttest_1samp`"""

scs.ttest_1samp(rstock, 0.0)

"""H0 : times series mean different from 0. Rejected here implies mean is equal to 0.

Test the significativity of the correlations : `sm.stats.diagnostic.acorr_ljungbox`
"""

LB = sm.stats.diagnostic.acorr_ljungbox(rstock, lags=20)
plt.plot(range(1, 21, 1), LB[1], 'o')  # print p-values
plt.plot(range(1, 21, 1), 0.05 * np.ones(20))

"""H0 : nullity of the autocorrelations from 1 to h : h abscissa axis.

Some p-values are below 5% implies it is not a white noise.

To detect an ARCH effect, we need to do the same thing on the return squared and to perform an ARCH effect test by using `sm.stats.diagnostic.het_arch`.
"""

rstock2 = rstock ** 2
rstock2.plot()

# Simple & partial autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(rstock2, ax=ax1)
smt.graphics.plot_pacf(rstock2, ax=ax2)

"""There are autocorrelations between the squared log-return, it's not a strong white noise Maybe a GARCH because the squared return seem to be linearly dependant from the previous ones."""

sm.stats.diagnostic.het_arch(rstock)

"""H0 : no ARCH effect.

- LM (Lagrange multiplier)
- p-value
- Fisher statistic
- p-value of this statistic

Really low p-value, we reject H0 so H1 is considered true : there is an ARCH effect.
"""

from statsmodels.tsa.stattools import adfuller
adfuller(rstock2, autolag='AIC')

"""H1 is stationary : the squared return here is a stationary time series.

## III. Model estimation and selection of the best one

Estimate possible GARCH models using the function `arch_model` from the `arch` library.
"""

model = arch_model(rstock, p=1, q=1, mean='Zero')  # (1, 1) default value, mean=0 to remove any mean value and to remove one parameter in our model
result = model.fit()
result.summary()

"""t is our statistic, P>|t| represents the p-value. The value is really low for all our parameters.

Here, H0 : nullity of a coefficient. The null hypothesis is reject : *in this model, the coefficients are significants*
"""

# We can extract some key info
# result.loglikelihood
# result.aic
# result.bic
result.pvalues

"""**Remark :** We could have changed the conditional distribution by adding the parameter `dist`. The default value is the normal law.

There is also the Student law (`'t'`), asymetric Student law (`'skewt'`) and the generalized normal distribution (`'ged'`). To choose the best distribution, one needs to look at the qq-plot of the standardized residuals and compare it to the chosen conditional law.

X_{t}=\epsilon_{t}\sqrt(h_{t}), changing the distribution equal changing the law of epsilon_t
"""

residus = result.resid / result.conditional_volatility
sm.qqplot(residus, scs.norm, line="s")

"""The tail of the normal distribution is not adapted.

Let's look at the Student law :
"""

residus = result.resid / result.conditional_volatility
sm.qqplot(residus, scs.t, line="s", distargs=(8,))  # distargs=degree of freedom for the Student law (t)

"""Compare the models by looking at information criteria."""

model = arch_model(rstock, p=1, q=1, mean='Zero', dist='t')
result1 = model.fit()

model2 = arch_model(rstock, p=1, q=1, mean='Zero', dist='skewt')
result2 = model2.fit()

model3 = arch_model(rstock, p=1, q=1, mean='Zero', dist='ged')
result3 = model3.fit()

print("AIC : " + str(result1.aic) + " " + str(result2.aic) + " " + str(result3.aic))
print("BIC : " + str(result1.bic) + " " + str(result2.bic) + " " + str(result3.bic))
print("Log-L : " + str(result1.loglikelihood) + " " + str(result2.loglikelihood) + " " + str(result3.loglikelihood))

"""Student law seems to be better (lower AIC, BIC and higher Log-Likelihood)"""

result1.summary()

"""The parameter nu in the summary is the best degree of freedom for the Student Law (for QQ-plot).

Optional : Look the conditional volatility
"""

result.conditional_volatility.plot()

"""## IV. Best model : residuals analysis

Check the residuals' autocorrelograms & the associated Ljung-Box test. Check if the residuals are gaussians (histogram, qq-plot, tests).
"""

residuals = result.resid / result.conditional_volatility
residuals.plot()

# Autocorrelograms
fig = plt.figure(figsize=(15, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax2 = plt.subplot2grid(layout, (0, 1))
smt.graphics.plot_acf(residuals, ax=ax1)
smt.graphics.plot_pacf(residuals, ax=ax2)

LB = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=20)
plt.plot(range(1, 21, 1), LB[1], 'o')  # print p-values
plt.plot(range(1, 21, 1), 0.05 * np.ones(20))

"""Nothing below 5% for the significativity of the autocorrelograms : it seems to be a white noise."""

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

"""Check if there is no ARCH effect in the residuals."""

sm.stats.diagnostic.het_arch(residuals)

""""High" p-value (last of the four values), we accept H0, there is no ARCH effect.

## V. Predictions

Calculate predictions for the conditional volatility using our model.
"""

result.forecast(horizon=20).variance.tail(1)
