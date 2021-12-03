# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

# %%
gt_means = np.array([-2, 1])
gt_pi = np.array([.3, .7])
var = 1

N = 1000
X = np.random.normal(gt_means[1], var**0.5, N)
X[:int(gt_pi[0]*N)] = np.random.normal(gt_means[0], var**0.5, int(gt_pi[0]*N))
# NB: we generated with the exact proportions (a real generation would not necessarily have the exact proportions)

# %%
plt.hist(X, bins=100);


# %%

def plot_params(mu, w):
    x = np.linspace(-10, 10, 1001)
    plt.plot(x, w[0] * stats.norm.pdf(x, mu[0], var**0.5))
    plt.plot(x, w[1] * stats.norm.pdf(x, mu[1], var**0.5))
    plt.hist(X, density=True, bins=100, alpha=0.5)
    plt.show()

def E(X, mu, w):
    X = X[None,:]
    mu = mu[:, None]
    rprime = stats.norm.pdf(X, mu, var**0.5)
    return rprime / rprime.sum(axis=0)



# %%

mu = np.random.normal(0, 10, 2) # initiliaze the parameters
w = [0.5, 0.5] # initiliaze the parameters
plot_params(mu, w)
r = E(X, mu, w)
print(r)

# %%
print(mu)


# %%
def M(X, r):
    mu = np.sum(X*r, axis=1) / np.sum(r, axis=1)
    w = np.sum(r, axis=1) / X.size
    return mu, w

history = [(mu, w)]

mu, w = M(X, r)
plot_params(mu, w)
history.append((mu, w))


# %%
for i in range(10):
    r = E(X, mu, w)
    mu, w = M(X, r)
    plot_params(mu, w)
    history.append((mu, w))


# %%
print(mu)

# %%
print(w)

# %%
plt.plot([h[0] for h in history], label=["μ0", "μ1"])
plt.plot([0, len(history)], [[gt_means[0], gt_means[1]]]*2, '--', label=["gt μ0", "gt μ1"])
plt.legend()
plt.show()
plt.plot([h[1] for h in history], label=["π0", "π1"])
plt.plot([0, len(history)], [[gt_pi[0], gt_pi[1]]]*2, '--', label=["gt π0", "gt π1"])
plt.legend()
plt.show()


# %%
