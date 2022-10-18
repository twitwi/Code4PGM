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
import xarray as xr

# %%
coordsA = {'A': ['bad', 'good']}
coordsF = {'F': ['unfit', 'fit']}
coordsR = {'R': ['muddy', 'dry']}
coordsT = {'T': ['poor', 'average', 'top']}
coordsQ = {'Q': ['fail', 'qualified']}
pF = xr.DataArray([0.7, 0.3], dims=('F'), coords=coordsF)
pA_F = xr.DataArray([[0.95, 0.05], [0.2, 0.8]], dims=('F', 'A'), coords={**coordsA, **coordsF})
pR = xr.DataArray([0.6, 0.4], dims=('R'), coords=coordsR)
pT_RF = xr.DataArray([[[0.3, 0.4, 0.3], [0.7, 0.25, 0.05]], [[0.02, 0.08, 0.90], [0.2, 0.3, 0.5]]], dims=('F', 'R', 'T'), coords={**coordsT, **coordsR, **coordsF}) # times/groups are in "reverse" order compared to Koller
pQ_T = xr.DataArray([[0.99, 0.01], [0.4, 0.6], [0.1, 0.9]], dims=('T', 'Q'), coords={**coordsT, **coordsQ}) # here it changes the order of the 3 rows

def normalize(a):
    return a / a.sum()

# derived joint probabilities for the exercices
pAF = pF * pA_F
pAFT = (pAF * pR * pT_RF).sum(dim='R')
p_all = pR*pF*pT_RF*pA_F*pQ_T

# %%
p_all.sum()

# %% [markdown]
# # Simple 2 variable case

# %%
pAF # joint distribution

# %%
# Q1
# proba of being fit, sum rule
print("A proportion of", pAF.sum(dim='A').sel(F='fit').values, "is fit")
# but we also have all values p(A)
pAF.sum(dim='A')

# %%
# Q2
# proba of being fit, knowing we did good at the aerobic test
# conditioning (taking just a value) and renormalizing
print("Given A=good, the proba of being fit is", normalize(pAF.sel(A='good')).sel(F='fit').values)
# but we also have all values p(F|A=good)
normalize(pAF.sel(A='good'))

# %%
# exploring
normalize(pAF.sel(A='bad'))

# %%
# more exploration
pAF.sum(dim='F')

# %% [markdown]
# # Three variables

# %%
print("shape:", pAFT.shape)
print("total size:", pAFT.size)
print(pAFT.sum().values)

pAFT

# %%
# Q3
# knowing there was a good test result A=good,
# what is the probability of getting a time in the top, T=top?
print("Given A=good, prob(T=top) =", normalize(pAFT.sel(A='good').sum(dim='F')).sel(T='top').values)
# the distribution (on T)
normalize(pAFT.sel(A='good').sum(dim='F'))

# %%
# let's check what we get with a bad test result
normalize(pAFT.sel(A='bad').sum(dim='F'))

# %%
# can we be fit but fail tests and races
normalize(pAFT.sel(A='bad', T='poor')) # ... not so much

# %%
# if we know we're fit, what is the probabily to perform at the race? 
normalize(pAFT.sel(F='fit').sum('A'))

# %%
# same, but knowing a test results
normalize(pAFT.sel(A='bad', F='fit'))
# (no change, so there is a conditional independance)
# (T seems indep of A, given F=fit)
# (already knowing F=fit, knowing A does not bring information on T)

# %%
# same with F='unfit'
normalize(pAFT.sel(A='bad', F='unfit')) / normalize(pAFT.sel(F='unfit').sum('A'))
# ... the ratio p(T|A=bad,F=unfit) / p(T|F=unfit) is 1 for all T

# %% [markdown]
# # All variables

# %%
print("shape:", p_all.shape)
print("total size:", p_all.size)
print(p_all.sum().values)

# %%
p_all

# %%
# Q4
# probability of getting qualified, given a good aerobic test (and nothing else)
normalize(p_all.sel(A='good').sum(dim=('R', 'F', 'T')))

# %%
# it is still low, so let's check the probability with a bad test result
normalize(p_all.sel(A='bad').sum(dim=('R', 'F', 'T')))

# %%
# and the "prior" probability with no information
p_all.sum(dim=('A', 'R', 'F', 'T'))

# %%

# %%
