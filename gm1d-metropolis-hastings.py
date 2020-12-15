# %%

from tools import obj_dic, show_heatmap, show_heatmap_contours
import numpy as np
from scipy.stats import norm, invgamma, beta
import scipy.stats
from matplotlib import pyplot as plt

md = lambda *args: print(*args)

np.random.seed(12345)




# Generate the Dataset as N draws from a normal distribution with base_mean and base_stdev

base_mean = 4
base_stdev = 1
N = 100
#X = np.random.normal(40, 10, (N, 1))
X = np.random.normal(base_mean, base_stdev, (N, 1))


# Look at the data
plt.scatter([],[])
plt.scatter(X[:,0], np.random.uniform(-1, 1, (N)))
plt.ylim(-5, 5)
plt.title(str(N)+" 1d observations (y-axis is random)")
plt.show()




# define a function to compute (on a grid, up to a constant factor) and draw the true posterior p(θ|X)

def plot_true_distribution(contours=True, cmap="gray",
                           minx=base_mean-0.3,
                           maxx=base_mean+0.3,
                           miny=base_stdev/1.3,
                           maxy=base_stdev*1.3,
                           prior=None):
    mm = lambda a: (np.min(a), np.max(a))
    Lprop = lambda μ, σ, X: 1/(2*np.pi*σ**2)**(X.shape[0]/2) * np.exp(- np.sum((X - μ)**2, axis=0, keepdims=True) / (2*σ**2))
    linspaceμ = np.linspace(minx, maxx, 101)
    linspaceσ = np.linspace(miny, maxy, 103)
    Lmap = Lprop(linspaceμ[None,:,None], linspaceσ[None,None,:], X[:,None])
    if prior is not None:
        Lmap *= prior(linspaceμ[:,None], linspaceσ[None,:])
    if contours:
        plt.contour((Lmap[0].T), extent=[*mm(linspaceμ), *mm(linspaceσ)], origin='lower', cmap=cmap)
    else:
        plt.imshow((Lmap[0].T), extent=[*mm(linspaceμ), *mm(linspaceσ)], origin='lower', aspect='auto')
        plt.colorbar()
    return obj_dic(locals())


# Show the true posterior

default_plot = plot_true_distribution(False)
plot_true_distribution()
plt.show()
plot_true_distribution()
plt.show()


md('# METROPOLIS-HASTINGS EXAMPLE')

def do_metropolis_hastings(prior=False, show=True, ITER=20000, INIT=(10,0.1)):
    μ, σ = INIT ####### METROPOLIS-HASTINGS: initialize

    μs = [μ]
    σs = [σ]
    rej_μs = []
    rej_σs = []

    # log likelihood function
    lnLproportional = lambda μ, σ: -(X.shape[0]*np.log(σ) + np.sum((X - μ)**2)/2/σ**2) # up to log((2π)^½N)
    lnLratio = lambda μn, σn, μ, σ: lnLproportional(μn, σn) - lnLproportional(μ, σ) # log(a/b) = log(a) - log(b)
    # log prior function
    # NB: non-conjugate prior, we have no contraint here
    lnprior = lambda μ, σ: -(μ-5)**2/.2 + -(σ-3)**2/.2  # kind of gaussian prior around μ=5, σ=2, pretty tight, so we see that even with 100 observations the prior impacts significantly what is the optimal p(μ,σ|X)
    
    if prior == "Funky": # softly "forbid" a region to force the sample to get around it
        # a improper prior that is almost 0 if σ≥2.5, and transitions to 1 at σ=1.5
        # the 100 is the importance of the prior compared to the data term lnLproportional
        lnprior = lambda μ, σ: 100*np.log(np.clip(2.5 - σ, 0.0001, 1))
        print("Funky")
        # warning: we could create bad local optimum with the prior, which can cause a lot of rejected samples
    
    BURN = 200
    PLOT_AT = [200, 400]
    if not show:
        PLOT_AT = [ITER-1]
        
    for i in range(ITER): ####### METROPOLIS-HASTINGS: loop
        μσvar = .2
        μnew, σnew = norm.rvs(μ, μσvar), norm.rvs(σ, μσvar) # using a proposal distribution that is a normal around the current parameters
        
        if prior is not False:
            paccept = np.exp(
                lnLproportional(μnew, σnew) - lnLproportional(μ, σ)
                + lnprior(μnew, σnew) - lnprior(μ, σ) # prior
            ) ####### METROPOLIS-HASTINGS: compute acceptance "probability" (might be > 1)
        else:
            #paccept = L(μnew, σnew, X) / L(μ, σ, X) # no prior, not numerically stable
            #paccept = np.exp( lnLproportional(μnew, σnew) - lnLproportional(μ, σ)   )  # no prior
            paccept = np.exp(lnLratio(μnew, σnew, μ, σ)) # no prior
        
        if np.random.uniform(0, 1) < paccept: ####### METROPOLIS-HASTINGS: draw the acceptance
            μ, σ = μnew, σnew ####### METROPOLIS-HASTINGS: accepted
        else:
            rej_μs.append(μnew)
            rej_σs.append(σnew)
        μs.append(μ)
        σs.append(σ)

        if i in PLOT_AT:
            plt.plot(μs, σs, alpha=0.6)
            if show:
                plt.scatter(μs, σs, marker='x')
                plt.scatter(rej_μs, rej_σs, marker='x', c='r')
                show_heatmap_contours(norm.rvs(μ, μσvar, 10000), norm.rvs(σ, μσvar, 10000))
                plt.show()
            if i > BURN:
                plt.scatter(μs[BURN:], σs[BURN:], marker='x')
                if show:
                    plt.show()
            
    if show:
        show_heatmap(μs[BURN:], σs[BURN:], bins=25)
        plot_true_distribution()
        plt.show()
    md("Accepted " + str(len(μs)-len(rej_μs)) + " and rejected " + str(len(rej_μs)))
    return obj_dic(locals())


md("## Showing accepted samples and rejected samples (1 chain)")
md("### No prior")
mh = do_metropolis_hastings()
md("### Prior that biases the estimate")
mh = do_metropolis_hastings(True)
md("### Prior that forbids σ > 2.5")
mh = do_metropolis_hastings("Funky")


md("## Draw a few chains")
for setup in [False, True, 'Funky']:
    for i in range(10):
        do_metropolis_hastings(setup, show=False, ITER=500, INIT=(norm.rvs(10, 2), np.abs(norm.rvs(1, 2))))
    plt.show()


