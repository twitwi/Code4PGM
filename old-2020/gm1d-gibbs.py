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



md("# GIBBS SAMPLING EXAMPLE")

def do_gibbs_uninformative_prior():
    μ = 5  ####### GIBBS: initialize
    σ = 6  ####### GIBBS: initialize

    μs = [μ]
    σs = [σ]

    ITER = 20000 #*100
    BURN = 100
    PLOT_AT = [200 // 2]
    PLOT_NICE_AT = [0, 1, 2, 3, 4]
    md('### Running the Gibbs sampler, showing the successive conditional probabilities')
    for i in range(ITER//2):  ####### GIBBS: loop
        
        # scipy normal distribution uses the standard deviation
        μ = norm.rvs(np.mean(X), σ / N**0.5)  ####### GIBBS: one step
        μs.append(μ)
        σs.append(σ)
        
        σ = invgamma.rvs(N/2 - 1, 0, np.sum((X-μ)**2)/2) ** 0.5  ####### GIBBS: another step
        μs.append(μ)
        σs.append(σ)
        
        if i in PLOT_NICE_AT:
            minx = min(base_mean-0.3, np.min(μs))
            maxx = max(base_mean+0.3, np.max(μs))
            miny = min(base_stdev/1.3, np.min(σs))
            maxy = max(base_stdev*1.3, np.max(σs))
            ax = plt.subplot(3, 3, (4,8))
            plt.plot(μs, σs, alpha=0.6)
            plt.scatter(μs, σs, marker='x')
            
            plt.subplot(3, 3, (1,2), sharex=ax)
            x = np.linspace(minx, maxx, 151)
            plt.plot(x, norm.pdf(x, np.mean(X), σs[-3] / N**0.5))

            plt.subplot(3, 3, (6,9), sharey=ax)
            x = np.linspace(miny, maxy, 151)
            plt.plot(invgamma.pdf(x**2, N/2 - 1, 0, np.sum((X-μs[-2])**2)/2), x)

            ax.scatter(μs[-3:], σs[-3:], c='r')
            plt.show()
        if i in PLOT_AT:
            md('### Plotting after '+str(i)+' samples')
            plt.plot(μs, σs, alpha=0.2)
            plt.scatter(μs, σs, marker='.')
            plt.title('All samples')
            plt.show()

            plt.scatter(μs[BURN:], σs[BURN:], marker='x')
            plt.xlim(np.min(μs), np.max(μs))
            plt.ylim(np.min(σs), np.max(σs))
            plt.title('Samples after burn-in period')
            plt.show()
            
            plt.scatter(μs[BURN:], σs[BURN:], marker='x')
            plt.title('Samples after burn-in period (zoomed)')
            plt.show()


    print('Estimating the p(θ|X) as an histogram from samples (true posterior with lines)')
    show_heatmap(μs[BURN:], σs[BURN:], bins=25)
    plot_true_distribution()
    plt.show()
    return obj_dic(locals())

gibbs = do_gibbs_uninformative_prior()


