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


md('# VARIATIONAL INFERENCE EXAMPLE')

def do_variational_inference(PRIOR=(5,1 , 1,1.5**2), ITER=1000000):
    µ0, λ0, α0, β0 = PRIOR ####### VARIATIONAL INFERENCE: prior
    
    ####### VARIATIONAL INFERENCE:  set inital values of the variational parameters using the prior
    μN = µ0
    σN = (β0/α0/λ0)**0.5
    αN = α0
    βN = β0

    ####### VARIATIONAL INFERENCE: constants
    Xsum = np.sum(X)
    X2sum = np.sum(X**2)
    #print(Xsum, X2sum, N)

    history = []
    history.append([μN,σN,αN,βN])

    PLOT_AT = [ITER-1]
    PLOT_PDF_AT = [0, 1, 2, 3, 4, ITER-1]
    PLOT_BROAD =  [0, 1, 2, 3]
    for i in range(ITER): ####### VARIATIONAL INFERENCE: loop
        # actually it is all at the same time
        αN,βN,μN,σN = (
            α0 + (N+1)/2,
            β0 + 0.5 * ( (λ0+N)*(σN**2+μN**2) - 2*(μ0*λ0+Xsum)*μN + X2sum + λ0*μ0**2 ),
            (Xsum + λ0*μ0) / (N + λ0),
            (βN/αN/(N+λ0))**0.5,
        ) ####### VARIATIONAL INFERENCE: closed form deterministic update
        history.append([μN,σN,αN,βN])

    for i in sorted(list(set(PLOT_AT + PLOT_PDF_AT))):
        μN,σN,αN,βN = history[i]
        if i in PLOT_PDF_AT:
            md('Current variational estimation q(θ) (heatmap) vs true posterior p(θ|X) (isolines) (and also without prior)')
            linspaceμ = default_plot.linspaceμ
            linspaceσ = default_plot.linspaceσ
            if i in PLOT_BROAD:
                linspaceμ = np.linspace(0, 6, 303)
                linspaceσ = np.linspace(0.01, 2, 301)
            pσ = invgamma.pdf(linspaceσ**2, αN, 0, βN)
            pμ = norm.pdf(linspaceμ[:,None], μN, σN)
            pμσ = pμ*pσ[None,:]

            mm = default_plot.mm
            plt.imshow(pμσ.T, extent=[*mm(linspaceμ), *mm(linspaceσ)], origin='lower', aspect='auto')
            #plt.colorbar()
            plot_true_distribution()
            plot_true_distribution(prior=lambda μ,σ: invgamma.pdf(σ**2, α0, 0, β0) * norm.pdf(μ, μ0, σ/λ0**0.5))
            plt.title("$q(\\mu, \\sigma)^{(color)}$ vs $p(\\mu, \\sigma | X)^{(isolines)}$, after "+str(i+1)+" iter.")
            plt.show()
 
        if i in PLOT_AT:
            md('Plotting the mean of the estimate accross iterations (we see a convergence)')
            h = np.array(history)
            μs = h[:,0]
            σs = (h[:,3]/h[:,2])**0.5
            plt.title("successive positions of the mean of $q(\\mu, \\sigma)$")
            plt.plot(μs, σs, alpha=0.6)
            plt.scatter(μs, σs, marker='x')
            plt.show()

    return obj_dic(locals())

md('## Not too bad prior and initialization')
vi = do_variational_inference()
md('## Poor prior and initialization')
# 42 virtual points at value 5, 
vi = do_variational_inference(PRIOR=(5,42 , 1,1.5**2), ITER=10)
md('## Fuzzy prior and initialization')
# 5 points at 5
vi = do_variational_inference(PRIOR=(5,5 , 1,1.5**2), ITER=10)







# In[ ]:




