from matplotlib import pyplot as plt
import numpy as np

def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        setattr(top, i, j)
    return top
    

def show_heatmap(xs, ys, bins=20, weights=None, plt=plt, range=None):
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins, weights=weights, range=range)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto')

def show_heatmap_contours(xs, ys, bins=20, weights=None, plt=plt, range=None):
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins, weights=weights, range=range)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.contour(heatmap.T, extent=extent, origin='lower')

