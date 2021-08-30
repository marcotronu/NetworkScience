from utilities.preprocessing import *
import seaborn as sns
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import networkx as nx
import igraph as ig
# import numba
import itertools
from console_progressbar import ProgressBar
import re
from graph_tool.all import *
import graph_tool as gt
from math import *
import matplotlib
from scipy.stats import norm
import matplotlib.mlab as mlab
from colormap import rgb2hex
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
import scipy.stats as stats
import random 


'------------------------------------------------------------------------------------------------------------------'
def generate_plot(ars,labs,cols,dx,dy,tdx,tdy,pnorm,plog,addline,title,save,name,xlab,ylab,displaytext,dec,style):
    '''
    Create plot with some useful customizations.
    --------------------------------------------------------------------------
    Parameters:
        - ars: list of arrays to plot
        - labs: list of labels
        - cols: list of colors
        - dx: float, delta x for the text
        - dy: float, delta y for the text
        - tdx: float, adjustment for the text
        - tdy: float, adjustment for the text
        - pnorm: bool, if True add gaussian density
        - plog: bool, if True plot in log scale
        - addline: bool, if True add vertical lines;
        - title: string, title of the plot
        - save: bool, if True saves to -->
        - name: string, path in which to save the plot
        - xlab: string, label of the xaxis
        - ylab: string, label of the yaxis
        - displaytext: bool, if True display the text
        - dec: number of decimal places to consider in text value
        - style: style of the plot
    
    Returns:
        - None

    '''
    if not style:
        style = 'seaborn-paper'
    
    if not dec:
        dec = 4
        
    plt.style.use(style)

    fig = plt.figure()

    N = len(ars)

    if plog:
        for ar in range(N):
            ars[ar] = np.log10(ars[ar])

    x_axis = np.linspace(min([min(ar) for ar in ars]),max([max(ar) for ar in ars]))

    mus = []
    sigmas = []
    for ar in ars:
        (mu,sigma) = norm.fit(ar)
        mus.append(mu)
        sigmas.append(sigma)

    for i in range(N):
        ar = ars[i]
        col = cols[i]
        lab = labs[i]
        mu = mus[i]
        sigma = sigmas[i]

        plt.hist(ar,color = col,alpha=0.4,label=lab,density=pnorm)

        if pnorm:
            plt.plot(x_axis,norm.pdf(x_axis,mu,sigma), color = col)

        if addline:
            plt.vlines(mu,ymin=0,ymax=max(norm.pdf(x_axis,mu,sigma)),linestyles='dashed',color = col)
            pmu = int(10**mu) if int(10**mu) != 0 else np.around(10**mu,dec)
            plt.text(x = (1+dx)*mu,y = (1+dy)*max(norm.pdf(x_axis,mu,sigma)),s = '$\mu = ${}'.format(pmu),color=col)

    plt.title(title)
    plt.legend()
    plt.ylabel(ylab)
    plt.xlabel(xlab)

    if displaytext:
        if N > 1:
            textstr = ''.join((
                r'$mean(\frac{\mu_{d}}{\mu_{l}}) = $',
                '{}'.format(np.around(10**(mus[1]-mus[0]),4))
            ))

            props = dict(boxstyle = 'round',facecolor='wheat', alpha =0.5)

            plt.text(min(x_axis)*(1+tdx), max([max(norm.pdf(x_axis,mus[i],sigmas[i])) for i in range(N)])*(1+tdy), textstr,  fontsize=10,
                    verticalalignment='top', bbox=props)
                
    # plt.suptitle('Log10 distribution of likes & dislikes to views ratios')
    if save:
        plt.savefig('../images/'+name.split('.')[0]+'.pdf')

    # plt.show()
'------------------------------------------------------------------------------------------------------------------'
