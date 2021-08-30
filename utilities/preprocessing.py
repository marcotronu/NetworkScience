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




'----------------------------------------------------------------------------------------'
def build_edgesnodes(videos,categories,save=False,path=None):
    '''
    ----------------------------------------------------------------------------------------
    Takes a dataframe and returns the list of nodes and edges, and the dictionary mapping 
    for each tag the corresponding category
    ----------------------------------------------------------------------------------------
    Parameters:
        - videos: dataframe
        - categories: dictionary of the categories
        - save: if True saves to "path"
        - path: string

    Returns:
        - nodes, edges, dictionary of categories
    ----------------------------------------------------------------------------------------
    '''
    
    pb = ProgressBar(total=100,prefix='Done:', suffix='Now', decimals=0, length=50, fill='#', zfill='-')

    list_of_tags = [videos.channel_title.values[idx] + '|' + videos.tags.values[idx] for idx in range(len(videos))]
    list_of_tags = [re.sub("\"","",tag).split('|') for tag in list_of_tags]

    len_graph = []
    for vg in list_of_tags:
        len_graph.extend(vg)    

    if save and not path:
        raise ValueError('Please insert a valid path in which to save the nodes and edges!')
    elif save and path:
        if not os.path.exists(path):
            os.makedirs(path)
    
    nodes = {node:k for k,node in enumerate(list(set(len_graph)))}
    
    # print(list_of_tags)
    tagcategories = {}
    for row,tags in enumerate(list_of_tags):
        for tag in tags:
            cat_id = str(videos.iloc[row].category_id)
            if tag not in tagcategories:
                # tag_categories[tag] = [str(int(videos.iloc[row].category_id))]
                tagcategories[tag] = [categories[cat_id]]
            elif tag in tagcategories and categories[cat_id] not in tagcategories[tag]:
                # tag_categories[tag].append(str(int(videos.iloc[row].category_id)))
                tagcategories[tag].append(categories[cat_id])
        

    len_graph = len(set(len_graph))


    edges = []


    for counter,tags in enumerate(list_of_tags):
        progress = int((counter+1)/len(list_of_tags) * 100)
        pb.print_progress_bar(progress)     
        edges.extend(list(itertools.combinations(tags,2)))
        edges = list(set(edges))
    
    if save:
        '''
        Save nodes:
        '''
        nodes_df = pd.DataFrame(nodes.keys())
        nodes_df['Id'] = nodes.values()
        nodes_df.columns = ['Label','Id']
        nodes_df['Category'] = [tagcategories[tag] for tag in nodes_df.Label.values]
        nodes_df.drop(nodes_df.index[0],inplace=True)
        nodes_df = nodes_df.reindex(columns = ['Id','Label','Category'])
        nodes_df.drop(nodes_df[nodes_df.Id == 0].index, inplace = True)
        pd.DataFrame(nodes_df).to_csv('nodes.csv',index=False)

        '''
        Save edges:
        '''
        edges_df = pd.DataFrame([source[0] for source in edges])
        edges_df.columns = ['From']
        edges_df['To'] = [source[1] for source in edges]
        edges_df['Source'] = [nodes[source[0]] for source in edges]
        edges_df['Target'] = [nodes[source[1]] for source in edges]
        edges_df = edges_df.reindex(columns = ['Source','Target','From','To'])

        edges_df.drop(edges_df[edges_df.Target == 0].index, inplace = True)
        edges_df.drop(edges_df[edges_df.Source == 0].index, inplace = True)

        'Save nodes to nodes.csv'
        edges_df.to_csv('edges.csv',index=False)
        

    return nodes,edges,tagcategories
'----------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------'
def build_graph(nodes,edges,categories,directed=False):
    '''
    ----------------------------------------------------------------------------------------
    Use the python module graph-tool to build the graph.
    ----------------------------------------------------------------------------------------
    Parameters:
        - nodes: the dataframe containing the nodes
        - edges: the dataframem containting the edges
        - categories: the dictionary containing, for each node key, the corresponding category
        - directed: bool, if True the graph is directed
    Returns:
        - the graph
    ----------------------------------------------------------------------------------------
    '''
    g = gt.Graph(directed=directed)
    v_cat = g.new_vertex_property("string")
    v_lab = g.new_vertex_property("string")


    for j in range(0, len(nodes)):
        v = g.add_vertex()
        v_lab[v] = nodes.Label.values[j]
        v_cat[v] = categories[nodes.Label.values[j]]

    for i in range(0,len(edges)):

        v1 = edges.Source.values[i] #- 1 #you need to subtract -1 because the vertices IDs go from 0 to len(nodes) - 1
        v2 = edges.Target.values[i] #- 1 

        # if v1 not in list(v_lab):
        #     v = g.add_vertex()
        #     v_lab[v] = nodes.Label.values[v1]
        #     v_cat[v] = categories[nodes.Label.values[v1]]
        
        # if v2 not in list(v_lab):
        #     v = g.add_vertex()
        #     v_lab[v] = nodes.Label.values[v2]
        #     v_cat[v] = categories[nodes.Label.values[v2]]
        
        e = g.add_edge(v1,v2)
    return g
'----------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------'
def compute_gamma(graph,kmin):
    '''
    ----------------------------------------------------------------------------------------
    Computes gamma and C with the formula given by the power-law distribution.
    ----------------------------------------------------------------------------------------
    Parameters:
        - graph: graph to be considered;
        - kmin: min degree to consider to compute gamma, if None kmin will be min(degrees)
    Returns:
        - gamma, kmin, C
    ----------------------------------------------------------------------------------------
    '''
    vhist = graph.get_out_degrees(vs=list(graph.vertices()))

    if not kmin:
        kmin = min(vhist)

    true_k = vhist[vhist>=kmin]
    N = len(true_k)

    den = np.sum(np.log(vhist[vhist>=kmin]/kmin))

    gamma = 1 + N/den
    C = (gamma - 1)*kmin**(gamma - 1)
    
    return gamma,kmin,C
'----------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------'
def compute_n(graph,kmin):
    '''
    ----------------------------------------------------------------------------------------
    Computes the number of nodes with degrees over threshold.
    ----------------------------------------------------------------------------------------
    Parameters:
        - graph: it's the graph stored with graph-tool format
        - kmin:  the minimum node's degree before saturation 
    Returns:
        - list of N and list of largest hubs
    
    '''
    gamma,kmin,C = compute_gamma(graph,kmin)

    vhist = graph.get_out_degrees(vs=list(graph.vertices()))

    vhist = vhist[vhist>=kmin]

    N = []
    kmaxs = []
    for kmax in sorted(vhist):
        N.append(1/len(vhist[vhist>=kmax]))
        kmaxs.append(kmax)
    
    return N,kmaxs
'----------------------------------------------------------------------------------------'


'----------------------------------------------------------------------------------------'
def compute_degree_correlation_matrices(g):
    '''
    ----------------------------------------------------------------------------------------
    Return the 4 degree correlation matrices.
    ----------------------------------------------------------------------------------------
    Parameters:
        - g: graph
    Returns:
        - the four combinations of the degree correlations matrices: out_out, out_in, in_out, in_in
    ----------------------------------------------------------------------------------------
    '''

    #in --> in
    in_in = np.asarray([np.mean(g.get_in_neighbors(x)) for x in g.get_vertices()])
    in_degrees = g.get_in_degrees(g.get_vertices())

    ind = np.where(~np.isnan(in_in))[0]
    in_in = in_in[ind]
    in_degrees = in_degrees[ind]
    in_in = [in_degrees,in_in]

    #in --> out
    in_out = np.asarray([np.mean(g.get_in_neighbors(x)) for x in g.get_vertices()])
    out_degrees = g.get_out_degrees(g.get_vertices())

    ind = np.where(~np.isnan(in_out))[0]
    in_out = in_out[ind]
    out_degrees = out_degrees[ind]
    in_out = [out_degrees,in_out]

    #out --> out
    out_out = np.asarray([np.mean(g.get_out_neighbors(x)) for x in g.get_vertices()])
    out_degrees = g.get_out_degrees(g.get_vertices())

    ind = np.where(~np.isnan(out_out))[0]
    out_out = out_out[ind]
    out_degrees = out_degrees[ind]
    out_out = [out_degrees,out_out]

    #out --> in
    out_in = np.asarray([np.mean(g.get_out_neighbors(x)) for x in g.get_vertices()])
    in_degrees = g.get_in_degrees(g.get_vertices())

    ind = np.where(~np.isnan(out_in))[0]
    out_in = out_in[ind]
    in_degrees = in_degrees[ind]
    out_in = [in_degrees,out_in]

    return out_out, out_in, in_out, in_in
'----------------------------------------------------------------------------------------'