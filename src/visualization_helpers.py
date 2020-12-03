#-------------------------------------------------------------
# Keeps code for visualization pitch scapes in 2D or 3D space
#-------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import Isomap


def plot_by_groups(data, colours, type_name='colours', opacity=1.0, name=None, size=4, dim=2):
    '''
    Plots high-dimensional points in chosen space (2D or 3D) colorizing them
    by groups. It uses Isomap algorithm for projecting onto 2D or 3D space.
    Parameters:
        - data: array-like (number of samples, number of features)
          A high-dimensional data (pitch scapes)
        - colours: array-like (number of samples, )
          An array of digits or strings denoting affiliation to a certain group
        - type_name: str
          A name that characterizes given groups. Default: 'colours'
        - opacity: float or array-like (number of samples, )
          A parameter is in the interval [0, 1] that determines the visibility of points. 
          Default: 1.0
        - name: str
          A name that is used for saving the plot. If None is given, it doesn't save the plot.
          Default: None
        - size: int
          A parameter that defines the size of points. Default: 4
        - dim: int, 2 or 3
          A parameter that defines the dimension of the projected data. Default: 2     
    '''
    assert (dim == 2) or (dim == 3), 'The parameter dim must be 2 or 3'
    
    fig = plt.figure(figsize=(8, 8))
    data_proj = Isomap(n_components=dim).fit_transform(data) # gets projected data by means of Isomap
    if dim == 2:
        df = pd.DataFrame({type_name: colours, 'x':data_proj[:, 0], 'y':data_proj[:, 1], \
                           'size':size*np.ones(X.shape[0])}) # forms DataFrame for using plotly functions
        fig = px.scatter(df, x='x', y='y', color=type_name, opacity=opacity, size='size', size_max=size)
    else:
        df = pd.DataFrame({type_name: colours, 'x':data_proj[:, 0], 'y':data_proj[:, 1], 'z':data_proj[:, 2], \
                           'size':size*np.ones(X.shape[0])}) # forms DataFrame for using plotly functions
        fig = px.scatter_3d(df, x='x', y='y', z='z', color=type_name, opacity=opacity, size='size', size_max=size)
    if name != None:
        fig.write_html(name + ".html")
    fig.show()
    
def plot_transposition_with_centers(data, transposition, major_minor, text, assignments=None, type_name='colours', opacity=0.7, size=4, save=True):
    '''
    Plots data points in 3D space colorizing them by their transposition as well as centers of the found clusters.
    Both major and minor parts of points can be viewed either simultaneously or separately. 
    Parameters:
        - data: array-like (number of samples, number of features)
          A high-dimensional data (pitch scapes)
        - major_minor: array-like (number of samples, )
          An array that identifies from major or minor the point came 
        - transposition: array-like (number of samples, )
          A transposition of every points mapped in numbers
        - text: array-like (number of samples, )
          An arrayof strings denoting the key and the note of a certain pitch scape
        - assignments: array-like (number of samples, )
          An array of integer numbers that assigns points to a certain cluster. 
          If None is given, it doesn't plot centers. Default: None
        - type_name: str
          A name that characterizes given groups. Default: 'colours'
        - opacity: float or array-like (number of samples, )
          A parameter is in the interval [0, 1] that determines the visibility of points. 
          Default: 0.7
        - size: int
          A parameter that defines the size of points. Default: 4 
        - save: bool
          It saves the plot in HTML format when the parameter is True. Default: True
    '''
    
    data_proj = Isomap(n_components=3).fit_transform(data) # gets projected data by means of Isomap
    # Forms major and minor DataFrames
    df_major = pd.DataFrame({'note': transposition[major_minor==1], 'x': data_proj[major_minor==1, 0], \
                       'y': data_proj[major_minor==1, 1], 'z': data_proj[major_minor==1, 2], 'text': text[major_minor==1]})
    df_minor = pd.DataFrame({'note': transposition[major_minor==0], 'x': data_proj[major_minor==0, 0], \
                        'y': data_proj[major_minor==0, 1], 'z': data_proj[major_minor==0, 2], 'text': text[major_minor==0]})
    
    fig_data = [go.Scatter3d(x=df_major['x'], y=df_major['y'], z=df_major['z'], mode='markers', text=df_major['text'], \
                             marker=dict(size=size, color=df_major['note'], opacity=opacity)),
                go.Scatter3d(x=df_minor['x'], y=df_minor['y'], z=df_minor['z'], mode='markers', text=df_minor['text'], \
                             marker=dict(size=size, color=df_minor['note'], opacity=opacity))]
    name = 'major_minor_transposition'
    if np.all(assignments != None):
        # Finds cluster centers
        centers = []
        for i in np.unique(assignments):
            centers.append(np.mean(data_proj[assignments == i], axis=0))
        centers = np.array(centers)
        centers = pd.DataFrame({'x': centers[:, 0], 'y': centers[:, 1], 'z': centers[:, 2]})
        fig_data.append(go.Scatter3d(x=centers['x'], y=centers['y'], z=centers['z'], mode='markers', text='Center', \
                                     marker=dict(size=12, colorscale='Viridis', color=np.arange(centers.shape[0]),  \
                                     symbol='diamond', opacity=1.0)))
        name += '_with_centers_' + str(len(centers))
        
    fig = go.Figure(data=fig_data)
    fig.update_layout()
    if save:
        fig.write_html(name+'.html')
    fig.show()

# Here is auxiliary functions for plotting 
# the distribution of clusters over the circle of fifths
# They prepare the output of get_note_pairs_per_cluster for visualization

circle_of_fifths = ['A E', 'E B', 'B F#', 'F# C#', \     # The original circle use flats in its notation,
                    'C# G#', 'G# D#', 'D# A#', 'A# F',\  # but for implementation simplicity
                    'F C', 'C G', 'G D', 'D A']          # we abuse the fact that C# == Db


def clear_cluster_dict(cluster_dict):
    '''
    Removes notes pairs that have less than 5% of points in a given cluster
    Parameters:
        - cluster_dict: dictionary
          An output of get_note_pairs_per_cluster function
    Return:
        - updated_cluster_dict: dictionary
          An updated dictionary without unrepresented notes pairs in every cluster
    '''
    updated_cluster_dict = {}
    for cluster_num in cluster_dict:
        threshold = 0.05 * sum(cluster_dict[cluster_num].values())
        updated_dict = {}
        for key in cluster_dict[cluster_num]:
            if cluster_dict[cluster_num][key] >= threshold:
                updated_dict[key] = cluster_dict[cluster_num][key] 
        updated_cluster_dict[cluster_num] = updated_dict
    return updated_cluster_dict

def rename_keys(cluster_dict):
    '''
    Transform a tuple of notes to one string for better comprehension
    Parameters:
        - cluster_dict: dictionary
          An output of clear_cluster_dict function
    Return:
        - updated_cluster_dict: dictionary
          An updated dictionary with keys represented by one string 
    '''
    for cluster_num in cluster_dict:
        updated_dict = {}
        for key in cluster_dict[cluster_num]:
            updated_dict[key[0]+' '+key[1]] = cluster_dict[cluster_num][key]
        cluster_dict[cluster_num] = updated_dict
    return cluster_dict

def normalize(cluster_dict):
    '''
    Maps the number of notes pairs to [0, 10] interval
    Parameters:
        - cluster_dict: dictionary
          An output of rename_keys function
    Return:
        - updated_cluster_dict: dictionary
          An updated dictionary with the number of notes pairs scaled from 0 to 10 
    '''
    for cluster_num in cluster_dict:
        norm_constant = sum(cluster_dict[cluster_num].values())
        updated_dict = {}
        for key in cluster_dict[cluster_num]:
            updated_dict[key] = 10 * cluster_dict[cluster_num][key] / norm_constant
        cluster_dict[cluster_num] = updated_dict
    return cluster_dict

def prepare_radius(cluster_dict):
    '''
    Explicitly points out the number of pitch scapes for every pair of notes
    on the circle of fifths
    Parameters:
        - cluster_dict: dictionary
          An output of normalize function
    Return:
        - radius_values: array-like
          Numbers of pitch scapes for each notes pair on the circle of fifths in each cluster
    '''
    num_clusters = len(cluster_dict)
    num_notes_pairs = len(circle_of_fifths)
    radius_values = np.zeros((num_clusters, num_notes_pairs))
    for i, notes_pair in enumerate(circle_of_fifths):
        for key in cluster_dict:
            if notes_pair in cluster_dict[key]:
                radius_values[key, i] = cluster_dict[key][notes_pair]
            else:
                radius_values[key, i] = 0
    return radius_values

def plot_distr_over_circle(cluster_dict, max_radius=5):
    '''
    Plots the distribution of notes pairs over the circle of fifths
    for every obtained cluster
    Parameters:
        - cluster_dict: dictionary
          An output of get_note_pairs_per_cluster function
        - max_radius: float or int
          The maximum radius of the output circle. 
          One may adjust it to get a more artistic visualization 
    '''
    num_clusters = len(cluster_dict)
    cluster_dict_copy = cluster_dict.copy()
    cluster_dict_copy = clear_cluster_dict(cluster_dict_copy)
    cluster_dict_copy = rename_keys(cluster_dict_copy)
    cluster_dict_copy = normalize(cluster_dict_copy)
    radius_values = prepare_radius(cluster_dict_copy, circle_of_fifths)

    bar_polar_plots = []
    for i in range(num_clusters):
        bar_polar_plots.append(go.Barpolar(
                                              r=radius_values[i],
                                              theta=circle_of_fifths,
                                              width=np.ones(12),
                                              marker_color=i,
                                              marker_line_width=1,
                                              opacity=0.7
                                          ))

    fig = go.Figure(bar_polar_plots)
    fig.update_layout(
                      template=None,
                      polar = dict(
                          radialaxis = dict(range=[0, max_radius], showticklabels=False, ticks='',  showgrid=False),
                          angularaxis = dict(showticklabels=True, ticks='outside', direction = "clockwise",  rotation=0, showgrid=False),
                          barmode="overlay")
                     )
    fig.show()