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
    
profiles = {
    "metadata": {
        "major": [0.16728176182048993, 0.02588894338557735, 0.1171404498466961, 0.0356495908305276,
                  0.11123089694868556, 0.09004464377374272, 0.0382659859401006, 0.16751964441157266,
                  0.03408440554363338, 0.09272715883619839, 0.03950089152204068, 0.08066562714073418],
        "minor": [0.1556462295111984, 0.03462755036633632, 0.09903157107324087, 0.1044334346698825,
                  0.048477125346135526, 0.10397079256257924, 0.034020023696857145, 0.15515432119981398,
                  0.07456760938348618, 0.05334836234206381, 0.08307367815831439, 0.05364930169009233]},
    "bayes": {
        "major": [0.19965448155346477, 0.008990613278155785, 0.12994202213313824, 0.01242816337359103,
                  0.12638864214266632, 0.08545131087389726, 0.0263734257934788, 0.20655914875192785,
                  0.013393451149384972, 0.08825555654425607, 0.01458718179430356, 0.08797600261173529],
        "minor": [0.17930236689047313, 0.017073963004507538, 0.10822708838359425, 0.11953464160572477,
                  0.025539463722006557, 0.11184032733554274, 0.020187990226227148, 0.181798715257532,
                  0.07218923098556179, 0.0341394852891066, 0.08257135177438128, 0.047595375525342154]},
    "krumhansl": {
        "major": [0.15195022732711172, 0.0533620483369227, 0.08327351040918879, 0.05575496530270399,
                  0.10480976310122037, 0.09787030390045463, 0.06030150753768843, 0.1241923905240488,
                  0.05719071548217276, 0.08758076094759511, 0.05479779851639147, 0.06891600861450106],
        "minor": [0.14221523253201526, 0.06021118849696697, 0.07908335205571781, 0.12087171422152324,
                  0.05841383958660975, 0.07930802066951245, 0.05706582790384183, 0.1067175915524601,
                  0.08941810829027184, 0.06043585711076162, 0.07503931700741405, 0.07121995057290496]},
    "temperley": {
        "major": [0.12987012987012989, 0.05194805194805195, 0.09090909090909091, 0.05194805194805195,
                  0.1168831168831169, 0.1038961038961039, 0.05194805194805195, 0.1168831168831169,
                  0.05194805194805195, 0.09090909090909091, 0.03896103896103896, 0.1038961038961039],
        "minor": [0.12987012987012989, 0.05194805194805195, 0.09090909090909091, 0.1168831168831169,
                  0.05194805194805195, 0.1038961038961039, 0.05194805194805195, 0.1168831168831169,
                  0.09090909090909091, 0.05194805194805195, 0.03896103896103896, 0.1038961038961039]},
    "albrecht": {
        "major": [0.23800000000000004, 0.006000000000000002, 0.11100000000000003, 0.006000000000000002,
                  0.13700000000000004, 0.09400000000000003, 0.016000000000000004, 0.21400000000000005,
                  0.009000000000000001, 0.08000000000000002, 0.008000000000000002, 0.08100000000000002],
        "minor": [0.22044088176352702, 0.0060120240480961915, 0.10420841683366731, 0.12324649298597193,
                  0.019038076152304607, 0.10320641282565128, 0.012024048096192383, 0.21442885771543083,
                  0.06212424849699398, 0.022044088176352703, 0.06112224448897795, 0.05210420841683366]}
}

notes = ['C Major', 'C Minor', 'C# Major', 'C# Minor', 'D Major', 'D Minor',  
         'D# Major', 'D# Minor', 'E Major', 'E Minor', 'F Major', 'F Minor',  
         'F# Major', 'F# Minor', 'G Major', 'G Minor', 'G# Major', 'G# Minor',  
         'A Major', 'A Minor', 'A# Major', 'A# Minor', 'B Major', 'B Minor']

def get_landmarks(landmark_name):
    '''
    Returns all possible transpositions of a given profile
    Parameters:
    - landmark_name: str
      A name of a profile
    Return:
    - landmarks: array-like
      An array of all possible transpositions (12 major and 12 minor) for a given landmark_name
    '''
    num_transp = 12
    orig_profile = np.stack([profiles[landmark_name]['major'], profiles[landmark_name]['minor']], axis=0).astype(np.float64)
    transp_landmarks = orig_profile
    for i in range(1, num_transp):
        transp_landmarks = np.concatenate([transp_landmarks, np.roll(orig_profile, i, -1)], axis=0)
    return transp_landmarks  

def plot_transposition_with_centers(data, transposition, major_minor, text, assignments=None, landmarks=False, \
                                    landmark_name='albrecht', type_name='colours', opacity=0.7, size=4, save=True):
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
        - landmarks: bool
          If this parameter is True, it plots landmarks for a chosen in landmark_name
          profile. Default: False
        - landmark_name: str
          It defines a profile which is used for plotting landmarks. 
          Options: "metadata", "bayes", "krumhansl", "temperley", "albrecht"
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
    isomap = Isomap(n_components=3)
    data_proj = isomap.fit_transform(data) # gets projected data by means of Isomap
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
                                     marker=dict(size=10, colorscale='Viridis', color=np.arange(centers.shape[0]),  \
                                     symbol='diamond', opacity=1.0)))
        name += '_with_centers_' + str(len(centers))
    if landmarks:
        landmarks = isomap.transform(get_landmarks(landmark_name))
        landmarks = pd.DataFrame({'x': landmarks[:, 0], 'y': landmarks[:, 1], 'z': landmarks[:, 2]})
        fig_data.append(go.Scatter3d(x=landmarks['x'], y=landmarks['y'], z=landmarks['z'], mode='markers+text', text=notes, \
                                     marker=dict(size=7, color='black',  \
                                     symbol='cross', opacity=1.0)))
        name += '_with_landmarks_' + str(len(landmarks))
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
