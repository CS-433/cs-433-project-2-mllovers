import os
import numpy as np
import pandas as pd
import torch
from itertools import product
import pitchscapes.reader as rd
import pitchscapes.plotting as pt

def get_data_with_metadata(path_metadata= "metadata.csv",
                            data_dir= 'scores/Bach_JohannSebastian', 
                            prior_counts= 1,
                            n_times= 9, 
                            no_of_notes= 12, 
                            metaparam= ['filename','composer', 'no_notes', 'entropy', 'display_year',
                                'work_group', 'work catalogue','opus', 'no'],
                            report_broken_files= False,
                            store_csv = False):
    """
    Creates a dataframe from .xml files and metadata csv file
    For each music peace (n_times + 1)* n_times / 2 pitch scape is extracted.
    (with different time windows following triangle pattern i.e. first n_times points are from the smallest time window,
    the last point is the whole peace)
    First 12 columns are normalized pitch scape (for each time window pair), then metadata is matched with proper peace
    Peace identification is stored in song_id column

    Parameters:
        - path_metadata : string, path to metadata csv file
        - data_dir : string, path to directory with xml files
        - prior_counts: scalar, prior counts added to the aggregated values. Non-zero prior counts correspond to
                a Bayesian estimate with a uniform prior; the magnitude adjusts the prior strength. A value of None (default)
                does not add any prior counts. If return values are normalised (normalise=True) there is a difference between
                None and 0: A value of 0 is interpreted as the limitting case of non-zero prior counts, which means that
                outputs will always be normalised and if no data is available (zero counts, e.g. for zero-width time intervals
                or time intervals only zero values), a uniform distribution is returned. For a value of None, you will get an
                unnormalised all-zero output instead.
        - n_times : number of data points with smallest window size
        - no_of_notes : 12 
        - metaparam = [string], columns from metadata csv to be included in returned dataframe
        - report_broken_files: boolean, if True prints the details about broken files (for example xml files that can not be parsed)
        - store_csv : string, optional, path to csv where dataframe should be stored
    Return: 
        - dataframe that combines data and metadata
    """
    # metaparms need to contain filename for peace identification
    assert 'filename' in metaparam

    # dataframe from metadata.csv file 
    df_metadata = pd.read_csv(path_metadata, error_bad_lines=False, sep = '\t')

    # number of points in triangle
    data_points_per_scape = int(n_times * (n_times + 1) / 2)

    # to be converted to final dataframe 
    data = []
    # folder walk
    for r, _, f in os.walk(data_dir):
        for file in f:
            try:
                peace = np.zeros((data_points_per_scape, no_of_notes + 2))
                # read the score into a pitch scape
                scape = rd.get_pitch_scape(os.path.join(r, file), prior_counts= prior_counts)
                times = np.linspace(scape.min_time, scape.max_time, n_times + 1)
                cnt = 0
                if pitch_scape.min_time < 0:
                    continue
                for start, end in product(times, times):
                    if start >= end:
                        continue
                    pitch_scape = scape[start, end]
                    normalized_pitch_scape = pitch_scape / np.sum(pitch_scape) #.reshape(data_points_per_scape,1)
                    peace[cnt,:no_of_notes] = normalized_pitch_scape
                    peace[cnt,-2] = (end - start)
                    peace[cnt,-1] = (end - start) / (scape.max_time - scape.min_time)
                    cnt += 1
                
                # creating dictionary for data frame
                for i in range(peace.shape[0]): # iterate over scapes
                    # dataframe for specific peace
                    peace_dict = {}
                    for j in range(no_of_notes): # iterate over notes
                        # add pitch scape columns to peace dataframe 
                        peace_dict[str(j)] = peace[i][j]
                    df_peace = df_metadata.loc[df_metadata['filename'] == file]
                    # adding metadata parameters
                    for param in metaparam:
                        peace_dict[param] = df_peace[param].values[0]
                    data.append(peace_dict)
            except Exception as e:
                if report_broken_files:
                    print(os.path.join(r, file) + ": Error:  " + str(e))
                continue

    data = pd.DataFrame(data)
    # adding peace identification
    data['song_id'] = pd.factorize(data['filename'])[0]
    # removing NaNs from notes columns
    data = removeNaNs(data, no_of_notes)
    data = data.rename(columns={str(no_of_notes): "time_window_absolute", str(no_of_notes + 1): "time_window_normalized"})
    # save dataframe
    if store_csv:
        data.to_csv(data, float_format='%.15f')
    return data


#ignore zeros
def get_data_with_metadata_ignore_zeros(df_md,
                                        data_dir = 'scores/Bach_JohannSebastian',
                                        n_times= 9,
                                        no_of_notes = 12,
                                        metaparam = ['filename','composer', 'no_notes', 'entropy', 'display_year',
                                            'work_group', 'work catalogue','opus', 'no'],
                                        report_broken_files= False,
                                        store_csv = False):

    """
    Creates a dataframe from .xml files and metadata csv file
    For each music peace (n_times + 1)* n_times / 2 pitch scape is extracted.
    (with different time windows following triangle pattern i.e. first n_times points are from the smallest time window,
    the last point is the whole peace)
    All zeros are ignored
    First 12 columns are normalized pitch scape (for each time window pair), then metadata is matched with proper peace
    Peace identification is stored in song_id column
    
    Parameters:
        - path_metadata : path to metadata csv file
        - data_dir : path to directory with xml files
        - n_times : number of data points with smallest window size
        - no_of_notes : 12 
        - metaparam = [string], columns from metadata csv to be included in returned dataframe
        - report_broken_files: boolean, if True prints the details about broken files (for example xml files that can not be parsed)
        - store_csv : string, optional, path to csv where dataframe should be stored
    Return: 
        - dataframe that combines data and metadata
    """
    # metaparms need to contain filename for peace identification
    assert 'filename' in metaparam

    # number of points in triangle
    data_points_per_scape = int(n_times * (n_times + 1) / 2)

    # to be converted to final dataframe 
    data = []

    for r, _, f in os.walk(data_dir):
        for file in f:
            try:
                # extracting pitch scape
                pitch_scape = rd.get_pitch_scape(os.path.join(r, file))
                times = np.linspace(pitch_scape.min_time, pitch_scape.max_time, n_times)
                # initialization
                normalized_pitch_scape = np.zeros((data_points_per_scape, no_of_notes + 1))
                cnt = 0
                for start, end in product(times, times):
                    if end <= start:
                        continue
                    # extract_pitch_scape
                    data_point = np.array(pitch_scape[start, end])
                    data_point /= np.sum(data_point, axis = 0)
              
                    normalized_pitch_scape[cnt, :no_of_notes] = data_point
                    # adding time window length
                    normalized_pitch_scape[cnt,-1] = (end - start)
                    cnt += 1
                    
                pitch_scape_without_zeros = normalized_pitch_scape[np.any(normalized_pitch_scape== 0, axis = 1) == False]
                
                for i in range(pitch_scape_without_zeros.shape[0]):
                    peace_dict = {}
                    for j in range(no_of_notes + 1):
                        peace_dict[str(j)] = pitch_scape_without_zeros[i][j]
                    df_peace = df_md.loc[df_md['filename'] == file]
                    for param in metaparam:
                        peace_dict[param] = df_peace[param].values[0]
                    data.append(peace_dict)
            except Exception as e:
                if report_broken_files:
                    print(os.path.join(r, file) + ": Error:  " + str(e))
                continue

    data = pd.DataFrame(data) 
    # adding peace identification
    data['song_id'] = pd.factorize(data['filename'])[0]

    # name time_window column
    data = data.rename(columns={"12": "time_window"})
    # removing NaNs from notes columns
    data = removeNaNs(data, no_of_notes)
    # save dataframe
    if store_csv:
        data.to_csv(data, float_format='%.15f')
    return data


def removeNaNs(df, no_of_notes):
    """
    Remove rows with NaN values in notes columns from dataframe
    Parameters:
        - df: pandas dataframe, from which NaNs should be removed
    Return:
        - pandas dataframe without NaNs
    """
    note_columns = [str(i) for i in range(no_of_notes)]
    return df.dropna(subset= note_columns)
    