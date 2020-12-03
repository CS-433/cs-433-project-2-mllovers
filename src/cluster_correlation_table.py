import pandas as pd
import numpy as np

def cluster_correlation_table(prediction, metadata):
    """
    Table for correlation checking between predicted clusters 
    and known metadata for sample points
    Parameters:
        - prediction: array (number of samples)
          Cluster predictions for every sample
        - metadata: array (number of samples)
          Metadata values of interest for every sample
    Return: pandas dataframe (Number of unique values of metadata, Number of unique values of prediction)
    Table with number of points corresponding to each pair (metadata, prediction)
    """
    dat = []
    # Cluster unique values
    pred_labels = np.unique(prediction)
    for i in range(len(pred_labels)):
        dat.append([])
    # Metadata unique values
    matadata_labels = np.unique(metadata)
    for i in range(len(pred_labels)):
        ind = (prediction == i)
        for j in matadata_labels:
           data =  metadata[ind]
           # save number of points from cluster i satisfying (matadata_label = j)
           dat[i].append(np.sum(data == j))
    df = pd.DataFrame(dat, index=np.unique(prediction), columns = matadata_labels).T
    return df