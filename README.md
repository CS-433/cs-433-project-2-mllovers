# Music beyond Major and Minor

In general, Western music is usually thought of as being either in a major or minor key. In reality, there are more sophisticated characteristics in the tonality of the piece. The aim of the project is to discover novel relevant keys, the ones that are beyond major and minor, which can provide fascinating insights.

Different statistical approaches represent powerful tools to explain and confirm the findings from the music theory. However, the numbers or labels that represent music pieces are not easy to retrieve. Here, we use a model of the [pitch scape representation](https://github.com/robert-lieck/pitchscapes) introduced by Robert Lieck, which is applicable to different music eras.

In our work, we focus on visualization techniques that can help in explaining the structure of the space of pitch-class distributions. Auxiliary tools for this are Dirichlet mixture model (DMM) and Isomap method.

- The data is stored in XML files and it is converted to pitch-class distributions connected to its metadata info. Since we could not upload the original data (due to size), we uploaded already processed CSV-s to dropbox, but we provide (in the main notebook) the way how we got CSV-s from original data. The dataset is private and belongs to DCML EPFL lab. The folder with the original data is organized by composers and files in subdirectories are in XML format. Additionally, there is 'metadata.csv' where some additional metadata information about the pieces can be found (e.g., year). 


### Running the model

Since running the model is computationally intensive for usual laptop machines, the code should be run in the cloud. For example, one can use [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CS-433/cs-433-project-2-mllovers/blob/main/main.ipynb). One can try running the code on a local machine. However, due to the lack of memory that we faced, the only way we recommend running the code is on Google colab. 
If you are trying to reproduce the code on a local machine, you will likely get an error that not enough workspace and array storage has been allocated. The first solution would be to plot fewer points, but in that case, the results will not be meaningful (plots can be displayed incorrectly and are not seen anywhere besides Google colab). Therefore, we strongly advise reproducing our work on Google colab.

The external libraries are listed in 'requirements.txt'. The main library pitchscapes was used on the development branch and it should be used with the following command: 
'! git clone -b develop https://github.com/robert-lieck/pitchscapes.git'

The notebook 'main.ipynb' contains the whole pipeline of installing requirements, getting the data and source code and it shows the most exciting findings.


### Repository structure

- The folder '/src' has the following files:
<br />'DMM.py' - Dirichlet mixture model class. There likelihoods and predictions are calculated. It is also used for the plotting of clusters. 
<br />'cluster_correlation_table.py' - Calculates the table for correlation checking between predicted clusters.
<br />'estimate_scape.py' - Estimates a given set of pitch scapes and returns their scores agreed to profiles, their belonging to a certain key, whether it is major or minor
<br />'get_note_pair_per_cluster.py' - Obtains the most significant profiles for the cluster (calculate frequencies of pairs for each clusters)
<br />'prepare_data.py' - Reads data, calculates pitches and connect it to metadata. Metadata that is used is the duration of pitch scape (relatively to the whole piece)
<br />'training.py' - Trains the model
<br />'visualization_helpers.py' - This script is used for all visualizations. First, it can be used for plotting high-dimensional points in the chosen space (2D or 3D), colorizing them by groups using Isomap. Next, it plots data points in 3D space, colorizing them by their transposition and centers of the found clusters. Both major and minor parts of points can be viewed either simultaneously or separately. Finally, it plots the distribution of note pairs over the circle of fifths for every obtained cluster.

- There is also 'main.ipynb' - a notebook that uses the whole code and shows all the visualizations. 


### Authors

- Aleksandr Timofeev: aleksandr.timofeev@epfl.ch
- Andrei Afonin: andrei.afonin@epfl.ch
- Dubravka Kutlesic: dubravka.kutlesic@epfl.ch
