import numpy as np
from pitchscapes.keyfinding import KeyEstimator

def get_note_pairs_per_cluster(predictions, data):
    """
    Obtains the most significant profiles for the cluster (calculate frequencies of pairs for each clusters)
    Parameters:
        - predictions: tensor (number of samples)
        Cluster predictions for data 
        - data: tensor (number of samples, number of features)
        Input data(pitch scapes)
    Returns: dictionary (usage class_dictionary[index_of_the_class])
    """

    pitch_classes_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # identify classes
    classes = predictions.unique()
    # to be returned
    class_dictionary = {}
    for class_ in classes:
        empirical_array = data[predictions == class_]
        pairs = {}
        for empirical in empirical_array:
            empirical = empirical.cpu().detach().numpy()
            # get key estimates
            key_estimator = KeyEstimator()
            maj_min, tonic = key_estimator.get_estimate(empirical[None, :])[0]
            # get profile
            profile = key_estimator.profiles[maj_min]
            profile = np.roll(profile, shift=tonic)

            # the most significant 
            v1 = pitch_classes_sharp[np.argmax(profile)]
            # the second most significant
            v2 = pitch_classes_sharp[np.argsort(profile)[-2]]
            # calculate occurencies
            if (v1,v2) not in pairs:
                pairs[(v1, v2)] = 1
            else:
                pairs[(v1,v2)] = pairs[(v1,v2)] + 1
        #update dictionary
        class_dictionary[int(class_)] = pairs
    return class_dictionary