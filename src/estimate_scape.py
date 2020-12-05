from pitchscapes.keyfinding import KeyEstimator
from pitchscapes.util import key_estimates_to_str

def estimate_scape(data):
    '''
    Estimates a given set of pitch scapes and returns their scores agreed
    to profiles, their belongins to a certain key, whether it is major
    or minor, and the last two thing in a string format  
    Parameters:
        - data: array-like (number of samples, number of features)
          A given set of pitch scapes
    Return:
        - scores: array-like (number of samples, major/minor, transposition)
          An array that gives information about transposition and whether
          it is major or minor
        - major_minor: array-like (number of samples,)
          An array that contains only 0 and 1 which correspondingly are minor
          and major
        - transposition: array-like (number of samples,)
          An array that contains only integer numbers from 0 to 11
          which corresponds to keys of pitch scapes
        - full_key: array-like (number of samples,)
          An array of strings that contains a full key information, 
          i.e. major/minor and transposition
    '''
    # defines key estimator
    k = KeyEstimator()
    # scores: Nx2x12 (data points x major/minor x transposition)
    scores = k.get_score(data)
    # best matching key: Nx2
    best_match = k.get_estimate(pitch_class_counts)
    major_minor = best_match[:, 0]
    transposition = best_match[:, 1]
    full_key = key_estimates_to_str(best_match, use_capitalisation=False)
    return scores, major_minor, transposition, full_key
