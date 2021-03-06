import pickle
from sklearn.ensemble import RandomForestClassifier
import kernel as k
import classifier as rfc
import feature_extraction as fe
import numpy as np
import pandas as pd


def load_classifier():
    """
    load the random forest classifier trained to recognize the AGN
    on binary cases : AGNs vs non-AGNs  (pickle format).
    
    Returns
    ----------
    RandomForestClassifier
        
    
    """
    with open(k.CLASSIFIER, 'rb') as f:
        clf = pickle.load(f)

    return clf


def agn_classifier(data):
    """
    call the agn_classifier

    Parameters
    ----------
    data : DataFrame
        alerts from fink with aggregated lightcurves
        
    Returns
    ----------
    np.array
        ordered probabilities of being an AGN
        Return -1 if the minimum number of point per passband is not respected
    """
    
    clean = fe.clean_data(data)
    converted = fe.convert_full_dataset(clean)
    transformed_1, transformed_2 = fe.transform_data(converted)
    
    features_1 = fe.parametrise(transformed_1, k.MINIMUM_POINTS, 1)
    features_2 = fe.parametrise(transformed_2, k.MINIMUM_POINTS, 2)
    
    features, valid = fe.merge_features(features_1, features_2)

    clf = rfc.load_classifier()
    
    proba = fe.get_probabilities(clf, features, valid)

    return proba


if __name__ == '__main__':

    data = pd.read_parquet('20alerts.parquet')
    data = data.iloc[:10]
    print(agn_classifier(data))

