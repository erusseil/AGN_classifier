import pickle
from sklearn.ensemble import RandomForestClassifier
import kernel as k
import classifier as rfc
import feature_extraction as fe
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def load_classifier():
    """
    load the random forest classifier trained to recognize the AGN.
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
    """
    
    id_order = data[['objectId']]
    
    data_plasticc = fe.plastic_format(data, k.BAND)
    clean = fe.clean_data(data_plasticc, k.BAND)
    features, validdf = fe.parametrise(clean, k.BAND, k.MINIMUM_POINTS, k.COLUMNS, id_order)

    clf = rfc.load_classifier()

    final_proba = np.array([-1]*len(features['object_id'])).astype(np.float64)

    agn_or_not = clf.predict_proba(features.loc[validdf['valid']].iloc[:, 1:])

    index_to_replace = features.loc[validdf['valid']].iloc[:, 1:].index

    final_proba[index_to_replace.values] = agn_or_not[:, 0]


    return final_proba


if __name__ == '__main__':

    data = pd.read_parquet('../20alerts.parquet')
    data = data.iloc[:100]
    print(agn_classifier(data))

