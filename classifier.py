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

    data_plasticc = fe.plastic_format(data, k.BAND)
    clean = fe.clean_data(data_plasticc, k.BAND)
    features = fe.parametrise(clean, k.BAND, k.MINIMUM_POINTS, k.COLUMNS)

    clf = rfc.load_classifier()
    agn_or_not = clf.predict_proba(features.iloc[:, 1:])

    return agn_or_not[:, 0]




