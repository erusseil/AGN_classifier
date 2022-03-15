import pickle
from sklearn.ensemble import RandomForestClassifier
import kernel as k
import classifier as rfc
import feature_extraction as fe


def load_classifier():
    """
    load the random forest classifier trained to recognize the AGN.
    """
    with open("AGN_classifier/AGN_RF_classifier", 'rb') as f:
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
    features = fe.parametrise(clean, k.BAND, k.bump, k.guess_bump, k.MINIMUM_POINT, k.original_shift_bump)

    clf = rfc.load_classifier()
    agn_or_not = clf.predict_proba(features)
    return agn_or_not[:, 1]