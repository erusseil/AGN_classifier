import pickle
from sklearn.ensemble import RandomForestClassifier
import kernel as k
import classifier as rfc
import feature_extraction as fe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_classifier():
    """
    load the random forest classifier trained to recognize the AGN.
    """
    with open("AGN_RF_classifier", 'rb') as f:
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
    features = fe.parametrise(clean, k.BAND, k.bump, k.guess_bump, k.MINIMUM_POINT, 0)

    clf = rfc.load_classifier()
    agn_or_not = clf.predict_proba(features.iloc[:, 1:])

    features["proba"] = agn_or_not[:, 1]
    sample_with_proba = data.merge(features, left_on="candid", right_on="object_id")


    res = np.array([sample_with_proba["proba"][i] for i in range(len(sample_with_proba))])

    gb_X_test = sample_with_proba.apply(lambda x: is_AGN(x["fink_class"]), axis=1)

    x = np.array(np.array(res >= 0.5, dtype=np.int) == gb_X_test.to_numpy(), dtype=np.int).sum() / len(gb_X_test) * 100

    print(x)

    return sample_with_proba


def is_AGN(fink_class):
    if fink_class == "AGN":
        return 1
    else:
        return 0


if __name__ == "__main__":

    pdf = pd.read_parquet("../agn_sample.parquet")

    proba = agn_classifier(pdf)

    gb_X_test = proba.apply(lambda x: is_AGN(x["fink_class"]), axis=1)
    y_test = gb_X_test



    forest_model = load_classifier()
    X_test = proba.iloc[:, 8:-1]
    score = forest_model.score(X_test, y_test) * 100
    print("Precision: {}".format(score))

    target_dict = {0:'Other', 1:'AGN'}

    classes_test = np.unique(y_test)
    classe_names=[]
    for i in classes_test:
        classe_names.append(target_dict.get(i))
    fig = plt.figure(figsize=(19,15))
    ax = fig.add_subplot(111)
    plt.title(f'bump : confusion matrix. ~{round(score)} % accuracy',fontsize = 24)
    pred = forest_model.predict(X_test)
    # On calcul la matrice de confusion de nos fonctions
    conf_matrix = confusion_matrix(y_test, pred)
    #Show the conf matrix of Bazin
    c = sns.heatmap(conf_matrix, xticklabels=classe_names, yticklabels=classe_names,
                    annot=True, fmt="d",cbar=True, cmap="mako",linewidths=1,annot_kws={'size': 20});
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 14)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 14)
    plt.xlabel("\nPredicted label",fontsize = 18)
    plt.ylabel("Real label",fontsize = 18)
    interest = 'AGN'
    pos = np.argwhere(np.array(classe_names) == interest)[0,0]
    efficiency = (conf_matrix[pos,pos]/conf_matrix[pos,:].sum())
    purity = conf_matrix[pos,pos]/conf_matrix[:,pos].sum()
    print(f'Efficiency : {efficiency*100:.3}% of real {interest} were labeled as {interest}')
    print(f'Purity : {purity*100:.3}% of classified {interest} were indeed {interest}')
    # plt.savefig(f"Images/bump_confusion.png")
    plt.show()