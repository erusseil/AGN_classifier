import numpy as np
import pandas as pd

def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """ Conversion from magnitude to Fluxcal from SNANA manual
    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF.
    sigmapsf: float
        Error on PSF-fit magnitude from ZTF. 
    
    Returns
    ----------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)
    """

    if magpsf is None:
        return None, None

    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def remove_nan(x):
    """
    funtion that remove nan values from list contains in columns
    
    Paramters
    ---------
    x : pd.Series
        each rows of the dataframe. each entries must be numeric list
        
    Return
    ------
    list_without_nan : list
        list of the same size as x, each entries is the original list from the
        current rows without the nan values and the associated values from the other columns.
    """
    mask = np.equal(x['cmagpsf'], x['cmagpsf'])
    
    return [np.array(_col)[mask].astype(type(_col[0])) for _col in x]

def keep_filter(x, band):
    """
    funtion that remove filters other than g and r 
    
    Paramters
    ---------
    x : pd.Series
        each rows of the dataframe. each entries must be numeric list
        
    Return
    ------
    list_without_nan : list
        list of the same size as x, each entries is the original list from the
        current rows without the filter other than g and r and the associated values from the other columns.
    """
    
    mask = x['cfid']==band
    
    return [np.array(_col)[mask].astype(type(_col[0])) for _col in x]


def clean_data(pdf: pd.DataFrame):
    
        """
    Remove all nan values from 'cmagpsf' along with the corresponding values
    inside "cfid", "cjd", 'csigmapsf'.
    
    Paramters
    ---------
    pdf : pd.DataFrame 
        Dataframe of alerts from Fink
        Columns are : 'objectId', 'cjd', 'cmagpsf', 'csigmapsf', 'cfid'
        
    Return
    ------
    pdf_without_nan : pd.DataFrame
         DataFrame with nan and corresponding measurement removed
    """
        
        # Remove NaNs
        pdf[["cfid", "cjd", 'cmagpsf', 'csigmapsf']] = pdf[["cfid", "cjd", 'cmagpsf', 'csigmapsf']].apply(remove_nan, axis=1,result_type="expand")

        return pdf


def convert_full_dataset(pdf: pd.DataFrame):

    
    flux = pdf[['cmagpsf','csigmapsf']].apply(lambda x: mag2fluxcal_snana(x[0], x[1]), axis=1)
    pdf[['cmagpsf','csigmapsf']] = pd.DataFrame(flux.to_list())
    pdf = pdf.rename(columns={'cmagpsf':'cflux', 'csigmapsf':'csigflux'})
    
    return pdf

def translate(x):
    
    """Translate a cjd list by substracting min
    
    Parameters
    ----------
    x: np.array
        cjd array
        
    Returns
    -------
    np.array
        Translated array. Returns empty array if input was empty
    """
    
    if len(x)==0:
        return []
    
    else :
        return x - x.min()
    
    
def normalize(x, maxdf):
    
    """Normalize by dividing by a data frame of maximum
    
    Parameters
    ----------
    x: np.array
        Values to be divided
    max_value: float
        maximum value used for the normalisation
        
    Returns
    -------
    np.array
        Normalized array. Returns empty array if input was empty
    """
    
    max_value = maxdf[x.index]
    
    if len(x)==0:
        return []
    
    else :
        return x / max_value

def get_max(x):
    
    """Returns maximum of an array. Returns -1 if array is empty
    
    Parameters
    ----------
    x: np.array
        
    Returns
    -------
    float
        Maximum of the array or -1 if array is empty
    """
        
    if len(x)==0:
        return -1
    
    else :
        return x.max()
    


def transform_data(converted):
    
    # Create a dataframe with only measurement from band 1 
    transformed_1 = converted.copy()
    transformed_1[["cfid", "cjd", 'cflux', 'csigflux']] = transformed_1[["cfid", "cjd", 'cflux', 'csigflux']].apply(keep_filter, args=(1,), axis=1,result_type="expand")

    # Create a dataframe with only measurement from band 2
    transformed_2 = converted.copy()
    transformed_2[["cfid", "cjd", 'cflux', 'csigflux']] = transformed_2[["cfid", "cjd", 'cflux', 'csigflux']].apply(keep_filter, args=(2,), axis=1,result_type="expand")
    
    
    
    for df in [transformed_1, transformed_2]:
        
        df['cjd'] = df['cjd'].apply(translate)
        
        maxdf = df['cflux'].apply(get_max)
        
        df[['cflux']] = df[['cflux']].apply(normalize, args=(maxdf,))
        df[['csigflux']] = df[['csigflux']].apply(normalize, args=(maxdf,))
        
        df['peak'] = maxdf
    
    
    return transformed_1, transformed_2


def parametrise(transformed, minimum_points, band):

    
    nb_points = transformed['cflux'].apply(lambda x: len(x))
    peak = transformed['peak']
    std = transformed['cflux'].apply(np.std)
    ids = transformed['objectId']
  
    valid = nb_points>= minimum_points
    
    df_parameters = pd.DataFrame(data = {'object_id':ids,
                                         f'std_{band}':std,
                                         f'peak_{band}':peak,
                                         f'nb_points_{band}':nb_points,
                                         f'valid_{band}':valid})

    return df_parameters


def merge_features(features_1, features_2):
        
    features = pd.merge(features_1, features_2, on='object_id')
    valid = features['valid_1'] & features['valid_2']
    features = features.drop(columns=['valid_1', 'valid_2'])

    features = features[['object_id', 'std_1', 'std_2', 'peak_1', 'peak_2', 'nb_points_1', 'nb_points_2']]

    return features, valid

def get_probabilities(clf, features, valid):
    
    final_proba = np.array([-1]*len(features['object_id'])).astype(np.float64)

    agn_or_not = clf.predict_proba(features.loc[valid].iloc[:, 1:])

    index_to_replace = features.loc[valid].iloc[:, 1:].index

    final_proba[index_to_replace.values] = agn_or_not[:, 0]
    
    return final_proba
