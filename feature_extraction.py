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
    funtion that removes points fron other bands than the one specified
    
    Parameters
    ---------
    x : pd.Series
        each rows of the dataframe. each entries must be numeric list
        
    Return
    ------
    list_with_oneband : list
        list of the same size as x, each entries is the original list from the
        current rows with only the wanted filter and the associated values from the other columns.
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
        
    Return
    ------
    pdf_without_nan : pd.DataFrame
         DataFrame with nan and corresponding measurement removed
    """
    
    pdf = pdf.reset_index(drop=True)
    
    # Remove NaNs
    pdf[["cfid", "cjd", 'cmagpsf', 'csigmapsf']] = pdf[["cfid", "cjd", 'cmagpsf', 'csigmapsf']].apply(remove_nan, axis=1,result_type="expand")
    
    return pdf


def convert_full_dataset(clean: pd.DataFrame):
    """
    Convert all mag and mag err to flux and flux err
    
    Paramters
    ---------
    clean : pd.DataFrame 
        Dataframe of alerts from Fink with nan removed
        
    Return
    ------
    pd.DataFrame
         DataFrame with column cmagpsf replaced with cflux 
         and column csigmagpsf replaced with csigflux
    """

    
    flux = clean[['cmagpsf','csigmapsf']].apply(lambda x: mag2fluxcal_snana(x[0], x[1]), axis=1)
    clean[['cmagpsf','csigmapsf']] = pd.DataFrame(flux.to_list())
    clean = clean.rename(columns={'cmagpsf':'cflux', 'csigmapsf':'csigflux'})
    
    return clean

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
    
def compute_snr(pdf):
    """Compute a new column : signal to noise ratio
    
    Parameters
    ----------
    pdf: pd.DataFrame 
        Dataframe of alerts from Fink with nan removed and converted to flux
        Must contain columns ['cflux', 'csigflux']
        
    Returns
    -------
    pd.Series
        Signal to noise ratio
    """
    return pdf['cflux']/pdf['csigflux']
    


def transform_data(converted):
    
    """ Apply transformations for each band on a flux converted dataset
            - Shift cjd so that the first point is at 0
            - Normalize by dividing flux and flux err by the maximum flux
            - Add a column with maxflux before normalization
    
    
    Parameters
    ----------
    converted : pd.DataFrame 
        Dataframe of alerts from Fink with nan removed and converted to flux
        
    Returns
    -------
    transformed_1 : pd.DataFrame
        Transformed DataFrame that only contains passband g
        
    
    transformed_2 : pd.DataFrame
        Transformed DataFrame that only contains passband r
        
    """
    
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
        df['snr'] = df[['cflux', 'csigflux']].apply(compute_snr, axis=1)
        
        df['peak'] = maxdf
    
    return transformed_1, transformed_2


def parametrise(transformed, minimum_points, band, target_col='', mean_snr=False, std_snr=False):
    
    """Extract parameters from a transformed dataset. Construct a new DataFrame
    Parameters are :  - 'nb_points' : number of points
                      - 'std' : standard deviation of the flux
                      - 'peak' : maximum before normalization
                      - 'valid' : is the number of point above the minimum (boolean)
    
    Parameters
    ----------
    transformed : pd.DataFrame
        Transformed DataFrame that only contains a single passband 
    minimum_points : int
        Minimum number of points in that passband to be considered valid
    band : int
        Passband of the dataframe
    target_col: str
        If inputed a non empty str, add the corresponding column as a target column
        
    Returns
    -------
    df_parameters : pd.DataFrame
        DataFrame of parameters with columns 
        ['object_id', 'std', 'peak', 'nb_points', 'valid']
    """

    
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
    
    if mean_snr :
        mean_snr = transformed['snr'].apply(np.mean)
        df_parameters[f'mean_snr_{band}'] = mean_snr
        
    if std_snr :
        std_snr = transformed['snr'].apply(np.std)
        df_parameters[f'std_snr_{band}'] = std_snr
    
    if target_col != '':
        targets = transformed[target_col]
        df_parameters['target'] = targets

        
    return df_parameters


def merge_features(features_1, features_2, target_col='', mean_snr=False, std_snr=False):
    
    """Merge feature tables of band g and r. 
    Also merge valid columns into one
    
    Parameters
    ----------
    features_1: pd.DataFrame
        features of band g
    features_2: pd.DataFrame
        features of band r
    target_col: str
        If inputed a non empty str, add the corresponding column as a target column
        
    Returns
    -------
    features : pd.DataFrame
        Merged features with valid column dropped
        
    valid : np.array
        Bool array, indicates if both passband respect the minimum number of points
    """
        
    # Avoid having twice the same column
    
    if target_col == '':
        features_2 = features_2.drop(columns={'object_id'})
    else:
        features_2 = features_2.drop(columns={'object_id', target_col})
    
    features = features_1.join(features_2)
    valid = features['valid_1'] & features['valid_2']
    features = features.drop(columns=['valid_1', 'valid_2'])
    

    ordered_features = features[['object_id', 'std_1', 'std_2', 'peak_1', 'peak_2', 'nb_points_1', 'nb_points_2']].copy()
    
    
    if mean_snr :
        ordered_features['mean_snr_1'] = features['mean_snr_1']
        ordered_features['mean_snr_2'] = features['mean_snr_2']
        
    if std_snr :
        ordered_features['std_snr_1'] = features['std_snr_1']
        ordered_features['std_snr_2'] = features['std_snr_2']
    
    if target_col != '':
        ordered_features[target_col] = features[target_col]
    
        
    return ordered_features, valid

def get_probabilities(clf, features, valid):
    
    """Returns probabilty of being an AGN predicted by the classfier
    
    Parameters
    ----------
    clf: RandomForestClassifier
        Binary AGN vs non AGN classifier
        
    features: pd.DataFrame
        Features extracted from the objects. 
        Outputed by merge_features
        
    valid: np.array
        Bool array, indicates if both passband respect the minimum number of points
        
    Returns
    -------
    final_proba : np.array
        ordered probabilities of being an AGN
        
    """
    
    final_proba = np.array([-1]*len(features['object_id'])).astype(np.float64)

    agn_or_not = clf.predict_proba(features.loc[valid].iloc[:, 1:])

    index_to_replace = features.loc[valid].iloc[:, 1:].index

    final_proba[index_to_replace.values] = agn_or_not[:, 0]
    
    return final_proba
