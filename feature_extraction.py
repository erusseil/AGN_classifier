import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from sklearn.metrics import mean_squared_error

# Dictionnary to convert filters to PLAsTiCC format
filter_dict = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'Y':5}


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



def convert_full_dataset(pdf: pd.DataFrame, passbands, obj_id_header='objectId'):
    """Convert an entire data set from mag to fluxcal.
    
    Parameters
    ----------
    pdf: pd.DataFrame
        Read directly from parquet files.
    passbands: list
        List of all passbands used in the dataset
    obj_id_header: str (optional)
        Object identifier. Options are ['objectId', 'candid'].
        Default is 'candid'.
        
    Returns
    -------
    pd.DataFrame
        Columns are ['objectId', 'MJD', 'FLT', 
        'FLUXCAL', 'FLUXCALERR'].
    """

    # hard code ZTF filters
    filters = passbands
    
    lc_flux_sig = []

    for index in range(pdf.shape[0]):

        name = pdf[obj_id_header].values[index]

        for f in range(1,3):
            
            
            filter_flag = np.array(pdf['cfid'].values[index]) == f
            mjd = np.array(pdf['cjd'].values[index])[filter_flag]
            mag = np.array(pdf['cmagpsf'].values[index])[filter_flag]
            magerr = np.array(pdf['csigmapsf'].values[index])[filter_flag] 

            fluxcal = []
            fluxcal_err = []
            for k in range(len(mag)):
                f1, f1err = mag2fluxcal_snana(mag[k], magerr[k])
                fluxcal.append(f1)
                fluxcal_err.append(f1err)
        
            for i in range(len(fluxcal)):
                lc_flux_sig.append([name, mjd[i], filters[f - 1],
                                    fluxcal[i], fluxcal_err[i]])

    lc_flux_sig = pd.DataFrame(lc_flux_sig, columns=['id', 'MJD', 
                                                     'FLT', 'FLUXCAL', 
                                                     'FLUXCALERR'])

    return lc_flux_sig


def plastic_format(data, passbands, obj_id_header='objectId'):
    
    """Transform a data set from fink format to PLAsTiCC format .
    
    Parameters
    ----------
    data: pd.DataFrame
        Fink data set.
    passbands: list
        List of all passbands used in the dataset
    Returns
    -------
    dataset: pd.DataFrame
        Data set in PLAsTiCC format
        Columns are ["object_id", "passband", "mjd", 
        "flux", "flux_err"]
    """
    
    # Convert magnitude to flux
    converted = convert_full_dataset(data, passbands, obj_id_header=obj_id_header)

    # Rename columns 
    transformed = converted.rename(columns={"id": "object_id",
                              "FLT": "passband",
                              "MJD": "mjd",
                              "FLUXCAL":"flux",
                              "FLUXCALERR":"flux_err"})
    
    # Add a fictiv detected_bool column to match PLAsTiCC format
    transformed['detected_bool'] = 1 

    # Rename bands to numbers
    transformed['passband'] = transformed['passband'].map(filter_dict)
    
    # Finally we remove lines containing NaN values
    remove_mask = ~transformed['flux'].isnull()
    dataset = transformed[remove_mask]

    return dataset


def clean_data(data, band_used):
    
    """
    Apply passband cuts, normalize flux and translate mjd to construct a standardize database.

    Parameters
    ----------
    data: pd.DataFrame 
        The light curve data from PLAsTiCC zenodo files.
    band_used: list
       List of all the passband you want to keep 
       (all possible bands : ['u','g','r','i','z','Y'].

    Returns
    -------
    clean: pd.DataFrame
        A normalized and mjd translated version of the data. 
        Contains only desired passbands
    """

    #--------------------------------------------------------------------------------

    #Filter the passband

    #---------------------------------------------------------------------------------

    to_fuse=[]
    
    for i in (pd.Series(band_used).map(filter_dict)):
        to_fuse.append(data.loc[data['passband']==i])

    clean = pd.concat(to_fuse)

    #---------------------------------------------------------------------------------

    # Translate the mjd

    #---------------------------------------------------------------------------------

    mintable = clean.pivot_table(index="passband", columns="object_id", values="mjd",aggfunc='min')
    mindf = pd.DataFrame(data=mintable.unstack())
    clean = pd.merge(mindf, clean, on=["object_id","passband"])
    clean['mjd'] = clean['mjd']-clean[0]
    clean = clean.drop([0],axis=1)

    #---------------------------------------------------------------------------------

    # Normalise the flux

    #---------------------------------------------------------------------------------
    
    maxtable = clean.pivot_table(index="passband", columns="object_id", values="flux",aggfunc='max')
    maxdf = pd.DataFrame(data=maxtable.unstack())
    clean = pd.merge(maxdf, clean, on=["object_id","passband"])
    clean['flux'] = clean['flux']/clean[0]
    clean['flux_err'] = clean['flux_err'] / clean[0]
    clean = clean.drop([0],axis=1)
    
    # Merge peaktable with data
    clean = pd.merge(maxdf, clean, on=["object_id","passband"])
    clean = clean.rename(columns={0: "peak"})
        
    return clean



#--------------------------------------------------------------------------------------------------------------------------------------------


def parametrise(clean, band_used, func, guess, minimum_points, original_shift):
    
    """Find best fit parameters for the polynomial model.
    
    Parameters
    ----------
    clean: pd.DataFrame
        Lightcurves dataframe to parametrize.
    band_used: list
       List of all the passband you want to keep 
       (all possible bands : ['u','g','r','i','z','Y'].
    func: python function
        Function used for the minimization
    guess: list
        Initial parameters used for the minimization
    minimum_points: int
        Minimum number of points requiered per passband.
        Else features extraction will output only 0 values.     
    original_shift: float
        Shift the mjd to this amount. Can be used if the function
        is able to fit under specific max-flux mjd value.
        Default is 0
    """
    
    band_used = pd.Series(band_used).map(filter_dict)
    
    np.seterr(all='ignore')
        
    # Get ids of the objects
    objects = np.unique(clean['object_id'])
    
    #Get the number of passband used
    nb_passband = len(band_used)
    
    # Get targets of the object
    target_list = np.array(clean.pivot_table(columns="object_id", values="target"))[0]
    
    clean = pd.merge(peaktable, clean, on=["object_id","passband"])
    clean = clean.rename(columns={0: "peak"})

                    
    # We initialise a table
    table = pd.DataFrame(data={'object_id': objects, 'target': target_list})      
        
        
    #####################################################################################
    
    # CONSTRUCT TABLE
    
    #####################################################################################  
    
    
    stdtable = clean.pivot_table(index="passband", columns="object_id", values="flux",aggfunc=lambda x : np.std(x))
    stddf = pd.DataFrame(data=stdtable.unstack())
    stddf.reset_index(inplace=True)

    counttable = clean.pivot_table(index="passband", columns="object_id", values="flux",aggfunc=lambda x : len(x))
    countdf = pd.DataFrame(data=counttable.unstack())
    countdf.reset_index(inplace=True)
    
    maxtable = clean.pivot_table(index="passband", columns="object_id", values="peak",aggfunc=lambda x : x.iloc[0])
    maxdf = pd.DataFrame(data=maxtable.unstack())
    maxdf.reset_index(inplace=True)
        
    for i in band_used:
        
        # Add extra parameters 
                
        table = pd.merge(maxdf.loc[maxdf['passband']==i, ['object_id', 0]], table, on=["object_id"])
        table = table.rename(columns={0: f"peak_{i}"})
        
        table = pd.merge(meandf.loc[meandf['passband']==i, ['object_id', 0]], table, on=["object_id"])
        table = table.rename(columns={0: f"mean_{i}"})
        
        table = pd.merge(countdf.loc[countdf['passband']==i, ['object_id', 0]], table, on=["object_id"])
        table = table.rename(columns={0: f"nb_points_{i}"})
        
        table = pd.merge(stddf.loc[stddf['passband']==i, ['object_id', 0]], table, on=["object_id"])
        table = table.rename(columns={0: f"std_{i}"})

    return table
