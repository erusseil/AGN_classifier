import numpy as np
import pandas as pd
import timeit
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


def parametrise(clean, band_used, func, guess, minimum_points, original_shift = 0, checkpoint='', begin=0):
     
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
    checkpoint : str 
        The table is saved each time a line is calculated, 
        if a problem occured you can put the partially filled 
        table as a check point. With the right 'begin' it 
        avoids recalculating from start.
    begin : int
        Start table construction at this line 
    """
    
    band_used = pd.Series(band_used).map(filter_dict)
    
    np.seterr(all='ignore')

    # Get ids of the objects
    objects = np.unique(clean['object_id'])
 
    # list of parameter names
    param_names = func.__code__.co_varnames[1:len(guess)+1] 

    # Add the 3 extra parameters in the list of parameters name
    param_names = np.append(param_names,'cost')
    param_names = np.append(param_names,'nb_points') 
    param_names = np.append(param_names,'peak')
    
    total_parameters = len(band_used)*(len(param_names))
        
    #####################################################################################
    
    # INITIALISE PARAMETER TABLE    
    
    #####################################################################################
    
    
    # We initialise a table of correct size but filled with 0    
    if checkpoint == '' :               

        df = {'object_id': objects}
        
        table = pd.DataFrame(data=df)    
        for i in range(total_parameters):
            table[i]=0
            
    # Or we start from checkpoint if provided
    else:
        table = checkpoint
        
    # Rename the table 
    for idx1,i in enumerate(band_used):
        for idx2,j in enumerate(param_names):
            table = table.rename(columns={idx2+idx1*len(param_names):(j+'_'+str(i))})     
          
    #####################################################################################
    
    # COMPUTE NUMBER OF POINTS PER OBJECTS PER PASSBAND
    
    #####################################################################################          
        
     # Create a table describing how many points exist for each bands and each object   
    counttable = clean.pivot_table(index="passband", columns="object_id", values="mjd",aggfunc=lambda x: len(x))
    
    # Create a table describing how many bands are complete for each object
    df_validband = pd.DataFrame(data={'nb_valid' : (counttable>=minimum_points).sum()})

    clean = pd.merge(df_validband,clean, on=["object_id"])

        
    #####################################################################################
    
    # CONSTRUCT TABLE
    
    #####################################################################################  
    
    start = timeit.default_timer()
        
    print("Number of objects to parametrize : ",len(objects)-begin,"\n")
    
    # i and j loop over all light curves for all objects
    for j in range(begin,len(objects)):

        print(j, end="\r")
        
        ligne=[]
        ide=objects[j]
        
        # Only performes the fit if the number of points is respected
        if clean.loc[clean['object_id'] == ide,'nb_valid'].iloc[0] == len(band_used):
            
            for i in band_used:

                # Store point's information of the objects
                obj = clean.loc[(clean['object_id'] == ide) & (clean['passband'] == i)]
                flux = np.array(obj['flux'])
                time = np.array(obj['mjd'])
                fluxerr = np.array(obj['flux_err'])            
                shift = time[np.argmax(flux)]
            
                time = time-shift+original_shift


                # Create a dictionnary of form {parameter_name : initial_guess}
                # In the case of 'shift', the initial guess is the mjd of the max flux

                parameters_dict = {}
                for idx,i in enumerate(param_names[:len(guess)]):
                    parameters_dict[i]=guess[idx]

                # Performs least square regression using iminuit
                least_squares = LeastSquares(time,flux,fluxerr,func)
                fit = Minuit(least_squares, **parameters_dict)
                fit.migrad()

                # Collect all the fit values and store it inside ligne
                parameters = []
                for i in range(len(fit.values)):
                    parameters.append(fit.values[i])

                ligne = np.append(ligne,parameters)


                # Add extra parameters inside ligne

                # Mean square error of the fit
                mse = mean_squared_error(flux, func(time, *parameters))
                ligne = np.append(ligne,mse)

                # Number of points 
                ligne = np.append(ligne,len(time))

                # Maximum flux
                band_max = obj.loc[:,'peak'].iloc[0]
                ligne = np.append(ligne,band_max)
                
        # Else add a line of zero         
        else:
            ligne = [0]*total_parameters
            
        
                
        # Once all passband has been computed for one object, ligne is full, we add it to the final df
        table.iloc[j,1:] = np.array(ligne).flatten()
        # table.to_csv(f'{save}.csv',index=False) 

    stop = timeit.default_timer()

    print('Total time of the parametrisation %.1f sec'%(stop - start)) 

    return table