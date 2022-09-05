"""Feature package for puia."""

__author__ = """Alberto Ardid"""
__email__ = 'alberto.ardid@canterbury.ac.nz'
__version__ = '0.1.0'

# general imports
import os, shutil, warnings, gc
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
import pandas as pd
from multiprocessing import Pool
#from __init__ import ForecastModel

from data import *
from utilities import datetimeify, load_dataframe, save_dataframe, _is_eruption_in

# constants
month = timedelta(days=365.25/12)
day = timedelta(days=1)
'''
Here are two feature clases that operarte a diferent levels. 
FeatureSta oject manages single stations, and FeaturesMulti object manage multiple stations using FeatureSta objects. 
This objects just manipulates feature matrices that already exist. 
'''

class FeaturesSta(object):
    """ Object to manage manipulation of feature matrices derived from seismic data
        (Object works for one station).
        
        Constructor arguments (and attributes):
        ----------------------
        feat_dir: str
            Repository location on feature file
        station : str
            Name of station to download seismic data from (e.g., 'WIZ').
        window  :   float
            Length of data window in days (used in features calculation) (e.g., 2.)
        datastream  :   str
            data stream from where features were calculated (e.g., 'zsc2_dsarF')
        t0 : datetime.datetime
            Beginning of data range.
        t1 : datetime.datetime
            End of data range.
        tes_dir : str
            Repository location on eruptive times file

        Attributes:
        -----------
        file : str
            file from where dataframe is loaded
        df : pandas.DataFrame
            Time series of features.
        feat_list   :   list of string
            List of features
        fM  : pandas dataframe  
            Feature matrix
        ys  : pandas dataframe
            Label vector
        tes: list of datetimes
            Eruptive times

        Methods:
        --------
        load
            load feature matrices and create one dataframe
        save
            save feature matrix
        norm
            mean normalization of feature matrix (feature time series)
        reduce
            reduce number of features by filtering features with non-relevant variance (after normalization)
    """
    def __init__(self, station='WIZ', window = 2., datastream = 'zsc2_dsarF', feat_dir=None, ti=None, tf=None, tes_dir = None):
        self.station=station
        self.window=window
        self.datastream = datastream
        self.n_jobs=4
        self.feat_dir=feat_dir
        #self.file= os.sep.join(self._wd, 'fm_', str(window), '_', self.datastream,  '_', self.station, 
        self.ti=datetimeify(ti)
        self.tf=datetimeify(tf)
        self.fM=None
        self._load_tes(tes_dir) # create self.tes
        self.load() # create dataframe from feature matrices
    def _load_tes(self,tes_dir):
        ''' Load eruptive times, and create atribute self.tes
            Parameters:
            -----------
            tes_dir : str
                Repository location on eruptive times file
            Returns:
            --------
            Note:
            --------
            Attributes created:
            self.tes : list of datetimes
                Eruptive times
        '''
        # get eruptions
        fl_nm = os.sep.join([tes_dir,self.station+'_eruptive_periods.txt'])
        with open(fl_nm,'r') as fp:
            self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    def load(self, drop_nan = True):
        """ Load feature matrix and label vector.
            Parameters:
            -----------
            ti : str, datetime
                Beginning of period to load features.
            tf : str, datetime
                End of period to load features.
            drop_nan    : boolean
                True for droping columns (features) with NaN

            Returns:
            --------

            Note:
            --------
            Attributes created:
            self.fM : pd.DataFrame
                Feature matrix.
            self.ys : pd.DataFrame
                Label vector.
        """
        # boundary dates to be loaded 
        ts = []
        #yrs =  list(range(self.ti.year, self.tf.year+1))
        for yr in list(range(self.ti.year, self.tf.year+2)):
            t = np.max([datetime(yr,1,1,0,0,0),self.ti,self.ti+self.window*day])
            t = np.min([t,self.tf,self.tf])
            ts.append(t)
        if ts[-1] == ts[-2]: ts.pop()
        
        # load features one data stream and year at a time
        fM = []
        ys = []
        for t0,t1 in zip(ts[:-1], ts[1:]):
            #file name (could be improved)
            fl_nm = os.sep.join([self.feat_dir, 'fm_'+str(self.window)+'0w_'+self.datastream+'_'+self.station+'_'+str(t0.year)+'.pkl'])
            # load file
            fMi = load_dataframe(fl_nm, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=None, nrows=None)
            # filter between t0 and t1
            fMi = fMi.loc[t0:t1]
            # append to fMi
            fM.append(fMi)
        # vertical concat on time
        fM = pd.concat(fM)
        # horizontal concat on column
        #FM = pd.concat(FM, axis=1, sort=False)
        # Label vector corresponding to data windows
        ts = fM.index.values
        ys = [_is_eruption_in(days=2., from_time=t, tes = self.tes) for t in pd.to_datetime(ts)]
        ys = pd.DataFrame(ys, columns=['label'], index=fM.index)
        gc.collect()
        self.fM = fM
        self.ys = ys
    def save(self, fl_nm=None):
        ''' Save feature matrix constructed 
            Parameters:
            -----------
            fl_nm   :   str
                file name (include format: .csv, .pkl, .hdf)
            Returns:
            --------
            Note:
            --------
            File is save on feature directory (self.feat_dir)
        '''
        save_dataframe(self.fM, os.sep.join([self.feat_dir,fl_nm]), index=True)
    def norm(self):
        ''' Mean normalization of feature matrix (along columns): substracts mean value, 
            and then divide by standard deviation. 
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
            Method rewrite self.fM (inplace)
        '''
        self.fM =(self.fM-self.fM.mean())/self.fM.std()
    def reduce(self, ft_lt=None):
        ''' Reduce number of columns (features). This is a feature selection.  
            If list of features (ft_lt) are not given, only features with significant 
            variance before eruptive events are kept. 

            Parameters:
            -----------
            ft_lt   :   list of strings
                list of feature to keep. File need to be a comma separated text file 
                where the second column corresponds to the feature name.  

            Returns:
            --------
            Note:
            --------
            If list of feature not given, method assumes that matrix have been normalized (self.norm())
            Method rewrite self.fM (inplace)
        '''
        if ft_lt:
            v = self.fM.columns
            a = self.fM.shape
            with open(ft_lt,'r') as fp:
                colm_keep=[ln.rstrip().split(',')[1].rstrip() for ln in fp.readlines() if (self.datastream in ln and 'cwt' not in ln)]
                # temporal (to fix): if 'cwt' not in ln (features with 'cwt' contains ',' in their names, so its split in the middle)
            self.fM = self.fM[colm_keep] # not working
        else: 
            # drop by some statistical criteria to develop
            pass
            #col_drops = []
            #for i, column in enumerate(self.fM):
            #    std = self.fM.loc[:,column].std()
            #    if not std:
            #        col_drops.append(column)
            #self.fM = self.fM.drop(col_drops, axis=1)
            
class FeaturesMulti(object):
    """ Object to manage multiple feature matrices (lis of FeaturesSta objects)

        Attributes:
        -----------
        stations : list of strings
            list of stations in the feature matrix
        window  :   int
            window length for the features
        datastream  :   str
            data stream from where features were calculated
        df : pandas.DataFrame
            Time series of features for multiple stations
        feat_list   :   list of string
            List of features  
        ref :   vector 
            vector of increasing integers as reference (for each time row in dataframe)
            add as column 'ref' to dataframe (to be used instead of time column)
        labels  :   binary vector
            binary vector indicating if row in dataframe constains eruptions (same size as ref)
         
        Methods:
        --------
        load
            load multiple feature matrices
        save
            save feature matrix
        normalize
            normalize feature time series 
        filter_features
            filter features with non-relevant variance (after normalization)
        match_features
            match features between stations (remove non matching features)
        concatenate
            concatenate multiple feature matrices from diferent stations 
    """
    pass

class PCA(object):
    """ Object that performs PCA analiysis of feature matrices 
        
        Attributes:
        -----------
        stations : list of strings
            list of stations in the feature matrix
        window  :   int
            window length for the features
        datastream  :   str
            data stream from where features were calculated
        df : pandas.DataFrame
            Time series of features for multiple stations
        feat_list   :   list of string
            List of features  
        ref :   vector 
            vector of increasing integers as reference (for each time row in dataframe)
            add as column 'ref' to dataframe (to be used instead of time column)
        labels  :   binary vector
            binary vector indicating if row in dataframe constains eruptions (same size as ref)
         
        Methods:
        --------
        load
            load multiple feature matrices
        normalize
            normalize feature time series 
        save
            save feature matrix
        svd
            compute svd on feature matrix 
        plot_v
            plot eigen values from svd
        plot_scatter_2D_pc
            scatter plot of two principal components 
        cluster
            cluster principal components (e.g, DBCAN)
        plot_cluster
            plot cluster in a 2D scatter plot
            
    """
    pass

# testing
if __name__ == "__main__":
    # FeatureSta
    feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
    tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
    feat_sta = FeaturesSta(station='WIZ', window = 2., datastream = 'zsc2_dsarF', feat_dir=feat_dir, ti='2019-12-07', tf='2019-12-10', tes_dir = tes_dir)
    feat_sta.norm()
    fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
    feat_sta.reduce(ft_lt=fl_lt)
    
