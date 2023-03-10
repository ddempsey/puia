"""Feature package for puia."""

__author__="""Alberto Ardid"""
__email__='alberto.ardid@canterbury.ac.nz'
__version__='0.1.0'

# general imports
import os, shutil, warnings, gc
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
import pandas as pd
from multiprocessing import Pool
from fnmatch import fnmatch

from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from .data import SeismicData
from .utilities import datetimeify, load_dataframe, save_dataframe, _is_eruption_in, makedir
# from .model import MultiVolcanoForecastModel

# constants
month=timedelta(days=365.25/12)
day=timedelta(days=1)
minute=timedelta(minutes=1)
'''
Here are two feature clases that operarte a diferent levels. 
FeatureSta oject manages single stations, and FeaturesMulti object manage multiple stations using FeatureSta objects. 
This objects just manipulates feature matrices that already exist. 

Todo:
- method to mark rows with imcomplete data 
- check sensitivity to random number seed for noise mirror (FeaturesMulti._load_tes())
- criterias for feature selection (see FeaturesSta.reduce())
'''

class FeaturesSta(object):
    """ Class to loas manipulate feature matrices derived from seismic data
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
        dt : datetime.timedelta
            Length between data samples (10 minutes).
        lab_lb  :   float
            Days looking back to assign label '1' from eruption times
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
            remove features from fM. Method recibes a list of feature names (str),
            a direction to a file with names (see method doctstring), or a critiria to 
            selec feature (str, see method docstring).
    """
    def __init__(self, station='WIZ', window=2., datastream='zsc2_dsarF', feat_dir=None, ti=None, tf=None, 
        	tes_dir=None, dt=None, lab_lb=2.):
        self.station=station
        self.window=window
        self.datastream=datastream
        self.n_jobs=4
        self.feat_dir=feat_dir
        #self.file= os.sep.join(self._wd, 'fm_', str(window), '_', self.datastream,  '_', self.station, 
        self.ti=datetimeify(ti)
        self.tf=datetimeify(tf)
        if dt is None:
            self.dt=timedelta(minutes=10)
        else:
            if isinstance(dt,timedelta):
                self.dt=dt   
            else: 
                self.dt=timedelta(minutes=dt)
        self.lab_lb=lab_lb
        self.fM=None
        self.tes_dir=tes_dir
        self.colm_keep=None
        self._load_tes() # create self.tes (and self._tes_mirror)
        self.load() # create dataframe from feature matrices
    def _load_tes(self):
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
        fl_nm=os.sep.join([self.tes_dir,self.station+'_eruptive_periods.txt'])
        with open(fl_nm,'r') as fp:
            self.tes=[datetimeify(ln.rstrip()) for ln in fp.readlines()]
    def load(self, drop_nan=True):
        """ Load feature matrix and label vector.
            Parameters:
            -----------
            ti : str, datetime
                Beginning of period to load features.
            tf : str, datetime
                End of period to load features 
                (this exact time is not considered and period ends 10 minutes before tf)
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
        ts=[]
        #yrs= list(range(self.ti.year, self.tf.year+1))
        for yr in list(range(self.ti.year, self.tf.year+2)):
            t=np.max([datetime(yr,1,1,0,0,0),self.ti,self.ti+self.window*day])
            t=np.max([datetime(yr,1,1,0,0,0),self.ti,self.ti+2*day])
            t=np.min([t,self.tf,self.tf])
            ts.append(t)
        if ts[-1] == ts[-2]: ts.pop()
        
        # load features one data stream and year at a time
        fM=[]
        ys=[]
        for t0,t1 in zip(ts[:-1], ts[1:]):
            #file name (could be improved)
            try:
                fl_nm=os.sep.join([self.feat_dir, 'fm_'+str(self.window)+'0w_'+self.datastream+'_'+self.station+'_'+str(t0.year)+'.pkl'])
                fMi=load_dataframe(fl_nm, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=None, nrows=None)
            except:
                fl_nm=os.sep.join([self.feat_dir, 'fm_'+str(self.window)+'0w_'+self.datastream+'_'+self.station+'_'+str(t0.year)+'.csv'])
                fMi=load_dataframe(fl_nm, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=None, nrows=None)
            # filter between t0 and t1
            fMi=fMi.loc[t0:t1-self.dt] #_dt=10*minute
            # resample at a constant rate (def self.dt)
            # fMi=fMi.resample(str(int(self.dt.seconds/60))+'min').sum()
            # DED replace sum with median
            fMi=fMi.resample(str(int(self.dt.seconds/60))+'min').median()
            # append to fMi
            fM.append(fMi)
        # vertical concat on time
        fM=pd.concat(fM)
        # horizontal concat on column
        #FM=pd.concat(FM, axis=1, sort=False)
        # Label vector corresponding to data windows
        ts=fM.index.values
        ys=[_is_eruption_in(days=self.lab_lb, from_time=t, tes=self.tes) for t in pd.to_datetime(ts)]
        ys=pd.DataFrame(ys, columns=['label'], index=fM.index)
        gc.collect()
        self.fM=fM
        self.ys=ys
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
            ft_lt   :   str, list, boolean
                str: file directory containing list feature names (strs)
                    list of feature to keep. File need to be a comma separated text file 
                    where the second column corresponds to the feature name.  
                list:   list of names (strs) of columns (features) to keep
                True:   select 100 columns (features) with higher variance
            Returns:
            --------
            Note:
            --------
            Method rewrite self.fM (inplace)
            If list of feature not given, method assumes that matrix have been normalized (self.norm())
        '''
        if ft_lt:
            isstr=isinstance(ft_lt, str)
            islst=isinstance(ft_lt, list)
            if isstr:
                with open(ft_lt,'r') as fp:
                    #fp.readlines()
                    colm_keep=[ln.rstrip().split(',')[1].rstrip() for ln in fp.readlines() if (self.datastream in ln and 'cwt' not in ln)]
                    self.colm_keep=colm_keep
                    # temporal (to fix): if 'cwt' not in ln (features with 'cwt' contains ',' in their names, so its split in the middle)
                self.fM=self.fM[self.colm_keep] # not working
            elif islst:
                a=ft_lt[0]
                self.fM=self.fM[ft_lt] # not working
            else: 
                # Filter 100 feature with higher variance
                _l=[]
                _fM=self.fM
                for i in range(100):
                    _col=_fM.var().idxmax()
                    _l.append(_col)
                    _fM=_fM.drop(_col, axis=1)
                del _fM
                self.fM=self.fM[_l]
        else:
            # drop by some statistical criteria to develop
            pass
            #col_drops=[]
            #for i, column in enumerate(self.fM):
            #    std=self.fM.loc[:,column].std()
            #    if not std:
            #        col_drops.append(column)
            #self.fM=self.fM.drop(col_drops, axis=1)
            
class FeaturesMulti(object):
    """ Class to manage multiple feature matrices (list of FeaturesSta objects). 
        Feature matrices (from each station) are imported for the same period of time 
        using as references the eruptive times (see dtb and dtf). 
        This class also performs basic PCA analysis on the multi feature matrix. 

        Attributes:
        -----------
        stations : list of strings
            list of stations in the feature matrix
        window  :   int
            window length for the features calculated
        datastream  :   str
            data stream from where features were calculated
        dt : datetime.timedelta
            Length between data samples (10 minutes).
        lab_lb  :   float
            Days looking back to assign label '1' from eruption times
        dtb : float
            Days looking 'back' from eruptive times to import 
        dtf : float
            Days looking 'forward' from eruptive times to import
        fM : pandas.DataFrame
            Feature matrix of combined time series for multiple stations, for periods around their eruptive times.
            Periods are define by 'dtb' and 'dtf'. Eruptive matrix sections are concatenated. Record of labes and times
            are keept in 'ys'. 
        ys  :   pandas.DataFrame
            Binary labels and times for rows in fM. Index correspond to an increasing integer (reference number)
            Dates for each row in fM are kept in column 'time'
        noise_mirror    :   Boolean
            Generate a mirror feature matrix with exact same dimentions as fM (and ys) but for random non-eruptive times.
            Seed is set on  
        tes : dictionary
            Dictionary of eruption times (multiple stations). 
            Keys are stations name and values are list of eruptive times. 
        tes_mirror : dictionary
            Dictionary of random times (non-overlaping with eruptive times). Same structure as 'tes'.
        feat_list   :   list of string
            List of selected features (see reduce method on FeaturesSta class). Selection of features
            are performs by certain criterias or by given a list of selected features.  
        fM_mirror : pandas.DataFrame
            Equivalent to fM but with non-eruptive times
        ys_mirror  :   pandas.DataFrame
            Equivalent to ys but with non-eruptive times
        savefile_type : str
            Extension denoting file format for save/load. Options are csv, pkl (Python pickle) or hdf.
        feat_dir: str
            Repository location of feature matrices.
        no_erup : list of two elements
            Do not load a certain eruption. Need to specified station and number of eruption  
            (e.g., ['WIZ',4]; eruption number, as 4, start counting from 0)

        U   :   numpy matrix
            Unitary matrix 'U' from SVD of fM (shape nxm). Shape is nxn.
        S   :   numpy vector
            Vector of singular value 'S' from SVD of fM (shape nxm). Shape is nxm.
        VT  :   numpy matrix
            Transponse of unitary matrix 'V' from SVD of fM (shape mxm). Shape is mxm.
        Methods:
        --------
        _load_tes
            load eruptive dates of volcanoes in list self.stations
        _load
            load multiple feature matrices and create one combined matrix. 
            Normalization and feature selection is performed here. 
        norm
            mean normalization of feature matrix (feature time series)
        save
            save feature and label matrix
        svd
            compute svd (singular value decomposition) on feature matrix 
        plot_svd_evals
            plot eigen values from svd
        plot_svd_pcomps
            scatter plot of two principal components 
        cluster (not implemented)
            cluster principal components (e.g, DBCAN)
        plot_cluster (not implemented)
            plot cluster in a 2D scatter plot
    """
    def __init__(self, stations=None, window=2., datastream='zsc2_dsarF', feat_dir=None, 
        dtb=None, dtf=None, tes_dir=None, feat_selc=None,noise_mirror=None, 
        dt=None, lab_lb=2.,savefile_type='pkl', no_erup=None):
        self.stations=stations
        if self.stations:
            self.window=window
            self.datastream=datastream
            self.n_jobs=4
            self.feat_dir=feat_dir
            self.tes_dir=tes_dir
            if dt is None:
                self.dt=timedelta(minutes=10)
            else:
                self.dt=timedelta(minutes=dt)        
            #self.file= os.sep.join(self._wd, 'fm_', str(window), '_', self.datastream,  '_', self.station, 
            self.dtb=dtb*day
            self.dtf=dtf*day
            self.lab_lb=lab_lb
            self.fM=None
            self.ys=None
            self.noise_mirror=None
            self.fM_mirror=None
            self.ys_mirror=None
            self.feat_selc=feat_selc
            self.tes_dir=tes_dir
            self.noise_mirror=noise_mirror
            self.savefile_type=savefile_type
            self.no_erup=no_erup
            self._load_tes() # create self.tes (and self.tes_mirror) 
            self._load() # create dataframe from feature matrices
    def _load_tes(self):
        ''' Load eruptive times for list of volcanos (self.stations). 
            A dictionary is created with station names as key and list of eruptive times as values. 
            Parameters:
            -----------
            tes_dir : str
                Repository location on eruptive times file
            Returns:
            --------
            Note:
            --------
            Attributes created:
            self.tes : diccionary of eruptive times per stations. 
            self.tes_mirror : diccionary of non-eruptive times (noise mirror) per stations. 
        '''
        #
        self.tes={}
        for sta in self.stations:
            # get eruptions
            fl_nm=os.sep.join([self.tes_dir,sta+'_eruptive_periods.txt'])
            with open(fl_nm,'r') as fp:
                if self.no_erup:
                    self.tes[sta]=[datetimeify(ln.rstrip()) for i,ln in enumerate(fp.readlines()) if (i != self.no_erup[1] and sta is self.no_erup[0])]
                else:
                    self.tes[sta]=[datetimeify(ln.rstrip()) for i,ln in enumerate(fp.readlines())]
        # create noise mirror to fM
        if self.noise_mirror:
            self.tes_mirror={}
            for sta in self.stations:
                # get initial and final date of record 
                if True:# get period from data
                    _td=SeismicData(station=sta, data_dir=self.tes_dir)
                    _td.ti=_td.df.index[0]
                    _td.tf=_td.df.index[-1]
                    if sta is 'FWVZ':
                        _td.ti=datetimeify('2005-03-01 00:00:00')
                if False:# get period from feature matrices available by year
                    pass
                # select random dates (don't overlap with eruptive periods)
                _tes_mirror=[]
                for i in range(len(self.tes[sta])): # number of dates to create
                    _d=True
                    while _d:
                        _r=random_date(_td.ti, _td.tf)
                        # check if overlap wit eruptive periods
                        _=True # non overlap
                        for te in self.tes[sta]:
                            if _r in [te-1.5*month,te+1.5*month]:
                                _=False
                        if _:
                            _d=False
                        _tes_mirror.append(_r)
                self.tes_mirror[sta]=_tes_mirror
    def _load(self):
        """ Load and combined feature matrices and label vectors from multiple stations. 
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
            Matrices per stations are reduce to selected features (if given in self.feat_selc) 
            and normalize (before concatenation). Columns (features) with NaNs are remove too.
            Attributes created:
            self.fM : pd.DataFrame
                Combined feature matrix of multiple stations and eruptions
            self.ys : pd.DataFrame
                Label vector of multiple stations and eruptions
            self.fM_mirror : pd.DataFrame
                Combined feature matrix of multiple stations and eruptions
            self.ys_mirror : pd.DataFrame
                Label vector of multiple stations and eruptions
        """
        #
        FM=[]
        ys=[]
        _blk=0
        for sta in self.stations:
            fM=[]
            for i, te in enumerate(self.tes[sta]): 
                # FeatureSta
                feat_sta=FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
                    ti=te-self.dtb, tf=te+self.dtf, tes_dir=self.tes_dir, dt=self.dt, lab_lb=self.lab_lb)
                #if self.feat_selc and (isinstance(self.feat_selc, str) or isinstance(self.feat_selc, list)):
                #    feat_sta.reduce(ft_lt=self.feat_selc)
                    #feat_sta.norm()
                #elif self.feat_selc:
                    #feat_sta.norm()
                feat_sta.reduce(ft_lt=self.feat_selc)
                fM.append(feat_sta.fM)
                # add column to labels with stations name
                feat_sta.ys['station']=sta
                feat_sta.ys['noise_mirror']=False
                feat_sta.ys['block']=_blk
                _blk+=1
                ys.append(feat_sta.ys)
                #
                if self.noise_mirror:
                    te= self.tes_mirror[sta][i]
                    # FeatureSta
                    _feat_sta=FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
                        ti=te-self.dtb, tf=te+self.dtf, tes_dir=self.tes_dir)
                    #feat_sta.norm()
                    # filter to features in fM (columns)
                    _feat_sta.fM=_feat_sta.fM[list(feat_sta.fM.columns)]
                    #
                    fM.append(_feat_sta.fM)
                    # add column to labels with stations name
                    _feat_sta.ys['station']=sta
                    _feat_sta.ys['noise_mirror']=True
                    _feat_sta.ys['block']=_blk
                    _blk+=1
                    ys.append(_feat_sta.ys)
                    #
                    del _feat_sta  
                #
                del feat_sta             
            #
            _fM=pd.concat(fM)
            # norm
            _fM =(_fM-_fM.mean())/_fM.std() # norm station matrix
            FM.append(_fM) # add to multi station
            del _fM
        # concatenate and modifify index to a reference ni
        self.fM=pd.concat(FM)
        # drop columns with NaN
        self.fM=self.fM.drop(columns=self.fM.columns[self.fM.isna().any()].tolist())
        # create index with a reference number and create column 'time' 
        #self.fM['time']=self.fM.index # if want to keep time column
        self.fM.index=range(self.fM.shape[0])
        self.ys=pd.concat(ys)
        self.ys['time']=self.ys.index
        self.ys.index=range(self.ys.shape[0])
        # modifiy index 
        del fM, ys
        #
        # if self.noise_mirror:
        #     fM=[]
        #     ys=[]
        #     for sta in self.stations:
        #         for te in self.tes_mirror[sta]: 
        #             # FeatureSta
        #             feat_sta=FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
        #                 ti=te-self.dtb, tf=te+self.dtf, tes_dir=self.tes_dir)
        #             feat_sta.norm()
        #             # filter to features in fM (columns)
        #             feat_sta.fM=feat_sta.fM[list(self.fM.columns)]
        #             #
        #             fM.append(feat_sta.fM)
        #             # add column to labels with stations name
        #             feat_sta.ys['station']=sta
        #             ys.append(feat_sta.ys)
        #             #
        #             del feat_sta
        #     # concatenate and modifify index to a reference ni
        #     self.fM_mirror=pd.concat(fM)
        #     # drop columns with NaN
        #     #self.fM_mirror=self.fM_mirror.drop(columns=self.fM_mirror.columns[self.fM_mirror.isna().any()].tolist())
        #     # create index with a reference number and create column 'time' 
        #     #self.fM['time']=self.fM.index # if want to keep time column
        #     self.fM_mirror.index=range(self.fM_mirror.shape[0])
        #     self.ys_mirror=pd.concat(ys)
        #     self.ys_mirror['time']=self.ys_mirror.index
        #     self.ys_mirror.index=range(self.ys_mirror.shape[0])
        #     # modifiy index 
        #     del fM, ys
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
    def save(self,fl_nm=None):
        ''' Save feature matrix and label matrix
            Parameters:
            -----------
            fl_nm   :   str
                file name (include format: .csv, .pkl, .hdf)
            Returns:
            --------
            Note:
            --------
            File is save on feature directory (self.feat_dir)
            Default file name: 
                'FM_'+window+'w_'+datastream+'_'+stations(-)+'_'+dtb+'_'+dtf+'dtf'+'.'+file_type
                e.g., FM_2w_zsc2_hfF_WIZ-KRVZ_60dtb_0dtf.csv
        '''
        if not fl_nm:
            fl_nm='FM_'+str(int(self.window))+'w_'+self.datastream+'_'+'-'.join(self.stations)+'_'+str(self.dtb.days)+'dtb_'+str(self.dtf.days)+'dtf.'+self.savefile_type
        save_dataframe(self.fM, os.sep.join([self.feat_dir,fl_nm]), index=True)
        # save labels
        _=fl_nm.find('.')
        _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        save_dataframe(self.ys, os.sep.join([self.feat_dir,_fl_nm]), index=True)
        # if self.noise_mirror:
        #     fl_nm=fl_nm[:_]+'_nmirror'+fl_nm[_:]
        #     save_dataframe(self.fM_mirror, os.sep.join([self.feat_dir,fl_nm]), index=True)
        #     _=fl_nm.find('.')
        #     _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        #     save_dataframe(self.ys_mirror, os.sep.join([self.feat_dir,_fl_nm]), index=True)
    def load_fM(self, feat_dir,fl_nm,noise_mirror=None):
        ''' Load feature matrix and lables from file (fl_nm)
            Parameters:
            -----------
            feat_dir    :   str
                feature matrix directory
            fl_nm   :   str
                file name (include format: .csv, .pkl, .hdf)
            Returns:
            --------
            Note:
            --------
            Method load feature matrix atributes from file name:
                'FM_'+window+'w_'+datastream+'_'+stations(-)+'_'+dtb+'_'+dtf+'dtf'+'.'+file_type
                e.g., FM_2w_zsc2_hfF_WIZ-KRVZ_60dtb_0dtf.csv
        '''
        # if noise_mirror:
        #     self.noise_mirror=True
        # else:
        #     self.noise_mirror=False
        # assing attributes from file name
        def _load_atrib_from_file(fl_nm): 
            _=fl_nm.split('.')[0]
            _=_.split('_')[1:]
            self.stations=_[-3].split('-')
            self.window=int(_[0][0])
            self.dtf=int(_[-1][0])
            self.dtb=int(_[-2][0])
            self.data_stream=('_').join(_[1:-3])
            #
            self.feat_dir=feat_dir
        _load_atrib_from_file(fl_nm)
        # load feature matrix
        self.fM=load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        # load labels 
        _=fl_nm.find('.')
        _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        self.ys=load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        self.ys['time']=pd.to_datetime(self.ys['time'])
        #
        # if self.noise_mirror:
        #     fl_nm=fl_nm[:_]+'_nmirror'+fl_nm[_:]
        #     # load feature matrix
        #     self.fM_mirror=load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, 
        #         infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        #     load labels 
        #     _=fl_nm.find('.')
        #     _fl_nm=fl_nm[:_]+'_labels'+_fl_nm[_-1:]
        #     self.ys_mirror=load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, 
        #         infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        #     self.ys_mirror['time']=pd.to_datetime(self.ys['time'])
        #
    def svd(self, norm=None, noise_mirror=False):
        ''' Compute SVD (singular value decomposition) on feature matrix. 
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
            Attributes created:
            U   :   numpy matrix
                Unitary matrix 'U' from SVD of fM (shape nxm). Shape is nxn.
            S   :   numpy vector
                Vector of singular value 'S' from SVD of fM (shape nxm). Shape is nxm.
            VT  :   numpy matrix
                Transponse of unitary matrix 'V' from SVD of fM (shape mxm). Shape is mxm.
        '''
        if norm:
            self.norm()
        #_fM=self.fM.drop('time',axis=1)
        if noise_mirror:
            self.U,self.S,self.VT=np.linalg.svd(self.fM,full_matrices=True)
        else:
            # filter noise rows from fM
            _fM=self.fM[self.ys["noise_mirror"] == False] 
            self.U,self.S,self.VT=np.linalg.svd(_fM,full_matrices=True)
        del _fM
    def plot_svd_evals(self):
        ''' Plot eigen values from svd
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
        '''
        plt.rcParams['figure.figsize']=[8, 8]
        fig1=plt.figure()
        #
        ax1=fig1.add_subplot(221)
        #ax1.semilogy(S,'-o',color='k')
        ax1.semilogy(self.S[:int(len(self.S))],'-o',color='k')
        ax2=fig1.add_subplot(222)
        #ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')
        ax2.plot(np.cumsum(self.S[:int(len(self.S))])/np.sum(self.S[:int(len(self.S))]),'-o',color='k')
        #
        ax3=fig1.add_subplot(223)
        #ax1.semilogy(S,'-o',color='k')
        ax3.semilogy(self.S[:int(len(self.S)/10)],'-o',color='k')
        ax4=fig1.add_subplot(224)
        #ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')
        ax4.plot(np.cumsum(self.S[:int(len(self.S)/10)])/np.sum(self.S[:int(len(self.S)/10)]),'-o',color='k')
        #
        [ax.set_title('eigen values') for ax in [ax1,ax3]]
        [ax.set_title('cumulative eigen values') for ax in [ax2,ax4]]
        [ax.set_xlabel('# eigen value') for ax in [ax1,ax2,ax3,ax4]]
        plt.tight_layout()
        plt.show() 
    def plot_svd_pcomps(self,labels=None,quick=None):
        ''' Plot fM (feature matrix) into principal component (first nine).
            Rows of fM are projected into rows of VT.  
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
        '''
        #
        plt.rcParams['figure.figsize']=[10, 3.3]
        #fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9))=plt.subplots(3, 3)
        fig, (ax1, ax2, ax3)=plt.subplots(1, 3)
        fig.suptitle('fM projected in VT')
        #
        if labels:
            import random 
            random.seed(9001)
            colors={}
            n=len(self.stations)
            for i,sta in enumerate(self.stations):
                colors[sta]='#%06X' % random.randint(0, 0xFFFFFF)
        #
        #quick=True
        _N=self.fM.shape[0]
        N=range(_N)
        if quick:
            N=N[N[0]:N[-1]:2]#int(len(N)/1000)]
        #
        for i, ax in enumerate(fig.get_axes()): #[ax1,ax2,ax3])
            for j in N:
                #x=VT[3,:] @ covariance_matrix[j,:].T 
                y=self.VT[i,:] @ self.fM.values[j,:].T
                z=self.VT[i+1,:] @ self.fM.values[j,:].T
                # check if point is background
                _nm=self.ys["noise_mirror"][j]
                if labels:
                    if _nm:
                        ax.plot(y,z, marker='.',color='k') 
                    else:
                        ax.plot(y,z, marker='.',color=colors[self.ys['station'][j]])                
                else:
                    ax.plot(y,z,'b.')
            #ax1.set_xlabel('pc3')
            ax.set_xlabel('pc'+str(i+1))
            ax.set_ylabel('pc'+str(i+2))
            #ax1.view_init(0,0)
            if i==0:
                for sta in self.stations:
                    try:
                        ax.plot([],[], marker='.',color=colors[sta], label=sta_code[sta])
                    except:
                        ax.plot([],[], marker='.',color=colors[sta], label=sta)
        if labels:
            ax.plot([],[], marker='.',color='k', label='noise')
        fig.legend()   
        plt.tight_layout()
        #plt.savefig('foo.png')
        plt.show()
    def plot_svd_pcomps_noise_mirror(self):
        ''' Plot fM (feature matrix) into principal component (first nine).
            Rows of fM are projected into rows of VT.  
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
        '''
        #
        plt.rcParams['figure.figsize']=[8, 8]
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9))=plt.subplots(3, 3)
        fig.suptitle('fM projected in VT')
        #
        for i, ax in enumerate(fig.get_axes()):#[ax1,ax2,ax3]):#
            for j in range(self.fM.shape[0]):
                #x=VT[3,:] @ covariance_matrix[j,:].T
                y=self.VT[i,:] @ self.fM.values[j,:].T
                z=self.VT[i+1,:] @ self.fM.values[j,:].T
                ax.plot(y,z,'r.',zorder=0)
                #
                if j<self.fM_mirror.shape[0]-1:
                    a=self.VT[i,:]
                    b=self.fM_mirror.values[j,:].T
                    y_m=self.VT[i,:] @ self.fM_mirror.values[j,:].T
                    z_m=self.VT[i+1,:] @ self.fM_mirror.values[j,:].T
                    #
                    ax.plot(y_m,z_m,'b.',zorder=1)
                    points_m=j+1
                #ax.plot(y_m,z_m,'b.')
            #ax1.set_xlabel('pc3')
            ax.set_xlabel('pc'+str(i+1))
            ax.set_ylabel('pc'+str(i+2))
            #ax1.view_init(0,0)
            if i==0:
                for sta in self.stations:
                    ax.plot([],[], marker='.',color='r', label='eruptive')
                    ax.plot([],[], marker='.',color='w', label='('+str(j)+' points)')
                    ax.plot([],[], marker='.',color='b', label='non-eruptive')
                    ax.plot([],[], marker='.',color='w', label='('+str(points_m)+' points)')
        fig.legend()   
        plt.tight_layout()     
        plt.show()

class Feature(object):
    def __init__(self, parent, window, overlap, look_forward, feature_dir):
        self.parent=parent
        self.window=window
        self.overlap=overlap
        self.look_forward=look_forward

        self.compute_only_features=[]
        
        # self.feature_root=feature_root
        self.feat_dir=feature_dir if feature_dir else f'{self.parent.root_dir}/features'
        self.featfile=lambda ds,yr,st: (f'{self.feat_dir}/fm_{self.window:3.2f}w_{ds}_{st}_{yr:d}.{self.parent.savefile_type}')
  
        # time stepping variables
        self.dtw=timedelta(days=self.window)
        self.dtf=timedelta(days=self.look_forward)
        self.dt=timedelta(seconds=600)
        self.dto=(1.-self.overlap)*self.dtw
        self.iw=int(self.window*6*24)         
        self.io=int(self.overlap*self.iw)      
        if self.io == self.iw: 
            self.io -= 1
        self.window=self.iw*1./(6*24)
        self.dtw=timedelta(days=self.window)
        self.overlap=self.io*1./self.iw
        self.dto=(1.-self.overlap)*self.dtw
    def _exclude_dates(self, X, y, exclude_dates):
        """ Drop rows from feature matrix and label vector.
            Parameters:
            -----------
            X : pd.DataFrame
                Matrix to drop columns.
            y : pd.DataFrame
                Label vector.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates=[['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
            yr : pd.DataFrame
                Reduced label vector.
        """
        if not exclude_dates:
            return X,y
        for exclude_date_range in exclude_dates:
            t0,t1=[datetimeify(dt) for dt in exclude_date_range]
            inds=(y.index<t0)|(y.index>=t1)
            X=X.loc[inds]
            y=y.loc[inds]
        return X,y
    def load_data(self, ti=None, tf=None, exclude_dates=[]):
        if self.parent.__class__.__init__.__qualname__.split('.')[0] == 'MultiVolcanoForecastModel':
            fMs=[]; yss=[]
            # load data from different stations
            for station,data in self.parent.data.items():
                ti,tf=self.parent._train_dates[station]
                # get matrices and remove date ranges
                self.data=data
                fM, ys=self._load_data(ti, tf)
                fM, ys=self._exclude_dates(fM, ys, exclude_dates[station])
                fMs.append(fM)
                yss.append(ys)
            fM=pd.concat(fMs, axis=0)
            ys=pd.concat(yss, axis=0)
        else:
            # default training intervals
            self.data=self.parent.data
            ti=self.data.ti+self.dtw if ti is None else datetimeify(ti)
            tf=self.data.tf if tf is None else datetimeify(tf)
            # get matrices and remove date ranges
            fM, ys=self._load_data(ti, tf)
            fM, ys=self._exclude_dates(fM, ys, exclude_dates)
        return fM, ys
    def _load_data(self, ti, tf):
        """ Load feature matrix and label vector.
            Parameters:
            -----------
            ti : str, datetime
                Beginning of period to load features.
            tf : str, datetime
                End of period to load features.
            yr : int
                Year to load data for. If None and hires, recursion will activate.
            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.DataFrame
                Label vector.
        """
        # # return pre loaded
        # try:
        #     if ti == self.ti_prev and tf == self.tf_prev:
        #         return self.fM, self.ys
        # except AttributeError:
        #     pass

        # range checking
        if tf > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(tf, data.tf))
        if ti < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(ti, data.ti))
        
        # subdivide load into years
        ts=[]
        for yr in list(range(ti.year, tf.year+2)):
            t=np.max([datetime(yr,1,1,0,0,0),ti,self.data.ti+self.dtw])
            t=np.min([t,tf,self.data.tf])
            ts.append(t)
        if ts[-1] == ts[-2]: ts.pop()
        
        # load features one data stream and year at a time
        FM=[]
        ys=[]
        for ds in self.data.data_streams:
            fM=[]
            ys=[]
            for t0,t1 in zip(ts[:-1], ts[1:]):
                fMi,y=self._extract_features(t0,t1,ds)
                fM.append(fMi)
                ys.append(y)
            # vertical concat on time
            FM.append(pd.concat(fM))
        # horizontal concat on column
        FM=pd.concat(FM, axis=1, sort=False)
        ys=pd.concat(ys)
        
        # self.ti_prev=ti
        # self.tf_prev=tf
        # self.fM=FM
        # self.ys=ys
        return FM, ys
    def _construct_windows(self, Nw, ti, ds, i0=0, i1=None, indx = None):
        """
        Create overlapping data windows for feature extraction.

        Parameters:
        -----------
        Nw : int
            Number of windows to create.
        ti : datetime.datetime
            End of first window.
        i0 : int
            Skip i0 initial windows.
        i1 : int
            Skip i1 final windows.
        indx : list of datetime.datetime
            Computes only windows for requested index list

        Returns:
        --------
        df : pandas.DataFrame
            Dataframe of windowed data, with 'id' column denoting individual windows.
        window_dates : list
            Datetime objects corresponding to the beginning of each data window.
        """
        if i1 is None:
            i1 = Nw
        if not indx:
            # get data for windowing period
            df = self.data.get_data(ti-self.dtw, ti+(Nw-1)*self.dto)[[ds,]]
            # create windows
            dfs = []
            for i in range(i0, i1):
                dfi = df[:].iloc[i*(self.iw-self.io):i*(self.iw-self.io)+self.iw]
                try:
                    dfi['id'] = pd.Series(np.ones(self.iw, dtype=int)*i, index=dfi.index)
                except ValueError:
                    print('this shouldn\'t be happening')
                dfs.append(dfi)
            df = pd.concat(dfs)
            window_dates = [ti + i*self.dto for i in range(Nw)]
            return df, window_dates[i0:i1]
        else:
            # get data for windowing define in indx
            dfs = []
            for i, ind in enumerate(indx): # loop over indx
                ind = np.datetime64(ind).astype(datetime)
                dfi = self.data.get_data(ind-self.dtw, ind)[[ds,]].iloc[:]
                try:
                    dfi['id'] = pd.Series(np.ones(self.iw, dtype=int)*i, index=dfi.index)
                except ValueError:
                    print('this shouldn\'t be happening')
                dfs.append(dfi)
            df = pd.concat(dfs)
            window_dates = indx
            return df, window_dates
    def _extract_features(self, ti, tf, ds):
        """
            Extract features from windowed data.

            Parameters:
            -----------
            ti : datetime.datetime
                End of first window.
            tf : datetime.datetime
                End of last window.

            Returns:
            --------
            fm : pandas.Dataframe
                tsfresh feature matrix extracted from data windows.
            ys : pandas.Dataframe
                Label vector corresponding to data windows

            Notes:
            ------
            Saves feature matrix to $root_dir/features/$root_features.csv to avoid recalculation.
        """
        makedir(self.feat_dir)
        # number of windows in feature request
        Nw = int(np.floor(((tf-ti)/self.dt-1)/(self.iw-self.io)))+1
        Nmax = 6*24*31 # max number of construct windows per iteration (6*24*30 windows: ~ a month of hires, overlap of 1.)

        # file naming convention
        yr = ti.year
        ftfl = self.featfile(ds,yr,self.data.station)

        # condition on the existence of fm save for the year requested
        if os.path.isfile(ftfl): # check if feature matrix file exists
            # load existing feature matrix
            fm_pre = load_dataframe(ftfl, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=None, nrows=None)
            # request for features, labeled by index
            l1 = [np.datetime64(ti + i*self.dto) for i in range(Nw)]
            # read the existing feature matrix file (index column only) for current computed features
            # testing
            l2 = fm_pre.index
            # identify new features for calculation
            l3 = list(set(l1)-set(l2))
            # alternative to last to commands
            l2 = load_dataframe(ftfl, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values
            l3 = []
            [l3.append(l1i.astype(datetime)) for l1i in l1 if l1i not in l2]
            # end testing
            # check is new features need to be calculated (windows)
            if l3 == []: # all features requested already calculated
                # load requested features (by index) and generate fm
                fm = fm_pre[fm_pre.index.isin(l1, level=0)]
                del fm_pre, l1, l2, l3

            else: # calculate new features and add to existing saved feature matrix
                # note: if len(l3) too large for max number of construct windows (say Nmax) l3 is chunked
                # into subsets smaller of Nmax and call construct_windows/extract_features on these subsets
                if len(l3) >= Nmax: # condition on length of requested windows
                    # divide l3 in subsets
                    n_sbs = int(Nw/Nmax)+1
                    def chunks(lst, n):
                        'Yield successive n-sized chunks from lst'
                        for i in range(0, len(lst), n):
                            yield lst[i:i + n]
                    l3_sbs =  chunks(l3,int(Nw/n_sbs))
                    # copy existing feature matrix (to be filled and save)
                    fm = pd.concat([fm_pre])
                    # loop over subsets
                    for l3_sb in l3_sbs:
                        # generate dataframe for subset
                        fm_new = self._const_wd_extr_ft(Nw, ti, ds, indx = l3_sb)
                        # concatenate subset with existing feature matrix
                        fm = pd.concat([fm, fm_new])
                        del fm_new
                        # sort new updated feature matrix and save (replace existing one)
                        fm.sort_index(inplace=True)
                        save_dataframe(fm, ftfl, index=True, index_label='time')
                else:
                    # generate dataframe
                    fm = self._const_wd_extr_ft(Nw, ti, ds, indx = l3)
                    fm = pd.concat([fm_pre, fm])
                    # sort new updated feature matrix and save (replace existing one)
                    fm.sort_index(inplace=True)
                    save_dataframe(fm, ftfl, index=True, index_label='time')
                # keep in feature matrix (in memory) only the requested windows
                fm = fm[fm.index.isin(l1, level=0)]
                #
                del fm_pre, l1, l2, l3

        else:
            # note: if Nw is too large for max number of construct windows (say Nmax) the request is chunk
            # into subsets smaller of Nmax and call construct_windows/extract_features on these subsets
            if Nw >= Nmax: # condition on length of requested windows
                # divide request in subsets
                n_sbs = int(Nw/Nmax)+1
                def split_num(num, div):
                    'List of number of elements subsets of num divided by div'
                    return [num // div + (1 if x < num % div else 0)  for x in range (div)]

                Nw_ls = split_num(Nw, n_sbs)
                ## fm for first subset
                # generate dataframe
                fm = self._const_wd_extr_ft(Nw_ls[0], ti, ds)
                save_dataframe(fm, ftfl, index=True, index_label='time')
                # aux intial time (vary for each subset)
                ti_aux = ti+(Nw_ls[0])*self.dto
                # loop over the rest subsets
                for Nw_l in Nw_ls[1:]:
                    # generate dataframe
                    fm_new = self._const_wd_extr_ft(Nw_l, ti_aux, ds)
                    # concatenate
                    fm = pd.concat([fm, fm_new])
                    # increase aux ti
                    ti_aux = ti_aux+(Nw_l)*self.dto
                    save_dataframe(fm, ftfl, index=True, index_label='time')
                # end working section
                del fm_new
            else:
                # generate dataframe
                fm = self._const_wd_extr_ft(Nw, ti, ds)
                save_dataframe(fm, ftfl, index=True, index_label='time')

        # Label vector corresponding to data windows
        ys = pd.DataFrame(self._get_label(fm.index.values), columns=['label'], index=fm.index)
        gc.collect()
        return fm, ys
    def _extract_featuresX(self, df, **kw):
        t0 = df.index[0]+self.dtw
        t1 = df.index[-1]+self.dt
        print('{:s} feature extraction {:s} to {:s}'.format(df.columns[0], t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')))
        return extract_features(df, **kw)
    def _const_wd_extr_ft(self, Nw, ti, ds, indx = None):
        'Construct windows, extract features and return dataframe'
        # features to compute
        cfp = ComprehensiveFCParameters()
        if self.compute_only_features:
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in self.compute_only_features])
        # else:
        #     # drop features if relevant
        #     _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]
        nj = self.parent.n_jobs
        if nj == 1:
            nj = 0
        kw = {'column_id':'id', 'n_jobs':nj,
            'default_fc_parameters':cfp, 'impute_function':impute}
        # construct_windows/extract_features for subsets
        df, wd = self._construct_windows(Nw, ti, ds, indx = indx)
        # extract features and generate feature matrixs
        fm = self._extract_featuresX(df, **kw)
        fm.index = pd.Series(wd)
        fm.index.name = 'time'
        return fm 
    def _get_label(self, ts):
        """ Compute label vector.
            Parameters:
            -----------
            t : datetime like
                List of dates to inspect look-forward for eruption.
            Returns:
            --------
            ys : list
                Label vector.
        """
        return [self.data._is_eruption_in(days=self.look_forward, from_time=t) for t in pd.to_datetime(ts)]
    
def _drop_features(X, drop_features):
    """ Drop columns from feature matrix.
        Parameters:
        -----------
        X : pd.DataFrame
            Matrix to drop columns.
        drop_features : list
            tsfresh feature names or calculators to drop from matrix.
        Returns:
        --------
        Xr : pd.DataFrame
            Reduced matrix.
    """
    if len(drop_features)==0:
        return X
    cfp=ComprehensiveFCParameters()
    df2=[]
    for df in drop_features:
        if df in X.columns:
            df2.append(df)          # exact match
        else:
            if df in cfp.keys() or df in ['fft_coefficient_hann']:
                df='*__{:s}__*'.format(df)    # feature calculator
            # wildcard match
            df2 += [col for col in X.columns if fnmatch(col, df)]              
    return X.drop(columns=df2)

# testing
if __name__ == "__main__":
    # station code dic
    sta_code={'WIZ': 'Whakaari',
                'FWVZ': 'Ruapehu',
                'KRVZ': 'Tongariro',
                'BELO': 'Bezymiany',
                'PVV': 'Pavlof',
                'VNSS': 'Veniaminof',
                'IVGP': 'Vulcano',
                'AUS': 'Agustine',
                'TBTN': 'Telica',
                'OGDI': 'Reunion',
                'TBTN': 'Telica',
                'MEA01': 'Merapi',
                'GOD' : 'Eyjafjallajökull',
                'ONTA' : 'Ontake',
                'REF' : 'Redoubt',
                'POS' : 'Kawa Ijen',
                'DAM' : 'Kawa Ijen',
                'VONK' : 'Holuhraun',
                'BOR' : 'Piton de la Fournaise',
                'VRLE' : 'Rincon de la Vega',
                'T01' : 'Tungurahua',
                'COP' : 'Copahue'
                }
    # dictionary of eruption names 
    erup_dict={'WIZ_1': 'Whakaari 2012',
                'WIZ_2': 'Whakaari 2013a',
                'WIZ_3': 'Whakaari 2013b',
                'WIZ_4': 'Whakaari 2016',
                'WIZ_5': 'Whakaari 2019',
                'FWVZ_1': 'Ruapehu 2006',
                'FWVZ_2': 'Ruapehu 2007',
                'FWVZ_3': 'Ruapehu 2009',
                'KRVZ_1': 'Tongariro 2012a',
                'KRVZ_2': 'Tongariro 2012b',
                'BELO_1': 'Bezymianny 2007a',
                'BELO_2': 'Bezymianny 2007b',
                'BELO_3': 'Bezymianny 2007c',
                'PVV_1': 'Pavlof 2014a',
                'PVV_2': 'Pavlof 2014b',
                'PVV_3': 'Pavlof 2016',
                'VNSS_1': 'Veniaminof 2013',
                'VNSS_2': 'Veniaminof 2018',
                'TBTN_1': 'Telica 2011',
                'TBTN_2': 'Telica 2013',
                'MEA01_1': 'Merapi 2014a',
                'MEA01_2': 'Merapi 2014b',
                'MEA01_3': 'Merapi 2014c',
                'MEA01_4': 'Merapi 2018a',
                'MEA01_5': 'Merapi 2018b',
                'MEA01_6': 'Merapi 2018c',
                'MEA01_7': 'Merapi 2018d',
                'MEA01_8': 'Merapi 2019a',
                'GOD_1' : 'Eyjafjallajökull 2010a',
                'GOD_2' : 'Eyjafjallajökull 2010b',
                'VONK_1' : 'Holuhraun 2014a'
                }

    if False: # FeatureSta class
        # FeatureSta
        feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
        tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
        feat_sta=FeaturesSta(station='WIZ', window=2., datastream='zsc2_dsarF', feat_dir=feat_dir, 
            ti='2019-12-07', tf='2019-12-10', tes_dir=tes_dir)
        feat_sta.norm()
        fl_lt=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        feat_sta.reduce(ft_lt=fl_lt)
    if True: # FeatureMulti class
        # 
        if False: # load from server
            feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
            datadir=r'U:\Research\EruptionForecasting\eruptions\data'
        if False: # load from local PC
            feat_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\features'
            datadir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
        if True: # load from high speed ext hd
            feat_dir=r'E:\EruptionForecasting\features'
            datadir=r'E:\EruptionForecasting\data'
        # feature selection
        fl_lt=feat_dir+r'\all.fts'
        #
        if False: # create combined feature matrix
            stations=['WIZ','FWVZ']#,'KRVZ']#,'VNSS','BELO','GOD','TBTN','MEA01']
            win=2.
            dtb=15
            dtf=0
            datastream='zsc2_rsamF'
            #ft=['zsc2_dsarF__median']
            feat_stas=FeaturesMulti(stations=stations, window=win, datastream=datastream, feat_dir=feat_dir, 
                dtb=dtb, dtf=dtf, lab_lb=7,tes_dir=datadir, noise_mirror=True, data_dir=datadir, 
                    dt=10,savefile_type='csv',feat_selc=fl_lt)#fl_lt
            #fl_nm='FM_'+str(int(win))+'w_'+datastream+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'+'.csv'
            #feat_stas.norm()
            feat_stas.save()#fl_nm=fl_nm)
            #
        if True: # load existing combined feature matrix 
            feat_stas=FeaturesMulti()
            #fl_nm='FM_'+str(win)+'w_'+datastream+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'+'.csv'
            fl_nm='FM_2w_zsc2_rsamF_WIZ-FWVZ_15dtb_0dtf.csv'
            #fl_nm='FM_2w_zsc2_dsarF_WIZ-FWVZ-KRVZ-PVV-VNSS-BELO-GOD-TBTN-MEA01_5dtb_2dtf.csv'
            feat_stas.load_fM(feat_dir=feat_dir,fl_nm=fl_nm)#,noise_mirror=True)
            #
            feat_stas.svd()
            #feat_stas.plot_svd_evals()
            feat_stas.plot_svd_pcomps(labels=True,quick=True)
            #feat_stas.plot_svd_pcomps_noise_mirror()
            #

