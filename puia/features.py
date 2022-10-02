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

from data import TremorData
from utilities import datetimeify, load_dataframe, save_dataframe, _is_eruption_in, random_date

# constants
month = timedelta(days=365.25/12)
day = timedelta(days=1)
minute = timedelta(minutes=1)
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
    def __init__(self, station='WIZ', window = 2., datastream = 'zsc2_dsarF', feat_dir=None, ti=None, tf=None, 
        	tes_dir=None, dt=None, lab_lb=2.):
        self.station=station
        self.window=window
        self.datastream = datastream
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
        fl_nm = os.sep.join([self.tes_dir,self.station+'_eruptive_periods.txt'])
        with open(fl_nm,'r') as fp:
            self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    def load(self, drop_nan = True):
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
        ts = []
        #yrs =  list(range(self.ti.year, self.tf.year+1))
        for yr in list(range(self.ti.year, self.tf.year+2)):
            t = np.max([datetime(yr,1,1,0,0,0),self.ti,self.ti+self.window*day])
            t = np.max([datetime(yr,1,1,0,0,0),self.ti,self.ti+2*day])
            t = np.min([t,self.tf,self.tf])
            ts.append(t)
        if ts[-1] == ts[-2]: ts.pop()
        
        # load features one data stream and year at a time
        fM = []
        ys = []
        for t0,t1 in zip(ts[:-1], ts[1:]):
            #file name (could be improved)
            try:
                fl_nm = os.sep.join([self.feat_dir, 'fm_'+str(self.window)+'0w_'+self.datastream+'_'+self.station+'_'+str(t0.year)+'.pkl'])
                fMi = load_dataframe(fl_nm, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=None, nrows=None)
            except:
                fl_nm = os.sep.join([self.feat_dir, 'fm_'+str(self.window)+'0w_'+self.datastream+'_'+self.station+'_'+str(t0.year)+'.csv'])
                fMi = load_dataframe(fl_nm, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=None, nrows=None)
            # filter between t0 and t1
            fMi=fMi.loc[t0:t1-self.dt] #_dt=10*minute
            # resample at a constant rate (def self.dt)
            fMi=fMi.resample(str(int(self.dt.seconds/60))+'min').sum()
            # append to fMi
            fM.append(fMi)
        # vertical concat on time
        fM = pd.concat(fM)
        # horizontal concat on column
        #FM = pd.concat(FM, axis=1, sort=False)
        # Label vector corresponding to data windows
        ts = fM.index.values
        ys = [_is_eruption_in(days=self.lab_lb, from_time=t, tes = self.tes) for t in pd.to_datetime(ts)]
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
                    colm_keep=[ln.rstrip().split(',')[1].rstrip() for ln in fp.readlines() if (self.datastream in ln and 'cwt' not in ln)]
                    self.colm_keep=colm_keep
                    # temporal (to fix): if 'cwt' not in ln (features with 'cwt' contains ',' in their names, so its split in the middle)
                self.fM = self.fM[self.colm_keep] # not working
            elif islst:
                a=ft_lt[0]
                self.fM = self.fM[ft_lt] # not working
            else: 
                # Filter 100 feature with higher variance
                _l=[]
                _fM=self.fM
                for i in range(100):
                    _col=_fM.var().idxmax()
                    _l.append(_col)
                    _fM = _fM.drop(_col, axis=1)
                del _fM
                self.fM=self.fM[_l]
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
    def __init__(self, stations=None, window = 2., datastream = 'zsc2_dsarF', feat_dir=None, 
        dtb=None, dtf=None, tes_dir=None, feat_selc=None,noise_mirror=None,data_dir=None, 
        dt=None, lab_lb=2.,savefile_type='pkl'):
        self.stations=stations
        if self.stations:
            self.window=window
            self.datastream=datastream
            self.n_jobs=4
            self.feat_dir=feat_dir
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
            self._load_tes(tes_dir) # create self.tes (and self.tes_mirror) 
            self._load() # create dataframe from feature matrices
    def _load_tes(self,tes_dir):
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
        self.tes = {}
        for sta in self.stations:
            # get eruptions
            fl_nm = os.sep.join([tes_dir,sta+'_eruptive_periods.txt'])
            with open(fl_nm,'r') as fp:
                self.tes[sta] = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        # create noise mirror to fM
        if self.noise_mirror:
            self.tes_mirror = {}
            for sta in self.stations:
                # get initial and final date of record 
                if True:# get period from data
                    _td=TremorData(station=sta, data_dir=self.tes_dir)
                    _td.ti=_td.df.index[0]
                    _td.tf=_td.df.index[-1]
                    if sta is 'FWVZ':
                        _td.ti=datetimeify('2005-03-01 00:00:00')
                if False:# get period from feature matrices available by year
                    pass
                # select random dates (don't overlap with eruptive periods)
                _tes_mirror=[]
                for i in range(len(self.tes[sta])): # number of dates to create
                    _d = True
                    while _d:
                        _r=random_date(_td.ti, _td.tf)
                        # check if overlap wit eruptive periods
                        _=True # non overlap
                        for te in self.tes[sta]:
                            if _r in [te-1.5*month,te+1.5*month]:
                                _=False
                        if _:
                            _d = False
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
        ys = []
        _blk=0
        for sta in self.stations:
            fM = []
            for i, te in enumerate(self.tes[sta]): 
                # FeatureSta
                feat_sta = FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
                    ti=te-self.dtb, tf=te+self.dtf, tes_dir = self.tes_dir, dt=self.dt, lab_lb=self.lab_lb)
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
                    _feat_sta = FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
                        ti=te-self.dtb, tf=te+self.dtf, tes_dir = self.tes_dir)
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
        #     fM = []
        #     ys = []
        #     for sta in self.stations:
        #         for te in self.tes_mirror[sta]: 
        #             # FeatureSta
        #             feat_sta = FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
        #                 ti=te-self.dtb, tf=te+self.dtf, tes_dir = self.tes_dir)
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
            fl_nm = 'FM_'+str(int(self.window))+'w_'+self.datastream+'_'+'-'.join(self.stations)+'_'+str(self.dtb.days)+'dtb_'+str(self.dtf.days)+'dtf.'+self.savefile_type
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
        if noise_mirror:
            self.noise_mirror=True
        else:
            self.noise_mirror=False
        # assing attributes from file name
        def _load_atrib_from_file(fl_nm): 
            _ = fl_nm.split('.')[0]
            _ = _.split('_')[1:]
            self.stations=_[-3].split('-')
            self.window = int(_[0][0])
            self.dtf=int(_[-1][0])
            self.dtb=int(_[-2][0])
            self.data_stream=('_').join(_[1:-3])
            #
            self.feat_dir = feat_dir
        _load_atrib_from_file(fl_nm)
        # load feature matrix
        self.fM = load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        # load labels 
        _=fl_nm.find('.')
        _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        self.ys = load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        self.ys['time'] = pd.to_datetime(self.ys['time'])
        #
        if self.noise_mirror:
            fl_nm=fl_nm[:_]+'_nmirror'+fl_nm[_:]
            # load feature matrix
            self.fM_mirror = load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, 
                infer_datetime_format=False, header=0, skiprows=None, nrows=None)
            # load labels 
            _=fl_nm.find('.')
            _fl_nm=fl_nm[:_]+'_labels'+_fl_nm[_-1:]
            self.ys_mirror = load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, 
                infer_datetime_format=False, header=0, skiprows=None, nrows=None)
            self.ys_mirror['time'] = pd.to_datetime(self.ys['time'])
        #
    def svd(self):
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
        #_fM = self.fM.drop('time',axis=1)
        self.U,self.S,self.VT=np.linalg.svd(self.fM,full_matrices=True)
    def plot_svd_evals(self):
        ''' Plot eigen values from svd
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
        '''
        plt.rcParams['figure.figsize'] = [8, 8]
        fig1 = plt.figure()
        #
        ax1 = fig1.add_subplot(221)
        #ax1.semilogy(S,'-o',color='k')
        ax1.semilogy(self.S[:int(len(self.S))],'-o',color='k')
        ax2 = fig1.add_subplot(222)
        #ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')
        ax2.plot(np.cumsum(self.S[:int(len(self.S))])/np.sum(self.S[:int(len(self.S))]),'-o',color='k')
        #
        ax3 = fig1.add_subplot(223)
        #ax1.semilogy(S,'-o',color='k')
        ax3.semilogy(self.S[:int(len(self.S)/10)],'-o',color='k')
        ax4 = fig1.add_subplot(224)
        #ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')
        ax4.plot(np.cumsum(self.S[:int(len(self.S)/10)])/np.sum(self.S[:int(len(self.S)/10)]),'-o',color='k')
        #
        [ax.set_title('eigen values') for ax in [ax1,ax3]]
        [ax.set_title('cumulative eigen values') for ax in [ax2,ax4]]
        [ax.set_xlabel('# eigen value') for ax in [ax1,ax2,ax3,ax4]]
        plt.tight_layout()
        plt.show() 
    def plot_svd_pcomps(self,labels=None):
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
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
        fig.suptitle('fM projected in VT')
        #
        if labels:
            import random 
            random.seed(9001)
            colors = {}
            n = len(self.stations)
            for i,sta in enumerate(self.stations):
                colors[sta]='#%06X' % random.randint(0, 0xFFFFFF)
        #
        for i, ax in enumerate(fig.get_axes()): #[ax1,ax2,ax3])
            for j in range(self.fM.shape[0]):
                #x = VT[3,:] @ covariance_matrix[j,:].T
                y = self.VT[i,:] @ self.fM.values[j,:].T
                z = self.VT[i+1,:] @ self.fM.values[j,:].T
                if labels:
                    ax.plot(y,z, marker='.',color=colors[self.ys['station'][j]])                
                else:
                    ax.plot(y,z,'b.')
            #ax1.set_xlabel('pc3')
            ax.set_xlabel('pc'+str(i+1))
            ax.set_ylabel('pc'+str(i+2))
            #ax1.view_init(0,0)
            if i==0:
                for sta in self.stations:
                    ax.plot([],[], marker='.',color=colors[sta], label = sta)
        fig.legend()   
        plt.tight_layout()     
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
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
        fig.suptitle('fM projected in VT')
        #
        for i, ax in enumerate(fig.get_axes()):#[ax1,ax2,ax3]):#
            for j in range(self.fM.shape[0]):
                #x = VT[3,:] @ covariance_matrix[j,:].T
                y = self.VT[i,:] @ self.fM.values[j,:].T
                z = self.VT[i+1,:] @ self.fM.values[j,:].T
                ax.plot(y,z,'r.',zorder=0)
                #
                if j<self.fM_mirror.shape[0]-1:
                    a=self.VT[i,:]
                    b=self.fM_mirror.values[j,:].T
                    y_m = self.VT[i,:] @ self.fM_mirror.values[j,:].T
                    z_m = self.VT[i+1,:] @ self.fM_mirror.values[j,:].T
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
                    ax.plot([],[], marker='.',color='r', label = 'eruptive')
                    ax.plot([],[], marker='.',color='w', label = '('+str(j)+' points)')
                    ax.plot([],[], marker='.',color='b', label = 'non-eruptive')
                    ax.plot([],[], marker='.',color='w', label = '('+str(points_m)+' points)')
        fig.legend()   
        plt.tight_layout()     
        plt.show()

# testing
if __name__ == "__main__":
    if False: # FeatureSta class
        # FeatureSta
        feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
        tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
        feat_sta = FeaturesSta(station='WIZ', window = 2., datastream = 'zsc2_dsarF', feat_dir=feat_dir, 
            ti='2019-12-07', tf='2019-12-10', tes_dir = tes_dir)
        feat_sta.norm()
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        feat_sta.reduce(ft_lt=fl_lt)
    if True: # FeatureMulti class
        # 
        feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
        #feat_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\features'
        #tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
        datadir=r'U:\Research\EruptionForecasting\eruptions\data'
        datadir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        #
        if True: # create combined feature matrix
            stations=['WIZ']#,'FWVZ']#,'KRVZ']#,'VNSS','BELO','GOD','TBTN','MEA01']
            win = 2.
            dtb = 4
            dtf = 0
            datastream = 'zsc2_dsarF'
            ft = ['zsc2_dsarF__median']
            feat_stas = FeaturesMulti(stations=stations, window = win, datastream = datastream, feat_dir=feat_dir, 
                dtb=dtb, dtf=dtf, lab_lb=2,tes_dir=datadir, noise_mirror=True, data_dir=datadir, 
                    dt=10,savefile_type='csv',feat_selc=ft)#fl_lt
            #fl_nm = 'FM_'+str(int(win))+'w_'+datastream+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'+'.csv'
            feat_stas.save()#fl_nm=fl_nm)
            #
        if False: # load existing combined feature matrix 
            feat_stas = FeaturesMulti()
            #fl_nm = 'FM_'+str(win)+'w_'+datastream+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'+'.csv'
            fl_nm = 'FM_2w_zsc2_rsamF_WIZ-FWVZ_30dtb_0dtf.csv'
            #fl_nm = 'FM_2w_zsc2_dsarF_WIZ-FWVZ-KRVZ-PVV-VNSS-BELO-GOD-TBTN-MEA01_5dtb_2dtf.csv'
            feat_stas.load_fM(feat_dir=feat_dir,fl_nm=fl_nm,noise_mirror=True)
            #
            feat_stas.svd()
            #feat_stas.plot_svd_evals()
            #feat_stas.plot_svd_pcomps(labels=True)
            feat_stas.plot_svd_pcomps_noise_mirror()
            #





