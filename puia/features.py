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
            ft_lt   :   file directory containing list feature names (strs)
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
    """ Class to manage multiple feature matrices (list of FeaturesSta objects). 
        Feature matrices (from each station) are imported for the same period of time 
        using as references the eruptive times (see dtb and dtf). 
        This class performs PCA analysis on the feature matrices. 

        Attributes:
        -----------
        stations : list of strings
            list of stations in the feature matrix
        window  :   int
            window length for the features
        datastream  :   str
            data stream from where features were calculated
        dtb : float
            Days looking 'back' from eruptive times to import (for each station)
        dtf : float
            Days looking 'forward' from eruptive times to import (for each station)

        feat_list   :   list of string
            List of features  
        ref :   vector 
            vector of increasing integers as reference (for each time row in dataframe)
            add as column 'ref' to dataframe (to be used instead of time column)
        fM : pandas.DataFrame
            Feature matrix of combined ime series of features for multiple stations
        ys  :   pandas.DataFrame
            Binary labels for rows in fM. Index correspond to an increasing integer (reference number)
            Dates for each row in fM ar kept in column 'time'
        U   :   numpy matrix
            Unitary matrix 'U' from SVD of fM (shape nxm). Shape is nxn.
        S   :   numpy vector
            Vector of singular value 'S' from SVD of fM (shape nxm). Shape is nxm.
        VT  :   numpy matrix
            Transponse of unitary matrix 'V' from SVD of fM (shape mxm). Shape is mxm.
        Methods:
        --------
        _load_tes
            load eruptive dates of volcanoes in list
        _load
            load multiple feature matrices. Normalization and feature delection is performed here. 
        save
            save feature matrix
        svd
            compute svd on feature matrix 
        plot_svd_v
            plot eigen values from svd
        plot_svd_pc
            scatter plot of two principal components 
        cluster
            cluster principal components (e.g, DBCAN)
        plot_cluster
            plot cluster in a 2D scatter plot
    """
    def __init__(self, stations=None, window = 2., datastream = 'zsc2_dsarF', feat_dir=None, 
        dtb=None, dtf=None, tes_dir=None, feat_selc=None):
        self.stations=stations
        if self.stations:
            self.window=window
            self.datastream=datastream
            self.n_jobs=4
            self.feat_dir=feat_dir
            #self.file= os.sep.join(self._wd, 'fm_', str(window), '_', self.datastream,  '_', self.station, 
            self.dtb=dtb*day
            self.dtf=dtf*day
            self.fM=None
            self.feat_selc=feat_selc
            self.tes_dir=tes_dir
            self._load_tes(tes_dir) # create self.tes
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
        '''
        #
        self.tes = {}
        for sta in self.stations:
            # get eruptions
            fl_nm = os.sep.join([tes_dir,sta+'_eruptive_periods.txt'])
            with open(fl_nm,'r') as fp:
                self.tes[sta] = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    def _load(self):
        """ Load and combined feature matrices and label vectors from multiple stations. 
            Parameters:
            -----------
            Returns:
            --------
            Note:
            --------
            Matrices per stations are reduce to selected features (if given; self.feat_selc) 
            and normalize (before concatenation). Columns (features) with nans are remove too.
            Attributes created:
            self.fM : pd.DataFrame
                Combined feature matrix of multiple stations and eruptions
            self.ys : pd.DataFrame
                Label vector of multiple stations and eruptions
        """
        #
        fM = []
        ys = []
        for sta in self.stations:
            for te in self.tes[sta]: 
                # FeatureSta
                feat_sta = FeaturesSta(station=sta, window=self.window, datastream=self.datastream, feat_dir=self.feat_dir, 
                    ti=te-self.dtb, tf=te+self.dtf, tes_dir = self.tes_dir)
                if self.feat_selc:
                    feat_sta.reduce(ft_lt=self.feat_selc)
                feat_sta.norm()
                fM.append(feat_sta.fM)
                ys.append(feat_sta.ys)
                del feat_sta
        # concatenate and modifify index to a reference ni
        self.fM=pd.concat(fM)
        # drop columns with NaN
        self.fM=self.fM.drop(columns=self.fM.columns[self.fM.isna().any()].tolist())
        # create index with a reference number and create column 'time' 
        self.fM['time']=self.fM.index
        self.fM.index=range(self.fM.shape[0])
        self.ys=pd.concat(ys)
        self.ys['time']=self.ys.index
        self.ys.index=range(self.ys.shape[0])
        # modifiy index 
        del fM, ys
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
        if not fl_nm:
            fl_nm = 'FM_'+str(int(self.window))+'w_'+self.datastream+'_'+'-'.join(self.stations)+'_'+str(self.dtb.days)+'dtb_'+str(self.dtf.days)+'dtf'+'.csv'
        save_dataframe(self.fM, os.sep.join([self.feat_dir,fl_nm]), index=True)
        #save_dataframe(self.ys, os.sep.join([self.feat_dir,fl_nm]), index=True)
    def load_fM(self, feat_dir, fl_nm):
        ''' Load feature matrix from file. 
        '''
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
        # load file
        self.fM = load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        self.fM['time'] = pd.to_datetime(self.fM['time'])
        self.fM = self.fM.drop('time', axis=1)

    def svd(self):
        ''' Compute SVD (singular value decomposition) on feature matrix. 
        '''
        #_fM = self.fM.drop('time',axis=1)
        self.U,self.S,self.VT=np.linalg.svd(self.fM,full_matrices=True)
    def plot_svd_v(self):
        ''' Plot eigen values from svd
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
    def plot_svd_pc(self):
        ''' Plot fM (feature matrix) into principal component.
            Rows of fM are projected into rows of VT.  
        '''
        #
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
        fig.suptitle('fM projected in VT')
        #
        for i, ax in enumerate(fig.get_axes()):
            for j in range(self.fM.shape[0]):
                #x = VT[3,:] @ covariance_matrix[j,:].T
                y = self.VT[i,:] @ self.fM.values[j,:].T
                z = self.VT[i+1,:] @ self.fM.values[j,:].T
                ax.plot(y,z,'b.')
            #ax1.set_xlabel('pc3')
            ax.set_xlabel('pc'+str(i+1))
            ax.set_ylabel('pc'+str(i+2))
            #ax1.view_init(0,0)
        #fig.legend()   
        plt.tight_layout()     
        plt.show()

# testing
if __name__ == "__main__":
    if False: # FeatureSta class
        # FeatureSta
        feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
        tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
        feat_sta = FeaturesSta(station='WIZ', window = 2., datastream = 'zsc2_dsarF', feat_dir=feat_dir, ti='2019-12-07', tf='2019-12-10', tes_dir = tes_dir)
        feat_sta.norm()
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        feat_sta.reduce(ft_lt=fl_lt)
    if True: # FeatureMulti class
        # 
        feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
        tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        if False: # create combined feature matrix
            stations=['WIZ','FWVZ','KRVZ','PVV','VNSS','BELO','GOD','TBTN','MEA01']
            win = 2.
            dtb = 7 
            dtf = 0
            datastream = 'zsc2_dsarF'
            feat_stas = FeaturesMulti(stations=stations, window = win, datastream = datastream, feat_dir=feat_dir, 
                dtb=5, dtf=2, tes_dir=tes_dir, feat_selc=fl_lt)
            fl_nm = 'FM_'+str(int(win))+'w_'+datastream+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'+'.csv'
            feat_stas.save()#fl_nm=fl_nm)
            #
        if True: # load existing combined feature matrix 
            feat_stas = FeaturesMulti()
            #fl_nm = 'FM_'+str(win)+'w_'+datastream+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'+'.csv'
            fl_nm = 'FM_2w_zsc2_dsarF_WIZ-FWVZ_5dtb_2dtf.csv'
            fl_nm = 'FM_2w_zsc2_dsarF_WIZ-FWVZ-KRVZ-PVV-VNSS-BELO-GOD-TBTN-MEA01_5dtb_2dtf.csv'
            feat_stas.load_fM(feat_dir=feat_dir, fl_nm = fl_nm)
            #
            feat_stas.svd()
            #feat_stas.plot_svd_v()
            feat_stas.plot_svd_pc()




