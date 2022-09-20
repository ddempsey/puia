"""Forecast package for puia."""

__author__ = """Alberto Ardid"""
__email__ = 'alberto.ardid@canterbury.ac.nz'
__version__ = '0.1.0'

# general imports
import os, shutil, warnings, gc, joblib, pickle
import numpy as np
from datetime import datetime, timedelta
from copy import copy
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
from glob import glob
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from tqdm import tqdm
from multiprocessing import Pool
from time import sleep
from scipy.integrate import cumtrapz
from scipy.signal import stft
from functools import partial
from fnmatch import fnmatch

# ObsPy imports
try:
    from obspy.clients.fdsn.header import FDSNException
    from obspy.clients.fdsn import Client as FDSNClient 
    from obspy import UTCDateTime
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from obspy.signal.filter import bandpass
    from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
    from obspy.clients.fdsn.header import FDSNNoDataException
    failedobspyimport = False
except:
    failedobspyimport = True

# feature recognition imports
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import FeatureSelector
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from imblearn.under_sampling import RandomUnderSampler

# classifier imports
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# package imports
from utilities import datetimeify, load_dataframe, save_dataframe
from data import TremorData
from features import FeaturesSta, FeaturesMulti
from forecast import *

# constants
all_classifiers = ["SVM","KNN",'DT','RF','NN','NB','LR']
_MONTH = timedelta(days=365.25/12)
month=_MONTH
_DAY = timedelta(days=1.)
day=_DAY
_MIN = timedelta(minutes=1)
makedir = lambda name: os.makedirs(name, exist_ok=True)
n_jobs = 0
'''
Here are two feature clases that operarte a diferent levels. 
FeatureSta oject manages single stations, and FeaturesMulti object manage multiple stations using FeatureSta objects. 
This objects just manipulates feature matrices that already exist. 
'''

class TrainModelCombined(object):
    '''

    Constructor arguments:
    ----------------------
    window : float
        Length of data window in days.
    overlap : float
        Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
    look_forward : float
        Length of look-forward in days.
    stations : str
        Seismic stations providing data for modelling.
    feature_root : str
        Root for feature naming.
    feature_dir : str
        Directory to save feature matrices.
    data_streams : list
        Data streams and transforms from which to extract features. Options are 'X', 'diff_X', 'log_X', 'inv_X', and 'stft_X' 
        where X is one of 'rsam', 'mf', 'hf', or 'dsar'.            
    root : str 
        Naming convention for train files. If not given, will default to 'fm_*Tw*wndw_*eta*ovlp_*Tlf*lkfd_*ds*' where
        Tw is the window length, eta is overlap fraction, Tlf is look-forward and ds are data streams.

    Attributes:
    -----------
    dtw : datetime.timedelta
        Length of window.
    dtf : datetime.timedelta
        Length of look-forward.
    dtb : float
        Days looking 'back' from eruptive times (for each station)
    dt : datetime.timedelta
        Length between data samples (10 minutes).
    dto : datetime.timedelta
        Length of non-overlapping section of window.
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of window.

    n_jobs : int
        Number of CPUs to use for parallel tasks.
    rootdir : str
        Repository location on file system.
    preddir : str
        Directory to save forecast model predictions.
    
    Methods:
    --------
    _detect_model
        Checks whether and what models have already been run.
    _construct_windows
        Create overlapping data windows for feature extraction.
    _extract_features
        Extract features from windowed data.
    _extract_featuresX
        Abstracts key feature extraction steps from bookkeeping in _extract_features
    _get_label
        Compute label vector.
    _load_data
        Load feature matrix and label vector.
    _model_alerts
        Compute issued alerts for model consensus.

    train
        Construct classifier models.
    plot_accuracy
        Plot performance metrics for model.
    plot_features
        Plot frequency of extracted features by most significant.
    plot_feature_correlation
        Corner plot of feature correlation.
    '''
    def __init__(self, stations=None, window = 2., overlap=.75, datastream = None, feat_dir=None, 
        dtb=None, dtf=None, tes_dir=None, feat_selc=None,noise_mirror=None,data_dir=None, 
        dt=None, lab_lb=2.,root=None,drop_features=None,savefile_type='pkl',feature_root=None):
        self.stations=stations
        self.window=window
        self.overlap = overlap
        self.stations=stations
        self.look_forward = dtf*day
        self.data_dir=tes_dir
        self.dtw = timedelta(days=self.window)
        self.dto = (1.-self.overlap)*self.dtw
        self.iw = int(self.window*6*24)         
        self.io = int(self.overlap*self.iw) 
        #
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
        self.drop_features = []   
        self.use_only_features = []        
        self.tes_dir=tes_dir
        self.noise_mirror=noise_mirror
        self.feat_selc=feat_selc
        self.noise_mirror=noise_mirror
        #
        # naming convention and file system attributes
        self.savefile_type = savefile_type
        if root is None:
            self.root = 'fm_{:3.2f}wndw_{:3.2f}ovlp_{:3.2f}lkfd'.format(self.window, self.overlap, self.look_forward)
            self.root += '_'+((('{:s}-')*len(self.data_streams))[:-1]).format(*sorted(self.data_streams))
        else:
            self.root = root
        self.feature_root=feature_root
        self.rootdir = '/'.join(getfile(currentframe()).split(os.sep)[:-2])
        self.plotdir = r'{:s}/plots/{:s}'.format(self.rootdir, self.root)
        self.modeldir = r'{:s}/models/{:s}'.format(self.rootdir, self.root)
        if feat_dir is None:
            self.feat_dir = r'{:s}/features'.format(self.rootdir)
        else:
            self.featdir = feat_dir
        self.featfile = lambda ds,yr: (r'{:s}/fm_{:3.2f}w_{:s}_{:s}_{:d}.{:s}'.format(self.feat_dir,self.window,ds,self.station,yr,self.savefile_type))
        self.preddir = r'{:s}/predictions/{:s}'.format(self.rootdir, self.root)
        #
        #self._load_tes(tes_dir) # create self.tes (and self.tes_mirror) 
        #self._load() # create dataframe from feature matrices
    def _load_feat(self):
        """ Load feature matrix and label vector.
            Parameters:
            -----------
            yr : int
                Year to load data for. If None and hires, recursion will activate.
            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.DataFrame
                Label vector.
        """
        # load featurs matrix through FeatureMulti class
        FM=[]
        for i,datastream in enumerate(self.datastream):
            fl_nm='FM_'+str(int(self.window))+'w_'+datastream+'_'+'-'.join(self.stations)+'_'+str(self.dtb.days)+'dtb_'+str(self.dtf.days)+'dtf'+'.csv'
            if os.path.isfile(fl_nm):
                # load feature matrix
                FM.append(load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None))
            else: 
                #
                print('Loading feature matrix:'+fl_nm+'\n . Will be saved in: '+self.feat_dir)
                feat_stas = FeaturesMulti(stations=self.stations, window = self.window, datastream = datastream, feat_dir=self.feat_dir, 
                    dtb=self.dtb.days, dtf=self.dtf.days, lab_lb=7,tes_dir=tes_dir, feat_selc=self.feat_selc, noise_mirror=self.noise_mirror, data_dir=self.tes_dir, dt=10)
                feat_stas.save()#fl_nm=fl_nm)
                # load feature matrix
                FM.append(load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None))

        # horizontal concat on column
        FM = pd.concat(FM, axis=1, sort=False)
        # load labels 
        _=fl_nm.find('.')
        _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        YS = load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        YS['time'] = pd.to_datetime(YS['time'])

        # load noise mirror feature sections
        if self.noise_mirror:
            fl_nm=fl_nm[:_]+'_nmirror'+fl_nm[_:]
            # load feature matrix
            fM_mirror = load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, 
                infer_datetime_format=False, header=0, skiprows=None, nrows=None)
            # load labels 
            _=fl_nm.find('.')
            _fl_nm=fl_nm[:_]+'_labels'+_fl_nm[_-1:]
            ys_mirror = load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, 
                infer_datetime_format=False, header=0, skiprows=None, nrows=None)
            ys_mirror['time'] = pd.to_datetime(ys_mirror['time'])
            # concatenate with eruptive dataframe FM
            FM=[FM,fM_mirror]
            YS=[YS,ys_mirror]
            # vertical concat on row
            FM = pd.concat(FM, axis=0, sort=False)
            YS = pd.concat(YS, axis=0, sort=False)
        #
        return FM, ys
    def _drop_features(self, X, drop_features):
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
        self.drop_features = drop_features
        if len(self.drop_features) != 0:
            cfp = ComprehensiveFCParameters()
            df2 = []
            for df in self.drop_features:
                if df in X.columns:
                    df2.append(df)          # exact match
                else:
                    if df in cfp.keys() or df in ['fft_coefficient_hann']:
                        df = '*__{:s}__*'.format(df)    # feature calculator
                    # wildcard match
                    df2 += [col for col in X.columns if fnmatch(col, df)]              
            X = X.drop(columns=df2)
        return X
    def _collect_features(self, save=None):
        """ Aggregate features used to train classifiers by frequency.
            Parameters:
            -----------
            save : None or str
                If given, name of file to save feature frequencies. Defaults to all.fts
                if model directory.
            Returns:
            --------
            labels : list
                Feature names.
            freqs : list
                Frequency of feature appearance in classifier models.
        """
        makedir(self.modeldir)
        if save is None:
            save = '{:s}/all.fts'.format(self.modeldir)
        
        feats = []
        fls = glob('{:s}/*.fts'.format(self.modeldir))
        for i,fl in enumerate(fls):
            if fl.split(os.sep)[-1].split('.')[0] in ['all','ranked']: continue
            with open(fl) as fp:
                lns = fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               

        labels = list(set(feats))
        freqs = [feats.count(label) for label in labels]
        labels = [label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs = sorted(freqs)[::-1]
        # write out feature frequencies
        with open(save, 'w') as fp:
            _ = [fp.write('{:d},{:s}\n'.format(freq,ft)) for freq,ft in zip(freqs,labels)]
        return labels, freqs
    def train(self, Nfts=20, Ncl=500, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, method=0.75):
        """ Construct classifier models.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of training period (default is beginning model analysis period).
            tf : str, datetime.datetime
                End of training period (default is end of model analysis period).
            Nfts : int
                Number of most-significant features to use in classifier.
            Ncl : int
                Number of classifier models to train.
            retrain : boolean
                Use saved models (False) or train new ones.
            classifier : str, list
                String or list of strings denoting which classifiers to train (see options below.)
            random_seed : int
                Set the seed for the undersampler, for repeatability.
            drop_features : list
                Names of tsfresh features to be dropped prior to training (for manual elimination of 
                feature correlation.)
            n_jobs : int
                CPUs to use when training classifiers in parallel.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Classifier options:
            -------------------
            SVM - Support Vector Machine.
            KNN - k-Nearest Neighbors
            DT - Decision Tree
            RF - Random Forest
            NN - Neural Network
            NB - Naive Bayes
            LR - Logistic Regression
        """
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.Ncl = Ncl
        makedir(self.modeldir)
        
        # check if any model training required
        if not retrain:
            run_models = False
            pref = type(get_classifier(self.classifier)[0]).__name__ 
            for i in range(Ncl):         
                if not os.path.isfile('{:s}/{:s}_{:04d}.pkl'.format(self.modeldir, pref, i)):
                    run_models = True
            if not run_models:
                return # not training required
        else:
            # delete old model files
            _ = [os.remove(fl) for fl in  glob('{:s}/*'.format(self.modeldir))]

        # get feature matrix and label vector
        fM, yss = self._load_feat()
        ys = yss['label']
        #

        # manually drop features (columns)
        fM = self._drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features = [df for df in self.use_only_features if df in fM.columns]
            fM = fM[use_only_features]
            Nfts = len(use_only_features)+1

        # check dimensionality has been preserved
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        f = partial(train_one_model, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed, method)

        # train models with glorious progress bar
        for i, _ in enumerate(mapper(f, range(Ncl))):
            cf = (i+1)/Ncl
            print(f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
        if self.n_jobs > 1:
            p.close()
            p.join()
        
        # free memory
        del fM, ys, yss
        gc.collect()
        self._collect_features()
   
# testing
if __name__ == "__main__":
    tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
    feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
    if False: # ForecastModel class
        #
        data_streams = ['zsc2_rsamF','zsc2_dsarF','zsc2_hfF','zsc2_mfF']
        fm = ForecastModel(station = 'WIZ', ti='2012-01-01', tf='2019-12-31', window=2., overlap=0.75, 
            look_forward=2., data_streams=data_streams, root='test', feature_dir=feat_dir, 
                data_dir=tes_dir,savefile_type='pkl')
        # drop features 
        drop_features = ['linear_trend_timewise','agg_linear_trend']
        drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
            '*attr_"angle"*']  
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
        # train
        te = fm.data.tes[-1]
        fm.train(ti='2012-01-01', tf='2019-12-31', drop_features=drop_features, exclude_dates=[[te-month/2,te+month/2],], 
            retrain=True, n_jobs=n_jobs, Nfts=10, Ncl=50) #  use_only_features=use_only_features, exclude_dates=[[te-month,te+month],]
        # forecast
        ys = fm.forecast(ti='2012-01-01', tf='2019-12-31', recalculate=True, n_jobs=n_jobs)    
        # plot
        fm.plot_forecast(ys, threshold=0.75, xlim = [te-month/4., te+month/15.], 
            save=r'{:s}/forecast.png'.format(fm.plotdir))
        pass
    
    if True: # TrainModelMulti class
        #
        tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
        feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        #
        datastream = ['zsc2_mfF','zsc2_hfF','zsc2_rsamF','zsc2_dsarF']#,'zsc2_mfF','zsc2_hfF']#['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF', 'log_zsc2_rsamF', 'diff_zsc2_rsamF']
        stations=['WIZ','FWVZ']
        dtb = 14
        dtf = 0
        #
        # load feature matrices for WIZ and FWVZ
        fm0 = TrainModelCombined(stations=stations,window=2., overlap=0.75, dtb=dtb, dtf=dtf, datastream=datastream,
            root='combined_trainer',feat_dir=feat_dir, data_dir=tes_dir,feat_selc=fl_lt, noise_mirror=True) # 
        #
        fm0.train(Nfts=10, Ncl=10, retrain=True, classifier="DT", random_seed=0, method=0.75, n_jobs=0)






