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
#from forecast import *

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

Todo:
- have a longer conversation on feat selection 
- need a forecast method (?)
'''

def get_classifier(classifier):
    """ Return scikit-learn ML classifiers and search grids for input strings.
        Parameters:
        -----------
        classifier : str
            String designating which classifier to return.
        Returns:
        --------
        model : 
            Scikit-learn classifier object.
        grid : dict
            Scikit-learn hyperparameter grid dictionarie.
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
    if classifier == 'SVM':         # support vector machine
        model = SVC(class_weight='balanced')
        grid = {'C': [0.001,0.01,0.1,1,10], 'kernel': ['poly','rbf','sigmoid'],
            'degree': [2,3,4,5],'decision_function_shape':['ovo','ovr']}
    elif classifier == "KNN":        # k nearest neighbour
        model = KNeighborsClassifier()
        grid = {'n_neighbors': [3,6,12,24], 'weights': ['uniform','distance'],
            'p': [1,2,3]}
    elif classifier == "DT":        # decision tree
        model = DecisionTreeClassifier(class_weight='balanced')
        grid = {'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    elif classifier == "RF":        # random forest
        model = RandomForestClassifier(class_weight='balanced')
        grid = {'n_estimators': [10,30,100], 'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    elif classifier == "NN":        # neural network
        model = MLPClassifier(alpha=1, max_iter=1000)
        grid = {'activation': ['identity','logistic','tanh','relu'],
            'hidden_layer_sizes':[10,100]}
    elif classifier == "NB":        # naive bayes
        model = GaussianNB()
        grid = {'var_smoothing': [1.e-9]}
    elif classifier == "LR":        # logistic regression
        model = LogisticRegression(class_weight='balanced')
        grid = {'penalty': ['l2','l1','elasticnet'], 'C': [0.001,0.01,0.1,1,10]}
    else:
        raise ValueError("classifier '{:s}' not recognised".format(classifier))
    
    return model, grid
def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, method, random_state):
    ''' helper function for parallelising model training
    '''
    # undersample data
    rus = RandomUnderSampler(method, random_state=random_state+random_seed)
    a=fM.shape
    b=ys.shape
    fMt,yst = rus.fit_resample(fM,ys)
    yst = pd.Series(yst>0, index=range(len(yst)))
    fMt.index = yst.index

    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    fMt = fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid = get_classifier(classifier)            
        
    # check if model has already been trained
    pref = type(model).__name__
    fl = '{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    # train and save classifier
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy",error_score=np.nan)
    model_cv.fit(fMt,yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)

class TrainModelCombined(object):
    ''' Object for train forecast models. 
        Training involve multiple stations, and multiple seismic datastreams.
        Models are saved to be used later by ForecastTransLearn class. 

    Constructor arguments (and attributes):
    ----------------------
    stations : list of str
        Seismic stations providing data for modelling.
    window : float
        Length of data window in days.
    overlap : float
        Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
    look_forward : float
        Length of look-forward in days.
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
    Nfts : int
        Number of most-significant features to use in classifier.
    Ncl : int
        Number of classifier models to train.
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
    lab_lb  :   float
        Days looking back to assign label '1' from eruption times
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of window.
    n_jobs : int
        Number of CPUs to use for parallel tasks.
    feat_dir: str
        Repository location of feature matrices.
    root:   str
        model name for the folder and files (default). 
    rootdir : str
        Repository location on file system.
    modeldir : str
        Repository location to save model
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
    savefile_type : str
        Extension denoting file format for save/load. Options are csv, pkl (Python pickle) or hdf.
    no_erup : list of two 
        Remove eruption from trainning. Need to specified station and number of eruption (e.g., ['WIZ',4]; eruption number, as 4, start counting from 0)
        
    Methods:
    --------
    _load_feat
        Load feature matrix and label vector.
    _drop_features
        Drop columns from feature matrix.
    _collect_features
        Aggregate features used to train classifiers by frequency.
    train
        Construct classifier models.
    '''
    def __init__(self, stations=None, window = 2., overlap=.75, datastream = None, feat_dir=None, 
        dtb=None, dtf=None, tes_dir=None, feat_selc=None,noise_mirror=None,data_dir=None, model_dir=None,
        dt=None, lab_lb=2.,root=None,drop_features=None,savefile_type='pkl',feature_root=None,
        rootdir=None, no_erup=None):
        self.stations=stations
        self.window=window
        self.overlap = overlap
        self.stations=stations
        self.look_forward = dtf*day
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

        self.fM_mirror=None
        self.ys_mirror=None
        self.drop_features = []   
        self.use_only_features = []        
        self.tes_dir=tes_dir
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
        if rootdir is None:
            self.rootdir = '/'.join(getfile(currentframe()).split(os.sep)[:-2])
        else:
            self.rootdir = rootdir
        self.plotdir = r'{:s}/plots/{:s}'.format(self.rootdir, self.root)
        if model_dir:
            self.modeldir = model_dir+os.sep+self.root
        else:
            self.modeldir = r'{:s}/models/{:s}'.format(self.rootdir, self.root)
        if feat_dir is None:
            self.feat_dir = r'{:s}/features'.format(self.rootdir)
        else:
            self.featdir = feat_dir
        self.featfile = lambda ds,yr: (r'{:s}/fm_{:3.2f}w_{:s}_{:s}_{:d}.{:s}'.format(self.feat_dir,self.window,ds,self.station,yr,self.savefile_type))
        self.preddir = r'{:s}/predictions/{:s}'.format(self.rootdir, self.root)
        self.no_erup=no_erup
        #
        #self._load_tes(tes_dir) # create self.tes (and self.tes_mirror) 
        #self._load() # create dataframe from feature matrices
    def _load_feat(self):
        """ Load feature matrix and label vector.
            Parameters:
            -----------
            Returns:
            --------
            FM : pd.DataFrame
                Feature matrix.
            YS : pd.DataFrame
                Label vector.
            Note:
            -----
            If noise mirror is True, both standard and mirror matrices 
            are loaded in FM and YS for training. 
        """
        # load featurs matrix through FeatureMulti class
        FM=[]
        _FM=[]
        for i,datastream in enumerate(self.datastream):
            fl_nm='FM_'+str(int(self.window))+'w_'+datastream+'_'+'-'.join(self.stations)+'_'+str(self.dtb.days)+'dtb_'+str(self.dtf.days)+'dtf'+'.'+self.savefile_type
            if not os.path.isfile(os.sep.join([self.feat_dir,fl_nm])):
                print('Creating feature matrix:'+fl_nm+'\n . Will be saved in: '+self.feat_dir)
                if self.no_erup:
                    print('Eruption not considered:\t'+self.no_erup[0]+'\t'+str(self.no_erup[1]))
                feat_stas = FeaturesMulti(stations=self.stations, window = self.window, datastream = datastream, feat_dir=self.feat_dir, 
                    dtb=self.dtb.days, dtf=self.dtf.days, lab_lb=self.lab_lb,tes_dir=self.tes_dir, feat_selc=self.feat_selc, 
                        noise_mirror=self.noise_mirror, data_dir=self.tes_dir, dt=10,savefile_type=self.savefile_type,no_erup=self.no_erup)
                feat_stas.save()#fl_nm=fl_nm)
                FM.append(feat_stas.fM)
                del feat_stas
            else:
                # load feature matrix
                FM.append(load_dataframe(os.sep.join([self.feat_dir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None))
                # if self.noise_mirror:
                #     _nm=fl_nm[:-4]+'_nmirror'+'.csv'
                #     _FM.append(load_dataframe(os.sep.join([self.feat_dir,_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None))
        # horizontal concat on column
        FM = pd.concat(FM, axis=1, sort=False)
        # if self.noise_mirror:
        #     _FM = pd.concat(_FM, axis=1, sort=False)
        #     FM=pd.concat([FM,_FM], axis=0, sort=False)
        #     # drop columns with NaN (NaN columns not remove from noise matrix)
        #     FM=FM.drop(columns=FM.columns[FM.isna().any()].tolist())
        # load labels 
        _=fl_nm.find('.')
        _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        YS = load_dataframe(os.sep.join([self.feat_dir,_fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        YS['time'] = pd.to_datetime(YS['time'])
        # if self.noise_mirror:
        #     _nm=fl_nm[:-4]+'_nmirror'+'_labels'+'.csv'
        #     ys_mirror = load_dataframe(os.sep.join([self.feat_dir,_nm]), index_col=0, parse_dates=False, 
        #         infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        #     ys_mirror['time'] = pd.to_datetime(ys_mirror['time'])
        #     # concatenate with eruptive dataframe FM
        #     YS = pd.concat([YS,ys_mirror], axis=0, sort=False)
        # #
        return FM, YS
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
    def train(self, Nfts=20, Ncl=500, retrain=None, classifier="DT", random_seed=0,
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
        self.Nfts = Nfts
        self.method=method
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

        # save meta info file 
        with open(self.modeldir+os.sep+'meta.txt', 'w') as f:
            f.write('stations\t')
            for i,sta in enumerate(self.stations):
                f.write(sta+',') if i<len(self.stations)-1 else f.write(sta)
            f.write('\n')
            if self.no_erup:
                f.write('no_erup\t'+self.no_erup[0]+'\t'+str(self.no_erup[1])+'\n')
            f.write('datastreams\t')
            for i,ds in enumerate(self.datastream):
                f.write(ds+',') if i<len(self.datastream)-1 else f.write(ds)
            f.write('\n')
            f.write('window\t{}\n'.format(self.window))
            f.write('overlap\t{}\n'.format(self.overlap))
            f.write('dtb\t{}\n'.format(self.dtb.days))
            f.write('dtf\t{}\n'.format(self.dtf.days))
            f.write('lab_lb\t{}\n'.format(self.lab_lb))
            f.write('classifier\t{}\n'.format(self.classifier))
            f.write('Ncl\t{}\n'.format(self.Ncl))
            f.write('Nfts\t{}\n'.format(self.Nfts))
            f.write('method\t{}\n'.format(self.method))
            f.write('features\t')
            for i,ft in enumerate(fM.columns.values):
                f.write(ft+',') if i<len(fM.columns.values)-1 else f.write(ft)
            f.write('\n')

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
    fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
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
    
    if False: # TrainModelMulti class
        #
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
        ## (1) Create model
        if True:
            datastream = ['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF']
            stations=['WIZ']#,'KRVZ']
            dtb = 60 # looking back from eruption times
            dtf = 0  # looking forward from eruption times
            win=2.   # window length
            lab_lb=4.# days to label as eruptive before the eruption times 
            #
            rootdir=r'U:\Research\EruptionForecasting\eruptions'
            root='FM_'+str(int(win))+'w_'+'-'.join(datastream)+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'
            #
            fm0 = TrainModelCombined(stations=stations,window=win, overlap=0.75, dtb=dtb, dtf=dtf, datastream=datastream,
                rootdir=rootdir,root=root,feat_dir=feat_dir, data_dir=tes_dir,feat_selc=fl_lt, 
                    lab_lb=lab_lb,noise_mirror=True) # 
            #
            fm0.train(Nfts=20, Ncl=300, retrain=True, classifier="DT", random_seed=0, method=0.75, n_jobs=4)