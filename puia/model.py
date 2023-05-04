"""Forecast package for puia."""

__author__="""Alberto Ardid"""
__email__='alberto.ardid@canterbury.ac.nz'
__version__='0.1.0'

# general imports
import os, warnings, gc, joblib, logging
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from copy import copy
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
from glob import glob
import pandas as pd
from multiprocessing import Pool
from functools import partial
from fnmatch import fnmatch

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
logger=logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
from sklearn.exceptions import FitFailedWarning
from tsfresh.utilities.dataframe_functions import impute
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

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
from .utilities import datetimeify, load_dataframe, save_dataframe, makedir
from .data import SeismicData, GeneralData
from .features import Feature, _drop_features
from .forecast import Forecast, load_forecast, merge_forecasts

# constants
all_classifiers=["SVM","KNN",'DT','RF','NN','NB','LR']
_MONTH=timedelta(days=365.25/12)
month=_MONTH
_DAY=timedelta(days=1.)
day=_DAY
_MIN=timedelta(minutes=1)
n_jobs=0
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
        model=SVC(class_weight='balanced')
        grid={'C': [0.001,0.01,0.1,1,10], 'kernel': ['poly','rbf','sigmoid'],
            'degree': [2,3,4,5],'decision_function_shape':['ovo','ovr']}
    elif classifier == "KNN":        # k nearest neighbour
        model=KNeighborsClassifier()
        grid={'n_neighbors': [3,6,12,24], 'weights': ['uniform','distance'],
            'p': [1,2,3]}
    elif classifier == "DT":        # decision tree
        model=DecisionTreeClassifier(class_weight='balanced')
        grid={'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    elif classifier == "RF":        # random forest
        model=RandomForestClassifier(class_weight='balanced')
        grid={'n_estimators': [10,30,100], 'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    elif classifier == "NN":        # neural network
        model=MLPClassifier(alpha=1, max_iter=1000)
        grid={'activation': ['identity','logistic','tanh','relu'],
            'hidden_layer_sizes':[10,100]}
    elif classifier == "NB":        # naive bayes
        model=GaussianNB()
        grid={'var_smoothing': [1.e-9]}
    elif classifier == "LR":        # logistic regression
        model=LogisticRegression(class_weight='balanced')
        grid={'penalty': ['l2','l1','elasticnet'], 'C': [0.001,0.01,0.1,1,10]}
    else:
        raise ValueError("classifier '{:s}' not recognised".format(classifier))
    
    return model, grid
def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, method, random_state):
    ''' helper function for parallelising model training
    '''
    # undersample data
    # ys=yss['label']
    rus=RandomUnderSampler(method, random_state=random_state+random_seed)
    # fMyss=pd.concat([fM,yss],axis=1)    # DED temporary concat for co-sampling
    fMt,yst=rus.fit_resample(fM,ys['label'])
    # ysst=fMt[yss.columns]               # DED split off label DF post sampling (for inspection)
    # fMt=fMt.drop(columns=yss.columns)   # DED split off feature matrix
    yst=pd.Series(yst>0, index=range(len(yst)))
    fMt.index=yst.index

    # find significant features
    select=FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts=select.features[:Nfts]
    pvs=select.p_values[:Nfts]
    fMt=fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss=ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid=get_classifier(classifier)            
        
    # check if model has already been trained
    pref=type(model).__name__
    fl='{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    # train and save classifier
    model_cv=GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy",error_score=np.nan)
    model_cv.fit(fMt,yst)
    _=joblib.dump(model_cv.best_estimator_, fl, compress=3)
def forecast_models(fM, model_path, flps,yr):
    ''' helper function to parallelise model forecasting
    '''
    ypdfs=[]
    for flp in tqdm(flps, desc='forecasting'):
        flp,fl=flp
        # print('start:',flp)

        if os.path.isfile(fl): # load forecast
            ypdf0=load_dataframe(fl, index_col='time', infer_datetime_format=True, parse_dates=['time'])

        num=flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
        model=joblib.load(flp)
        with open(model_path+'{:s}.fts'.format(num)) as fp:
            lns=fp.readlines()
        fts=[' '.join(ln.rstrip().split()[1:]) for ln in lns]            
        
        if not os.path.isfile(fl):
            # simulate forecast period
            yp=model.predict(fM[fts])
            # save forecast
            ypdf=pd.DataFrame(yp, columns=['fcst{:s}'.format(num)], index=fM.index)
        else:
            fM2=fM.loc[fM.index>ypdf0.index[-1], fts]
            if fM2.shape[0] == 0:
                ypdf=ypdf0
            else:
                yp=model.predict(fM2)
                ypdf=pd.DataFrame(yp, columns=['fcst{:s}'.format(num)], index=fM2.index)
                ypdf=pd.concat([ypdf0, ypdf])

        save_dataframe(ypdf, fl, index=True, index_label='time')
        # print('finish:',flp)
        ypdfs.append(ypdf)
    return ypdfs
def forecast_one_model(fM, model_path, flp):
    ''' helper function to parallelise model forecasting
    '''
    flp,fl=flp
    print('start:',flp)

    if os.path.isfile(fl):
        ypdf0=load_dataframe(fl, index_col='time', infer_datetime_format=True, parse_dates=['time'])

    num=flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
    model=joblib.load(flp)
    with open(model_path+'{:s}.fts'.format(num)) as fp:
        lns=fp.readlines()
    fts=[' '.join(ln.rstrip().split()[1:]) for ln in lns]            
    
    if not os.path.isfile(fl):
        # simulate forecast period
        yp=model.predict(fM[fts])
        # save forecast
        ypdf=pd.DataFrame(yp, columns=['fcst{:s}'.format(num)], index=fM.index)
    else:
        fM2=fM.loc[fM.index>ypdf0.index[-1], fts]
        if fM2.shape[0] == 0:
            ypdf=ypdf0
        else:
            yp=model.predict(fM2)
            ypdf=pd.DataFrame(yp, columns=['fcst{:s}'.format(num)], index=fM2.index)
            ypdf=pd.concat([ypdf0, ypdf])

    save_dataframe(ypdf, fl, index=True, index_label='time')
    print('finish:',flp)
    return ypdf
def _training_labels(_tl, tes, dtf, eds):
        # one denotes non-eruptive data that has been used for model training
        _tl=_tl*0+1
        
        # two denotes eruptive data that 
        for te in tes:
            _tl[(_tl.index>(te-dtf))&(_tl.index<te)]=2
        
        # zero denotes any data that has not been used for model training
        for t0,t1 in eds:
            t0,t1=[datetimeify(t) for t in [t0,t1]]
            _tl[(_tl.index>t0)&(_tl.index<t1)]=0

        return _tl
    
class ForecastModel(object):
    """ Object for train and running forecast models.
        
        Constructor arguments:
        ----------------------
        window : float
            Length of data window in days.
        overlap : float
            Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
        look_forward : float
            Length of look-forward in days.
        exclude_dates : list
            List of datetime pairs to be dropped.
        station : str
            Seismic station providing data for modelling.
        feature_root : str
            Root for feature naming.
        feature_dir : str
            Directory to save feature matrices.
        ti : str, datetime.datetime
            Beginning of analysis period. If not given, will default to beginning of tremor data.
        tf : str, datetime.datetime
            End of analysis period. If not given, will default to end of tremor data.
        data_streams : list
            Data streams and transforms from which to extract features. Options are 'X', 'diff_X', 'log_X', 'inv_X', and 'stft_X' 
            where X is one of 'rsam', 'mf', 'hf', or 'dsar'.            
        root : str 
            Naming convention for forecast files. If not given, will default to 'fm_*Tw*wndw_*eta*ovlp_*Tlf*lkfd_*ds*' where
            Tw is the window length, eta is overlap fraction, Tlf is look-forward and ds are data streams.
        savefile_type : str
            Extension denoting file format for save/load. Options are csv, pkl (Python pickle) or hdf.
        Attributes:
        -----------
        data : SeismicData
            Object containing tremor data.
        dtw : datetime.timedelta
            Length of window.
        dtf : datetime.timedelta
            Length of look-forward.
        dt : datetime.timedelta
            Length between data samples (10 minutes).
        dto : datetime.timedelta
            Length of non-overlapping section of window.
        iw : int
            Number of samples in window.
        io : int
            Number of samples in overlapping section of window.
        ti_model : datetime.datetime
            Beginning of model analysis period.
        tf_model : datetime.datetime
            End of model analysis period.
        ti_train : datetime.datetime
            Beginning of model training period.
        tf_train : datetime.datetime
            End of model training period.
        ti_forecast : datetime.datetime
            Beginning of model forecast period.
        tf_forecast : datetime.datetime
            End of model forecast period.
        drop_features : list
            List of tsfresh feature names or feature calculators to drop during training.
            Facilitates manual dropping of correlated features.
        exclude_dates : list
            List of time windows to exclude during training. Facilitates dropping of eruption 
            windows within analysis period. E.g., exclude_dates=[['2012-06-01','2012-08-01'],
            ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
        use_only_features : list
            List of tsfresh feature names or calculators that training will be restricted to.
        compute_only_features : list
            List of tsfresh feature names or calcluators that feature extraction will be 
            restricted to.
        update_feature_matrix : bool
            Set this True in rare instances you want to extract feature matrix without the code
            trying first to update it.
        n_jobs : int
            Number of CPUs to use for parallel tasks.
        root_dir : str
            Repository location on file system.
        plot_dir : str
            Directory to save forecast plots.
        modeldir : str
            Directory to save forecast models (pickled sklearn objects).
        feat_dir : str
            Directory to save feature matrices.
        featfile : str
            File to save feature matrix to.
        fcst_dir : str
            Directory to save forecast model forecasts.
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
        _drop_features
            Drop columns from feature matrix.
        _exclude_dates
            Drop rows from feature matrix and label vector.
        _collect_features
            Aggregate features used to train classifiers by frequency.
        _model_alerts
            Compute issued alerts for model consensus.
        get_features
            Return feature matrix and label vector for a given period.
        train
            Construct classifier models.
        forecast
            Use classifier models to forecast eruption likelihood.
        hires_forecast
            Construct forecast at resolution of data.
        _compute_CI
            Calculate confidence interval on model output.
        plot_forecast
            Plot model forecast.
        get_performance
            Compute quality measures of a forecast.
        plot_accuracy
            Plot performance metrics for model.
        plot_features
            Plot frequency of extracted features by most significant.
        plot_feature_correlation
            Corner plot of feature correlation.
    """
    def __init__(self, window, overlap, look_forward, data, root, data_streams=[], savefile_type='pkl', 
                 feature_dir=None, data_dir=None, forecast_dir=None, model_dir=None, plot_dir=None):                
        # file access paths
        self.savefile_type=savefile_type
        self.root_dir='/'.join(getfile(currentframe()).split(os.sep)[:-2])        
        self.root=root
        self.data_dir=data_dir if data_dir else f'{self.root_dir}/data/'
        self.model_dir=model_dir if model_dir else f'{self.root_dir}/models'
        self.model_dir=f'{self.model_dir}/{self.root}'
        self.fcst_dir=forecast_dir if forecast_dir else f'{self.root_dir}/forecasts'
        self.fcst_dir=f'{self.fcst_dir}/{self.root}'
        self.plot_dir=plot_dir if plot_dir else f'{self.root_dir}/plots'
        self.plot_dir=f'{self.plot_dir}/{self.root}'
        
        # load input data
        self.data_streams=data_streams if data_streams else GeneralData(data, 'seismic', data_dir=data_dir, headers_only=True)
        self._data=data
        self._parse_data(data)

        # feature specification
        self.ft=Feature(self, window, overlap, look_forward, self.data.dt, feature_dir)
        
        # default attribute values
        self.drop_features=[]
        self.exclude_dates=[]
        self.use_only_features=[]
        self.compute_only_features=[]
        self.update_feature_matrix=True
        self._trained=False
        self.n_jobs=6
    def _parse_data(self, data):
        self.stations=[data,]
        self.data=SeismicData(data,self,data_dir=self.data_dir, transforms=self.data_streams)
    def _detect_model(self):
        """ Checks whether and what models have already been run.
        """
        fls=glob(self._use_model+os.sep+'*.fts')
        if len(fls) == 0:
            raise ValueError("no feature files in '{:s}'".format(self._use_model))

        inds=[int(float(fl.split(os.sep)[-1].split('.')[0])) for fl in fls if ('all.fts' not in fl)]
        if max(inds) != (len(inds) - 1):
            raise ValueError("feature file numbering in '{:s}' appears not consecutive".format(self._use_model))
        
        self.classifier=[]
        for classifier in all_classifiers:
            model=get_classifier(classifier)[0]
            pref=type(model).__name__
            if all([os.path.isfile(self._use_model+os.sep+'{:s}_{:04d}.pkl'.format(pref,ind)) for ind in inds]):
                self.classifier=classifier
                return
        raise ValueError("did not recognise models in '{:s}'".format(self._use_model))
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
        makedir(self.model_dir)
        if save is None:
            save='{:s}/all.fts'.format(self.model_dir)
        
        feats=[]
        fls=glob('{:s}/*.fts'.format(self.model_dir))
        for i,fl in enumerate(fls):
            if fl.split(os.sep)[-1].split('.')[0] in ['all','ranked']: continue
            with open(fl) as fp:
                lns=fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               

        labels=list(set(feats))
        freqs=[feats.count(label) for label in labels]
        labels=[label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs=sorted(freqs)[::-1]
        # write out feature frequencies
        with open(save, 'w') as fp:
            _=[fp.write('{:d},{:s}\n'.format(freq,ft)) for freq,ft in zip(freqs,labels)]
        return labels, freqs
    # public methods
    def get_features(self, ti=None, tf=None, n_jobs=1, drop_features=[], compute_only_features=[]):
        """ Return feature matrix and label vector for a given period.
            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of period to extract features (default is beginning of model analysis).
            tf : str, datetime.datetime
                End of period to extract features (default is end of model analysis).
            n_jobs : int
                Number of cores to use.
            drop_feautres : list
                tsfresh feature names or calculators to exclude from matrix.
            compute_only_features : list
                tsfresh feature names of calculators to return in matrix.
            
            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.Dataframe
                Label vector.
        """
        # initialise training interval
        self.ft.compute_only_features=compute_only_features
        self.n_jobs=n_jobs
        ti=self.ti_model if ti is None else datetimeify(ti)
        tf=self.tf_model if tf is None else datetimeify(tf)
        fM, ys=self._load_data(ti, tf)
        fM=_drop_features(fM, drop_features)
        return fM, ys
    
    def train(self, *args, **kwargs):
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
                windows within analysis period. E.g., exclude_dates=[['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
            use_only_features : list
                For specifying particular features to train with.
            method : float, str
                Passed to RandomUndersampler. If float, proportion of minor class in final sampling (two label).
                If str, method used for multi-label undersampling.
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
        self._tl=_training_labels(self.data.df[self.data.df.columns[0]], self.data.tes, self.ft.dtf, kwargs['exclude_dates'])
        self._train(*args, **kwargs)
    def _train(self, ti=None, tf=None, Nfts=20, Ncl=500, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[], method=0.75):
        self._trained=True
        self.classifier=classifier
        self.exclude_dates=exclude_dates
        self.ft.use_only_features=use_only_features
        self.n_jobs=n_jobs
        self.Ncl=Ncl
        makedir(self.model_dir)
        
        # check if any model training required
        if not retrain:
            run_models=False
            pref=type(get_classifier(self.classifier)[0]).__name__ 
            for i in range(Ncl):         
                if not os.path.isfile('{:s}/{:s}_{:04d}.pkl'.format(self.model_dir, pref, i)):
                    run_models=True
            if not run_models:
                return # not training required
        else:
            # delete old model files
            _=[os.remove(fl) for fl in  glob('{:s}/*'.format(self.model_dir))]

        
        # initialise training interval
        # self.ti_train=self.ti_model if ti is None else datetimeify(ti)
        # self.tf_train=self.tf_model if tf is None else datetimeify(tf)
        # if self.ti_train - self.dtw < self.data.ti:
        #     self.ti_train=self.data.ti+self.dtw

        # get feature matrix and label vector
        fM, ys=self.ft.load_data(ti, tf, exclude_dates)

        # manually drop features (columns)
        fM=_drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features=[df for df in self.use_only_features if df in fM.columns]
            fM=fM[use_only_features]
            Nfts=len(self.ft.use_only_features)+1

        # manually drop windows (rows)
        # fM, ys=self._exclude_dates(fM, ys, exclude_dates)
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")
        
        # select training subset
        # inds=(ys.index>=self.ti_train)&(ys.index<self.tf_train)
        # fM=fM.loc[inds]
        # ys=ys['label'].loc[inds]

        # choose mapper based on serial vs. parallel model
        if self.n_jobs > 1:
            p=Pool(self.n_jobs)
            mapper=p.imap
        else:
            mapper=map

        # fix training arguments
        f=partial(train_one_model, fM, ys, Nfts, self.model_dir, self.classifier, retrain, random_seed, method)
        
        # call training loop with progress bar
        # f(0)            # uncomment this to debug one call of training
        list(tqdm(mapper(f, range(Ncl)), desc='building models', total=Ncl))
        
        # close pool if necessary
        if self.n_jobs > 1:
            p.close()
            p.join()
        
        # free memory
        del fM
        gc.collect()
        self._collect_features()
    def forecast(self, ti, tf, recalculate=False, use_model=None, n_jobs=None, yr=None):
        """ Use classifier models to forecast eruption likelihood.
            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period (default is beginning of model analysis period).
            tf : str, datetime.datetime
                End of forecast period (default is end of model analysis period).
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            use_model : None or str
                Optionally pass path to pre-trained model directory in 'models'.
            n_jobs : int
                Number of cores to use.
            yr : int
                Year to produce forecast for. If None and hires, recursion will be activated.
            Returns:
            --------
            consensus : pd.DataFrame
                The model consensus, indexed by window date.
        """
        # special case of high resolution forecast where multiple feature matrices exist
        if yr is None: 
            forecasts=[]
            # fr=copy(self.feature_root)

            # use hires feature matrices for each year
            for yr in list(range(ti.year, tf.year+1)):
                t0=np.max([datetime(yr,1,1,0,0,0),ti,self.data.ti+self.ft.dtw])
                t1=np.min([datetime(yr+1,1,1,0,0,0),tf,self.data.tf])
                fcst=self.forecast(t0,t1,recalculate,use_model,n_jobs,yr)    
                forecasts.append(fcst)

            # merge the individual forecasts and ensure that original limits are respected
            fcst=merge_forecasts(forecasts)
            fcst.clip(ti, tf)
            # forecast=pd.concat(forecast, sort=False)
            return fcst

        self._use_model=use_model
        makedir(self.fcst_dir)
        yr_str='_{:d}'.format(yr) if yr is not None else ''
        fcst_fl='{:s}/forecast{:s}'.format(self.fcst_dir,'{:s}.{:s}'.format(yr_str, self.savefile_type))
                #
        if n_jobs is not None: 
            self.n_jobs=n_jobs 

        self.ti_forecast=datetimeify(ti)
        self.tf_forecast=datetimeify(tf)
        if self.tf_forecast > self.data.tf:
            self.tf_forecast=self.data.tf
        if self.ti_forecast - self.ft.dtw < self.data.ti:
            self.ti_forecast=self.data.ti+self.ft.dtw

        model_path=self.model_dir + os.sep
        if use_model is not None:
            self._detect_model()
            model_path=self._use_model+os.sep
            
        model=get_classifier(self.classifier)[0]

        # logic to determine which models need to be run and which to be 
        # read from disk
        pref=type(model).__name__
        models=glob('{:s}/{:s}_*.pkl'.format(model_path, pref))
        run_forecast=[]
        ys=[]        
        tis=[]

        # create a forecast for each model
        for model in models:
            # change location
            fcst=model.replace(model_path, self.fcst_dir+os.sep)
            # update filetype
            fcst=fcst.replace('.pkl','{:s}.{:s}'.format(yr_str, self.savefile_type))                

            # check if forecast already exists
            if os.path.isfile(fcst):
                if recalculate:
                    # delete forecast to be recalculated
                    os.remove(fcst)
                    run_forecast.append([model, fcst])  
                    tis.append(self.ti_forecast)
                else:
                    # load an existing forecast
                    y=load_dataframe(fcst, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                    # check if forecast spans the requested interval
                    if y.index[-1] < self.tf_forecast:
                        run_forecast.append([model, fcst])
                        tis.append(y.index[-1])
                    else:
                        ys.append(y)
            else:
                run_forecast.append([model, fcst])  
                tis.append(self.ti_forecast)
        
        if len(tis)>0:
            ti=np.min(tis)

        # generate new forecast
        if len(run_forecast)>0:
            # load feature matrix
            fM,_=self.ft.load_data(ti, self.tf_forecast)
            fM=fM.fillna(1.e-8)
            if fM.shape[0] == 0: return pd.DataFrame([],columns=['consensus'])
            ys += forecast_models(fM, model_path, run_forecast, yr)
        else:
            forecast=load_forecast(fcst_fl)
        
        # condense data frames and write output
        ys=pd.concat(ys, axis=1, sort=False)
        y=np.mean([ys[col].values for col in ys.columns if 'fcst' in col], axis=0)
        iy=ys.index.values
        y0=self.ft._get_label(iy)
        ilf=int(self.ft.look_forward*24*3600/self.ft.dt.total_seconds())
        tl=np.interp(pd.to_datetime(iy), self._tl.index, self._tl.values)
        tl[np.where((tl>0)&(tl<=1))]=1
        tl[np.where((tl>1)&(tl<=2))]=2
        forecast=Forecast(y, y0, iy, ilf, self.data.tes, tl)
        forecast.save(fcst_fl)
        
        # memory management
        if len(run_forecast)>0:
            del fM
            gc.collect()
            
        return forecast
    def hires_forecast(self, *args, **kwargs):
        return self._hires_forecast(*args, **kwargs)
    def _hires_forecast(self, ti, tf, recalculate=True, save=None, root=None, nztimezone=False, 
        n_jobs=None, threshold=0.8, alt_rsam=None, xlim=None):
        """ Construct forecast at resolution of data.
            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period.
            tf : str, datetime.datetime
                End of forecast period.
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            save : None or str
                If given, plot forecast and save to filename.
            root : None or str
                Naming convention for saving feature matrix.
            nztimezone : bool
                Flag to plot forecast using NZ time zone instead of UTC.            
            n_jobs : int
                CPUs to use when forecasting in parallel.
            Notes:
            ------
            Requires model to have already been trained.
        """
        # error checking
        try:
            _=self._trained
        except AttributeError:
            raise ValueError('Train model before constructing hires forecast.')
        
        if save == '':
            save='{:s}/hires_forecast.png'.format(self.plot_dir)
            makedir(self.plot_dir)
        
        if n_jobs is not None: self.n_jobs=n_jobs
 
        # calculate hires feature matrix
        if root is None:
            root=self.root+'_hires'
        _fm=ForecastModel(self.ft.window, 1., self.ft.look_forward, data=self.data.station, 
            data_streams=self.data_streams, root=root, savefile_type=self.savefile_type,
            feature_dir=self.ft.feat_dir, data_dir=self.data_dir)
        _fm._tl=self._tl
        if type(self) is MultiDataForecastModel:
            _fm.data=self.data
        # _fm.compute_only_features=list(set([ft.split('__')[1] for ft in self._collect_features()[0]]))
        
        # forecast on hires features
        ys=_fm.forecast(ti, tf, recalculate, use_model=self.model_dir, n_jobs=n_jobs)
        
        if save is not None:
            self._plot_hires_forecast(ys, save, threshold, nztimezone=nztimezone, alt_rsam=alt_rsam, xlim=xlim)

        return ys
    # plotting methods
    def _compute_CI(self, y):
        """ Computes a 95% confidence interval of the model consensus.
            Parameters:
            -----------
            y : numpy.array
                Model consensus returned by ForecastModel.forecast.
            
            Returns:
            --------
            ci : numpy.array
                95% confidence interval of the model consensus
        """
        ci=1.96*(np.sqrt(y*(1-y)/self.Ncl))
        return ci
    def plot_forecast(self, ys, threshold=0.75, save=None, xlim=['2019-12-01','2020-02-01']):
        """ Plot model forecast.
            Parameters:
            -----------
            ys : pandas.DataFrame
                Model forecast returned by ForecastModel.forecast.
            threshold : float
                Threshold consensus to declare alert.
            save : str
                File name to save figure.
            local_time : bool
                If True, switches plotting to local time (default is UTC).
        """
        makedir(self.plot_dir)
        if save is None:
            save='{:s}/forecast.png'.format(self.plot_dir)
        # set up figures and axes
        f=plt.figure(figsize=(24,15))
        N=10
        dy1,dy2=0.05, 0.05
        dy3=(1.-dy1-(N//2)*dy2)/(N//2)
        dx1,dx2=0.37,0.04
        axs=[plt.axes([0.10+(1-i//(N/2))*(dx1+dx2), dy1+(i%(N/2))*(dy2+dy3), dx1, dy3]) for i in range(N)][::-1]
        
        for i,ax in enumerate(axs[:-1]):
            ti,tf=[datetime.strptime('{:d}-01-01 00:00:00'.format(2011+i), '%Y-%m-%d %H:%M:%S'),
                datetime.strptime('{:d}-01-01 00:00:00'.format(2012+i), '%Y-%m-%d %H:%M:%S')]
            ax.set_xlim([ti,tf])
            ax.text(0.01,0.95,'{:4d}'.format(2011+i), transform=ax.transAxes, va='top', ha='left', size=16)
            
        ti,tf=[datetimeify(x) for x in xlim]
        axs[-1].set_xlim([ti, tf])
        
        # model forecast is generated for the END of each data window
        t=ys.index

        # average individual model responses
        ys=np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
        for i,ax in enumerate(axs):

            ax.set_ylim([-0.05, 1.05])
            ax.set_yticks([0,0.25,0.5, 0.75, 1.0])
            if i//(N/2) == 0:
                ax.set_ylabel('alert level')
            else:
                ax.set_yticklabels([])

            # shade training data
            ax.fill_between([self.ti_train, self.tf_train],[-0.05,-0.05],[1.05,1.05], color=[0.85,1,0.85], zorder=1, label='training data')            
            for exclude_date_range in self.exclude_dates:
                t0,t1=[datetimeify(dt) for dt in exclude_date_range]
                ax.fill_between([t0, t1],[-0.05,-0.05],[1.05,1.05], color=[1,1,1], zorder=2)            
            
            # consensus threshold
            ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

            # modelled alert
            ax.plot(t, ys, 'c-', label='modelled alert', zorder=4)

            # eruptions
            for te in self.data.tes:
                ax.axvline(te, color='k', linestyle='-', zorder=5)
            ax.axvline(te, color='k', linestyle='-', label='eruption')

        for tii,yi in zip(t, ys):
            if yi > threshold:
                i=(tii.year-2011)
                axs[i].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                j=(tii+self.dtf).year - 2011
                if j != i:
                    axs[j].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                
                if tii > ti:
                    axs[-1].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                
        for ax in axs:
            ax.fill_between([], [], [], color='y', label='eruption forecast')
        axs[-1].legend()
        
        plt.savefig(save, dpi=400)
        plt.close(f)
    def _plot_hires_forecast(self, ys, save, threshold=0.75, station='WIZ', nztimezone=False, alt_rsam=None, xlim=None):
        """ Plot model hires version of model forecast (single axes).
            Parameters:
            -----------
            ys : pandas.DataFrame
                Model forecast returned by ForecastModel.forecast.
            threshold : float
                Threshold consensus to declare alert.
            save : str
                File name to save figure.
        """
        
        makedir(self.plot_dir)
        # set up figures and axes
        f=plt.figure(figsize=(8,4))
        ax=plt.axes([0.1, 0.08, 0.8, 0.8])
        t=pd.to_datetime(ys.index.values)
        if True: # plot filtered data
            if 'zsc_rsamF' in self.data_streams and 'rsamF' not in self.data_streams:
                rsam=self.data.get_data(t[0], t[-1])['zsc_rsamF']
            else: 
                rsam=self.data.get_data(t[0], t[-1])['rsamF']
        else: 
            if 'zsc_rsam' in self.data_streams and 'rsam' not in self.data_streams:
                rsam=self.data.get_data(t[0], t[-1])['zsc_rsam']
            else: 
                rsam=self.data.get_data(t[0], t[-1])['rsam']
        trsam=rsam.index
        if nztimezone:
            t=to_nztimezone(t)
            trsam=to_nztimezone(trsam)
            ax.set_xlabel('Local time')
        else:
            ax.set_xlabel('UTC')
        y=np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
                
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0,0.25,0.50,0.75,1.00])
        ax.set_ylabel('ensemble mean')
    
        # consensus threshold
        ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

        # modelled alert
        ax.plot(t, y, 'c-', label='ensemble mean', zorder=4, lw=0.75)
        ci=self._compute_CI(y)
        ax.fill_between(t, (y-ci), (y+ci), color='c', zorder=5, alpha=0.3)
        ax_=ax.twinx()
        ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
        ax_.set_ylim([0,5])
        # ax_.set_xlim(ax.get_xlim())
        ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.75)

        for tii,yi in zip(t, y):
            if yi > threshold:
                ax.fill_between([tii, tii+self.dtf], [0,0], [100,100], color='y', zorder=3)

        for te in self.data.tes:
            ax.axvline(te, color='r', linestyle='--', zorder=10)    
        ax.plot([],[], 'r--', label='eruption')    
        ax.fill_between([], [], [], color='y', label='eruption forecast')
        ax.plot([],[],'k-', lw=0.75, label='RSAM')

        ax.legend(loc=2, ncol=2)

        tmax=np.max([t[-1], trsam[-1]])
        tmin=np.min([t[0], trsam[0]])
        if xlim is None:
            xlim=[tmin,tmax]
        tmax=xlim[-1] 
        tf=tmax 
        t0=tf.replace(hour=0, minute=0, second=0)
        dt=(tmax-tmin).total_seconds()
        if dt < 10.*24*3600:
            ndays=int(np.ceil(dt/(24*3600)))
            xts=[t0 - timedelta(days=i) for i in range(ndays)][::-1]
            lxts=[xt.strftime('%d %b') for xt in xts]
        elif dt < 20.*24*3600:
            ndays=int(np.ceil(dt/(24*3600))/2)
            xts=[t0 - timedelta(days=2*i) for i in range(ndays)][::-1]
            lxts=[xt.strftime('%d %b') for xt in xts]
        elif dt < 70.*24*3600:
            ndays=int(np.ceil(dt/(24*3600))/7)
            xts=[t0 - timedelta(days=7*i) for i in range(ndays)][::-1]
            lxts=[xt.strftime('%d %b') for xt in xts]
        elif dt < 365.25*24*3600:
            t0=tf.replace(day=1, hour=0, minute=0, second=0)
            nmonths=int(np.ceil(dt/(24*3600*365.25/12)))
            xts=[t0 - timedelta(days=i*365.25/12) for i in range(nmonths)][::-1]
            lxts=[xt.strftime('%b') for xt in xts]
        elif dt < 2*365.25*24*3600:
            t0=tf.replace(day=1, hour=0, minute=0, second=0)
            nmonths=int(np.ceil(dt/(24*3600*365.25/12))/2)
            xts=[t0 - timedelta(days=2*i*365.25/12) for i in range(nmonths)][::-1]
            lxts=[xt.strftime('%b %Y') for xt in xts]
        ax.set_xticks(xts)
        ax.set_xticklabels(lxts)
        
        ax.set_xlim(xlim)
        ax_.set_xlim(xlim)

        props=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.85, 0.95, self.data.station +' '+ ys.index[-1].strftime('%Y'), size=12, ha='left', va='top', transform=ax.transAxes, bbox=props)
        plt.savefig(save, dpi=400)
        plt.close(f)
    def get_performance(self, t, y, thresholds, ialert=None, dti=None):
        ''' Compute performance metrics for a forecast.
            Parameters:
            -----------
            t : array-like
                Time vector corresponding to model consensus.
            y : array-like
                Model consensus.
            thresholds: float
                Consensus values above which an alert is issued.
            ialert : int
                Number of data windows spanning an alert period.
            dti : datetime.timedelta
                Length of window overlap.
            Returns:
            --------
            FP : int
                Number of false positives at each threshold level.
            FN : int
                Number of false negatves at each threshold level.
            TP : int
                Number of true positives at each threshold level.
            TN : int
                Number of true negatives at each threshold level.
            dur : float
                Proportion of time spent inside an alert.
            MCC : float
                Matthew's correlation coefficient.
        '''
        # time series
        makedir(self.fcst_dir)
        label_file=self.fcst_dir+'/labels.pkl'
        if not os.path.isfile(label_file):
            ys=np.array([self.data._is_eruption_in(days=self.look_forward, from_time=ti) for ti in pd.to_datetime(t)])
            save_dataframe(ys, label_file)
        self._ys=load_dataframe(label_file)

        if ialert is None:
            ialert=self.look_forward/((1-self.overlap)*self.window)
        if dti is None:
            dti=timedelta(days=(1-self.overlap)*self.window)
        FP, FN, TP, TN, dur, MCC=[np.zeros(len(thresholds)) for i in range(6)]
        for j,threshold in enumerate(thresholds):
            if threshold == 0:
                FP[j]=int(1e8); dur[j]=1.; TP[j]=len(self.data.tes); TN[j]=1
            else:
                FP[j], FN[j], TP[j], TN[j], dur[j], MCC[j]=self._model_alerts(t, y, threshold, ialert, dti)

        return FP, FN, TP, TN, dur, MCC

class MultiDataForecastModel(ForecastModel):
    def __init__(self, *args, **kwargs):
        super(MultiDataForecastModel,self).__init__(*args, **kwargs)
    def _parse_data(self, data):
        station=list(data.keys())[0]
        self.stations=[station,]
        datas=[]
        for d in data[station]:
            gd=GeneralData(station,d,self,data_dir=self.data_dir,transforms=self.data_streams)
            datas.append(gd)
        j=np.argmin([d.dt for d in datas])
        tf=np.min([d.tf for d in datas])
        ti=np.max([d.ti for d in datas])
        dfn=datas[j].df[(datas[j].df.index>=ti)&(datas[j].df.index<tf)]
        for i,d in enumerate(datas):
            if i == j:
                continue
            for c in d.df.columns:
                dfn[c]=np.interp(dfn.index, xp=d.df.index, fp=d.df[c])
        self.data=datas[j]
        self.data._df=dfn
        self.data.ti=ti
        self.data.tf=tf
        self.data.data_streams=self.data_streams
    
class MultiVolcanoForecastModel(ForecastModel):
    def __init__(self, *args, **kwargs):
        super(MultiVolcanoForecastModel,self).__init__(*args, **kwargs)
    def _parse_data(self, data):
        self.stations=list(data.keys())
        datas=[]
        for station in self.stations:
            gd=SeismicData(station,self,data_dir=self.data_dir,transforms=self.data_streams)
            datas.append(gd)
            if not len(data[station]):
                data[station]=[gd.ti, gd.tf]
            else:
                data[station]=[datetimeify(t) for t in data[station]]
        self.data=dict(zip(self.stations, datas))
        self._train_dates=data
    def train(self, *args, **kwargs):
        self._tls={}
        for station, data in self.data.items():
            _tl=_training_labels(data.df[data.df.columns[0]], data.tes, self.ft.dtf, kwargs['exclude_dates'][station])
            self._tls.update({station:_tl})            
        self._train(*args, **kwargs)
    def hires_forecast(self, station, *args, **kwargs):
        from copy import deepcopy
        data_copy=deepcopy(self.data)
        self.data=data_copy[station]
        self._tl=self._tls[station]
        ys=self._hires_forecast(*args, **kwargs)
        self.data=data_copy
        return ys

