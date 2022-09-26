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

# constants
all_classifiers = ["SVM","KNN",'DT','RF','NN','NB','LR']
_MONTH = timedelta(days=365.25/12)
month=_MONTH
_DAY = timedelta(days=1.)
day=_DAY
_MIN = timedelta(minutes=1)
makedir = lambda name: os.makedirs(name, exist_ok=True)
'''
Here are two feature clases that operarte a diferent levels. 
FeatureSta oject manages single stations, and FeaturesMulti object manage multiple stations using FeatureSta objects. 
This objects just manipulates feature matrices that already exist. 
'''
# ForecastModel class (from whakaari repo)
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

def predict_models(fM, model_path, flps,yr):
    ''' helper function to parallelise model forecasting
    '''
    ypdfs = []
    for flp in tqdm(flps, desc='forecasting {:d}'.format(yr)):
        flp,fl = flp
        # print('start:',flp)

        if os.path.isfile(fl):
            ypdf0 = load_dataframe(fl, index_col='time', infer_datetime_format=True, parse_dates=['time'])

        num = flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
        model = joblib.load(flp)
        with open(model_path+'{:s}.fts'.format(num)) as fp:
            lns = fp.readlines()
        fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]            
        
        if not os.path.isfile(fl):
            # simulate prediction period
            yp = model.predict(fM[fts])
            # save prediction
            ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM.index)
        else:
            fM2 = fM.loc[fM.index>ypdf0.index[-1], fts]
            if fM2.shape[0] == 0:
                ypdf = ypdf0
            else:
                yp = model.predict(fM2)
                ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM2.index)
                ypdf = pd.concat([ypdf0, ypdf])

        save_dataframe(ypdf, fl, index=True, index_label='time')
        # print('finish:',flp)
        ypdfs.append(ypdf)
    return ypdfs
def predict_one_model(fM, model_path, flp):
    ''' helper function to parallelise model forecasting
    '''
    flp,fl = flp
    print('start:',flp)

    if os.path.isfile(fl):
        ypdf0 = load_dataframe(fl, index_col='time', infer_datetime_format=True, parse_dates=['time'])

    num = flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
    model = joblib.load(flp)
    with open(model_path+'{:s}.fts'.format(num)) as fp:
        lns = fp.readlines()
    fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]            
    
    if not os.path.isfile(fl):
        # simulate predicton period
        yp = model.predict(fM[fts])
        # save prediction
        ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM.index)
    else:
        fM2 = fM.loc[fM.index>ypdf0.index[-1], fts]
        if fM2.shape[0] == 0:
            ypdf = ypdf0
        else:
            yp = model.predict(fM2)
            ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM2.index)
            ypdf = pd.concat([ypdf0, ypdf])

    save_dataframe(ypdf, fl, index=True, index_label='time')
    print('finish:',flp)
    return ypdf

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
        data : TremorData
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
            windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
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
        rootdir : str
            Repository location on file system.
        plotdir : str
            Directory to save forecast plots.
        modeldir : str
            Directory to save forecast models (pickled sklearn objects).
        featdir : str
            Directory to save feature matrices.
        featfile : str
            File to save feature matrix to.
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
    def __init__(self, window, overlap, look_forward, exclude_dates=[], station=None, ti=None, tf=None, 
        data_streams=['rsam','mf','hf','dsar'], root=None, savefile_type='pkl', feature_root=None, 
        feature_dir=None, data_dir=None):
        self.window = window
        self.overlap = overlap
        self.station = station
        self.exclude_dates = exclude_dates
        self.look_forward = look_forward
        self.data_streams = data_streams
        self.data_dir=data_dir
        self.data = TremorData(self.station, parent=self, data_dir=data_dir)
        if any(['_' in ds for ds in data_streams]):
            self.data._compute_transforms()
        if any([d not in self.data.df.columns for d in self.data_streams]):
            raise ValueError("data restricted to any of {}".format(self.data.df.columns))
        if ti is None: ti = self.data.ti
        if tf is None: tf = self.data.tf
        self.ti_model = datetimeify(ti)
        self.tf_model = datetimeify(tf)
        if self.tf_model > self.data.tf:
            t0,t1 = [self.tf_model.strftime('%Y-%m-%d %H:%M'), self.data.tf.strftime('%Y-%m-%d %H:%M')]
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(t0,t1))
        if self.ti_model < self.data.ti:
            t0,t1 = [self.ti_model.strftime('%Y-%m-%d %H:%M'), self.data.ti.strftime('%Y-%m-%d %H:%M')]
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(t0,t1))
        self.dtw = timedelta(days=self.window)
        self.dtf = timedelta(days=self.look_forward)
        self.dt = timedelta(seconds=600)
        self.dto = (1.-self.overlap)*self.dtw
        self.iw = int(self.window*6*24)         
        self.io = int(self.overlap*self.iw)      
        if self.io == self.iw: self.io -= 1
        self.window = self.iw*1./(6*24)
        self.dtw = timedelta(days=self.window)
        if self.ti_model - self.dtw < self.data.ti:
            self.ti_model = self.data.ti+self.dtw
        self.overlap = self.io*1./self.iw
        self.dto = (1.-self.overlap)*self.dtw
        
        self.drop_features = []
        self.exclude_dates = []
        self.use_only_features = []
        self.compute_only_features = []
        self.update_feature_matrix = True
        self.n_jobs = 6

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
        if feature_dir is None:
            self.featdir = r'{:s}/features'.format(self.rootdir)
        else:
            self.featdir = feature_dir
        self.featfile = lambda ds,yr: (r'{:s}/fm_{:3.2f}w_{:s}_{:s}_{:d}.{:s}'.format(self.featdir,self.window,ds,self.station,yr,self.savefile_type))
        self.preddir = r'{:s}/predictions/{:s}'.format(self.rootdir, self.root)
    # private helper methods
    def _detect_model(self):
        """ Checks whether and what models have already been run.
        """
        fls = glob(self._use_model+os.sep+'*.fts')
        if len(fls) == 0:
            raise ValueError("no feature files in '{:s}'".format(self._use_model))

        inds = [int(float(fl.split(os.sep)[-1].split('.')[0])) for fl in fls if ('all.fts' not in fl)]
        if max(inds) != (len(inds) - 1):
            raise ValueError("feature file numbering in '{:s}' appears not consecutive".format(self._use_model))
        
        self.classifier = []
        for classifier in all_classifiers:
            model = get_classifier(classifier)[0]
            pref = type(model).__name__
            if all([os.path.isfile(self._use_model+os.sep+'{:s}_{:04d}.pkl'.format(pref,ind)) for ind in inds]):
                self.classifier = classifier
                return
        raise ValueError("did not recognise models in '{:s}'".format(self._use_model))
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
                    print('this shouldn\'t be happening:',self.data.station,ti-self.dtw,ti+(Nw-1)*self.dto,self.data.df.index[0],self.data.df.index[-1])
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
            Saves feature matrix to $rootdir/features/$root_features.csv to avoid recalculation.
        """
        makedir(self.featdir)
        # number of windows in feature request
        Nw = int(np.floor(((tf-ti)/self.dt-1)/(self.iw-self.io)))+1
        Nmax = 6*24*31 # max number of construct windows per iteration (6*24*30 windows: ~ a month of hires, overlap of 1.)

        # file naming convention
        yr = ti.year
        ftfl = self.featfile(ds,yr)

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
        else:
            # drop features if relevant
            _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]
        nj = self.n_jobs
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
        # return pre loaded
        try:
            if ti == self.ti_prev and tf == self.tf_prev:
                return self.fM, self.ys
        except AttributeError:
            pass

        # range checking
        if tf > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(tf, self.data.tf))
        if ti < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(ti, self.data.ti))
        
        # subdivide load into years
        ts = []
        for yr in list(range(ti.year, tf.year+2)):
            t = np.max([datetime(yr,1,1,0,0,0),ti,self.data.ti+self.dtw])
            t = np.min([t,tf,self.data.tf])
            ts.append(t)
        if ts[-1] == ts[-2]: ts.pop()
        
        # load features one data stream and year at a time
        FM = []
        ys = []
        for ds in self.data_streams:
            fM = []
            ys = []
            for t0,t1 in zip(ts[:-1], ts[1:]):
                fMi,y = self._extract_features(t0,t1,ds)
                fM.append(fMi)
                ys.append(y)
            # vertical concat on time
            FM.append(pd.concat(fM))
        # horizontal concat on column
        FM = pd.concat(FM, axis=1, sort=False)
        ys = pd.concat(ys)
        
        self.ti_prev = ti
        self.tf_prev = tf
        self.fM = FM
        self.ys = ys
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
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
            yr : pd.DataFrame
                Reduced label vector.
        """
        self.exclude_dates = exclude_dates
        if len(self.exclude_dates) != 0:
            for exclude_date_range in self.exclude_dates:
                t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
                inds = (y.index<t0)|(y.index>=t1)
                X = X.loc[inds]
                y = y.loc[inds]
        return X,y
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
    def _model_alerts(self, t, y, threshold, ialert, dti):
        """ Compute issued alerts for model consensus.
            Parameters:
            -----------
            t : array-like
                Time vector corresponding to model consensus.
            y : array-like
                Model consensus.
            threshold : float
                Consensus value above which an alert is issued.
            ialert : int
                Number of data windows spanning an alert period.
            dti : datetime.timedelta
                Length of window overlap.
            Returns:
            --------
            false_alert : int
                Number of falsely issued alerts.
            missed : int
                Number of eruptions for which an alert not issued.
            true_alert : int
                Number of eruptions for which an alert correctly issued.
            true_negative : int
                Equivalent number of time windows in which no alert was issued and no eruption
                occurred. Each time window has the average length of all issued alerts.
            dur : float
                Total alert duration as fraction of model analysis period.
            mcc : float
                Matthews Correlation Coefficient.
        """
        # create contiguous alert windows
        inds = np.where(y>threshold)[0]

        if len(inds) == 0:
            return 0, len(self.data.tes), 0, int(1e8), 0, 0

        dinds = np.where(np.diff(inds)>ialert)[0]
        alert_windows = list(zip(
            [inds[0],]+[inds[i+1] for i in dinds],
            [inds[i]+ialert for i in dinds]+[inds[-1]+ialert]
            ))
        alert_window_lengths = [np.diff(aw) for aw in alert_windows]
        
        # compute true/false positive/negative rates
        tes = copy(self.data.tes)
        nes = len(self.data.tes)
        nalerts = len(alert_windows)
        true_alert = 0
        false_alert = 0
        inalert = 0.
        missed = 0
        total_time = (t[-1] - t[0]).total_seconds()

        for i0,i1 in alert_windows:

            inalert += ((i1-i0)*dti).total_seconds()
            # no eruptions left to classify, only misclassifications now
            if len(tes) == 0:
                false_alert += 1
                continue

            # eruption has been missed
            while tes[0] < t[i0]:
                tes.pop(0)
                missed += 1
                if len(tes) == 0:
                    break
            if len(tes) == 0:
                continue

            # alert does not contain eruption
            if not (tes[0] > t[i0] and tes[0] <= (t[i0] + (i1-i0)*dti)):
                false_alert += 1
                continue

            # alert contains eruption
            while tes[0] > t[i0] and tes[0] <= (t[i0] + (i1-i0)*dti):
                tes.pop(0)
                true_alert += 1
                if len(tes) == 0:
                    break

        # any remaining eruptions after alert windows have cleared must have been missed
        missed += len(tes)
        dur = inalert/total_time
        true_negative = int((len(y)-np.sum(alert_window_lengths))/np.mean(alert_window_lengths))-missed
        mcc = matthews_corrcoef(self._ys, (y>threshold)*1.)

        return false_alert, missed, true_alert, true_negative, dur, mcc
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
        self.drop_features = drop_features
        self.compute_only_features = compute_only_features
        self.n_jobs = n_jobs
        ti = self.ti_model if ti is None else datetimeify(ti)
        tf = self.tf_model if tf is None else datetimeify(tf)
        return self._load_data(ti, tf)
    def train(self, ti=None, tf=None, Nfts=20, Ncl=500, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[], method=0.75):
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
        self.classifier = classifier
        self.exclude_dates = exclude_dates
        self.use_only_features = use_only_features
        self.n_jobs = n_jobs
        self.Ncl = Ncl
        makedir(self.modeldir)

        # initialise training interval
        self.ti_train = self.ti_model if ti is None else datetimeify(ti)
        self.tf_train = self.tf_model if tf is None else datetimeify(tf)
        if self.ti_train - self.dtw < self.data.ti:
            self.ti_train = self.data.ti+self.dtw
        
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
        fM, ys = self._load_data(self.ti_train, self.tf_train)

        # manually drop features (columns)
        fM = self._drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features = [df for df in self.use_only_features if df in fM.columns]
            fM = fM[use_only_features]
            Nfts = len(use_only_features)+1

        # manually drop windows (rows)
        fM, ys = self._exclude_dates(fM, ys, exclude_dates)
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")
        
        # select training subset
        inds = (ys.index>=self.ti_train)&(ys.index<self.tf_train)
        fM = fM.loc[inds]
        ys = ys['label'].loc[inds]

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        f = partial(train_one_model, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed, method)

        # train models with glorious progress bar
        # f(0)
        for i, _ in enumerate(mapper(f, range(Ncl))):
            cf = (i+1)/Ncl
            print(f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
        if self.n_jobs > 1:
            p.close()
            p.join()
        
        # free memory
        del fM
        gc.collect()
        self._collect_features()
    def forecast(self, ti=None, tf=None, recalculate=False, use_model=None, n_jobs=None, yr=None):
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
        ti=datetimeify(ti)
        tf=datetimeify(tf)
        # special case of high resolution forecast where multiple feature matrices exist
        if yr is None: 
            forecast = []
            fr = copy(self.feature_root)

            # use hires feature matrices for each year
            for yr in list(range(ti.year, tf.year+1)):
                t0 = np.max([datetime(yr,1,1,0,0,0),ti,self.data.ti+self.dtw])
                t1 = np.min([datetime(yr+1,1,1,0,0,0),tf,self.data.tf])
                forecast_i = self.forecast(t0,t1,recalculate,use_model,n_jobs,yr)    
                forecast.append(forecast_i)

            # merge the individual forecasts and ensure that original limits are respected
            forecast = pd.concat(forecast, sort=False)
            return forecast[(forecast.index>=ti)&(forecast.index<=tf)]

        self._use_model = use_model
        makedir(self.preddir)
        yr_str = '_{:d}'.format(yr) if yr is not None else ''
        confl = '{:s}/consensus{:s}'.format(self.preddir,'{:s}.{:s}'.format(yr_str, self.savefile_type))
                #
        if n_jobs is not None: 
            self.n_jobs = n_jobs 

        self.ti_forecast = self.ti_model if ti is None else datetimeify(ti)
        self.tf_forecast = self.tf_model if tf is None else datetimeify(tf)
        if self.tf_forecast > self.data.tf:
            self.tf_forecast = self.data.tf
        if self.ti_forecast - self.dtw < self.data.ti:
            self.ti_forecast = self.data.ti+self.dtw

        model_path = self.modeldir + os.sep
        if use_model is not None:
            self._detect_model()
            model_path = self._use_model+os.sep
            
        model = get_classifier(self.classifier)[0]

        # logic to determine which models need to be run and which to be 
        # read from disk
        pref = type(model).__name__
        models = glob('{:s}/{:s}_*.pkl'.format(model_path, pref))
        run_predictions = []
        ys = []        
        tis = []

        # create a prediction for each model
        for model in models:
            # change location
            pred = model.replace(model_path, self.preddir+os.sep)
            # update filetype
            pred = pred.replace('.pkl','{:s}.{:s}'.format(yr_str, self.savefile_type))                

            # check if prediction already exists
            if os.path.isfile(pred):
                if recalculate:
                    # delete predictions to be recalculated
                    os.remove(pred)
                    run_predictions.append([model, pred])  
                    tis.append(self.ti_forecast)
                else:
                    # load an existing prediction
                    y = load_dataframe(pred, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                    # check if prediction spans the requested interval
                    if y.index[-1] < self.tf_forecast:
                        run_predictions.append([model, pred])
                        tis.append(y.index[-1])
                    else:
                        ys.append(y)
            else:
                run_predictions.append([model, pred])  
                tis.append(self.ti_forecast)
        
        if len(tis)>0:
            ti = np.min(tis)

        # generate new predictions
        if len(run_predictions)>0:
            # load feature matrix
            fM,_ = self._load_data(ti, self.tf_forecast)
            fM = fM.fillna(1.e-8)
            if fM.shape[0] == 0: return pd.DataFrame([],columns=['consensus'])

            # # setup predictor
            # if self.n_jobs > 1:
            #     p = Pool(self.n_jobs)
            #     mapper = p.imap
            # else:
            #     ys = predict_models(fM, model_path, run_predictions)
            # f = partial(predict_one_model, fM, model_path)

            # run models with glorious progress bar
            #f(run_predictions[0])
            # predict_models(fM, model_path, run_predictions)
            # not parallelized for now
            ys += predict_models(fM, model_path, run_predictions, yr)
            # if False:
            #     for i, y in enumerate(mapper(f, run_predictions)):
            #         cf = (i+1)/len(run_predictions)
            #         if yr is None:
            #             print(f'forecasting: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
            #         else:
            #             print(f'forecasting {yr:d}: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
            #         ys.append(y)
            # else:
            #     ys = p.imap(f, run_predictions)
            
            # if self.n_jobs > 1:
            #     p.close()
            #     p.join()
        
        # condense data frames and write output
        ys = pd.concat(ys, axis=1, sort=False)
        consensus = np.mean([ys[col].values for col in ys.columns if 'pred' in col], axis=0)
        forecast = pd.DataFrame(consensus, columns=['consensus'], index=ys.index)

        save_dataframe(forecast, confl, index=True, index_label='time')
        
        # memory management
        if len(run_predictions)>0:
            del fM
            gc.collect()
            
        return forecast
    def hires_forecast(self, ti, tf, recalculate=True, save=None, root=None, nztimezone=False, 
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
            _ = self.ti_train
        except AttributeError:
            raise ValueError('Train model before constructing hires forecast.')
        
        if save == '':
            save = '{:s}/hires_forecast.png'.format(self.plotdir)
            makedir(self.plotdir)
        
        if n_jobs is not None: self.n_jobs = n_jobs
 
        # calculate hires feature matrix
        if root is None:
            root = self.root+'_hires'
        _fm = ForecastModel(self.window, 1., self.look_forward, station=self.station, ti=ti, tf=tf, 
            data_streams=self.data_streams, root=root, savefile_type=self.savefile_type, feature_root=root,
            feature_dir=self.featdir, data_dir=self.data_dir)
        _fm.compute_only_features = list(set([ft.split('__')[1] for ft in self._collect_features()[0]]))
        
        # predict on hires features
        ys = _fm.forecast(ti, tf, recalculate, use_model=self.modeldir, n_jobs=n_jobs)
        
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
        ci = 1.96*(np.sqrt(y*(1-y)/self.Ncl))
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
        makedir(self.plotdir)
        if save is None:
            save = '{:s}/forecast.png'.format(self.plotdir)
        # set up figures and axes
        f = plt.figure(figsize=(24,15))
        N = 10
        dy1,dy2 = 0.05, 0.05
        dy3 = (1.-dy1-(N//2)*dy2)/(N//2)
        dx1,dx2 = 0.37,0.04
        axs = [plt.axes([0.10+(1-i//(N/2))*(dx1+dx2), dy1+(i%(N/2))*(dy2+dy3), dx1, dy3]) for i in range(N)][::-1]
        
        for i,ax in enumerate(axs[:-1]):
            ti,tf = [datetime.strptime('{:d}-01-01 00:00:00'.format(2011+i), '%Y-%m-%d %H:%M:%S'),
                datetime.strptime('{:d}-01-01 00:00:00'.format(2012+i), '%Y-%m-%d %H:%M:%S')]
            ax.set_xlim([ti,tf])
            ax.text(0.01,0.95,'{:4d}'.format(2011+i), transform=ax.transAxes, va='top', ha='left', size=16)
            
        ti,tf = [datetimeify(x) for x in xlim]
        axs[-1].set_xlim([ti, tf])
        
        # model forecast is generated for the END of each data window
        t = ys.index

        # average individual model responses
        ys = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
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
                t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
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
                i = (tii.year-2011)
                axs[i].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                j = (tii+self.dtf).year - 2011
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
        
        makedir(self.plotdir)
        # set up figures and axes
        f = plt.figure(figsize=(8,4))
        ax = plt.axes([0.1, 0.08, 0.8, 0.8])
        t = pd.to_datetime(ys.index.values)
        if True: # plot filtered data
            if 'zsc_rsamF' in self.data_streams and 'rsamF' not in self.data_streams:
                rsam = self.data.get_data(t[0], t[-1])['zsc_rsamF']
            else: 
                rsam = self.data.get_data(t[0], t[-1])['rsamF']
        else: 
            if 'zsc_rsam' in self.data_streams and 'rsam' not in self.data_streams:
                rsam = self.data.get_data(t[0], t[-1])['zsc_rsam']
            else: 
                rsam = self.data.get_data(t[0], t[-1])['rsam']
        trsam = rsam.index
        if nztimezone:
            t = to_nztimezone(t)
            trsam = to_nztimezone(trsam)
            ax.set_xlabel('Local time')
        else:
            ax.set_xlabel('UTC')
        y = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
                
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0,0.25,0.50,0.75,1.00])
        ax.set_ylabel('ensemble mean')
    
        # consensus threshold
        ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

        # modelled alert
        ax.plot(t, y, 'c-', label='ensemble mean', zorder=4, lw=0.75)
        ci = self._compute_CI(y)
        ax.fill_between(t, (y-ci), (y+ci), color='c', zorder=5, alpha=0.3)
        ax_ = ax.twinx()
        ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
        ax_.set_ylim([0,5])
        # ax_.set_xlim(ax.get_xlim())
        ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.75)

        for tii,yi in zip(t, y):
            if yi > threshold:
                ax.fill_between([tii, tii+self.dtf], [0,0], [100,100], color='y', zorder=3)

        for te in self.data.tes:
            ax.axvline(te, color = 'r', linestyle='--', zorder=10)    
        ax.plot([],[], 'r--', label='eruption')    
        ax.fill_between([], [], [], color='y', label='eruption forecast')
        ax.plot([],[],'k-', lw=0.75, label='RSAM')

        ax.legend(loc=2, ncol=2)

        tmax = np.max([t[-1], trsam[-1]])
        tmin = np.min([t[0], trsam[0]])
        if xlim is None:
            xlim = [tmin,tmax]
        tmax = xlim[-1] 
        tf = tmax 
        t0 = tf.replace(hour=0, minute=0, second=0)
        dt = (tmax-tmin).total_seconds()
        if dt < 10.*24*3600:
            ndays = int(np.ceil(dt/(24*3600)))
            xts = [t0 - timedelta(days=i) for i in range(ndays)][::-1]
            lxts = [xt.strftime('%d %b') for xt in xts]
        elif dt < 20.*24*3600:
            ndays = int(np.ceil(dt/(24*3600))/2)
            xts = [t0 - timedelta(days=2*i) for i in range(ndays)][::-1]
            lxts = [xt.strftime('%d %b') for xt in xts]
        elif dt < 70.*24*3600:
            ndays = int(np.ceil(dt/(24*3600))/7)
            xts = [t0 - timedelta(days=7*i) for i in range(ndays)][::-1]
            lxts = [xt.strftime('%d %b') for xt in xts]
        elif dt < 365.25*24*3600:
            t0 = tf.replace(day=1, hour=0, minute=0, second=0)
            nmonths = int(np.ceil(dt/(24*3600*365.25/12)))
            xts = [t0 - timedelta(days=i*365.25/12) for i in range(nmonths)][::-1]
            lxts = [xt.strftime('%b') for xt in xts]
        elif dt < 2*365.25*24*3600:
            t0 = tf.replace(day=1, hour=0, minute=0, second=0)
            nmonths = int(np.ceil(dt/(24*3600*365.25/12))/2)
            xts = [t0 - timedelta(days=2*i*365.25/12) for i in range(nmonths)][::-1]
            lxts = [xt.strftime('%b %Y') for xt in xts]
        ax.set_xticks(xts)
        ax.set_xticklabels(lxts)
        
        ax.set_xlim(xlim)
        ax_.set_xlim(xlim)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.85, 0.95, self.data.station +' '+ ys.index[-1].strftime('%Y'), size = 12, ha = 'left', va = 'top', transform=ax.transAxes, bbox=props)
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
        makedir(self.preddir)
        label_file = self.preddir+'/labels.pkl'
        if not os.path.isfile(label_file):
            ys = np.array([self.data._is_eruption_in(days=self.look_forward, from_time=ti) for ti in pd.to_datetime(t)])
            save_dataframe(ys, label_file)
        self._ys = load_dataframe(label_file)

        if ialert is None:
            ialert = self.look_forward/((1-self.overlap)*self.window)
        if dti is None:
            dti = timedelta(days=(1-self.overlap)*self.window)
        FP, FN, TP, TN, dur, MCC=[np.zeros(len(thresholds)) for i in range(6)]
        for j,threshold in enumerate(thresholds):
            if threshold == 0:
                FP[j]=int(1e8); dur[j]=1.; TP[j]=len(self.data.tes); TN[j]=1
            else:
                FP[j], FN[j], TP[j], TN[j], dur[j], MCC[j] = self._model_alerts(t, y, threshold, ialert, dti)

        return FP, FN, TP, TN, dur, MCC

class ForecastTransLearn(object):
    ''' Object for running forecast models.  
        
        Constructor arguments:
        ----------------------
        window : float
            Length of data window in days.
        overlap : float
            Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
        look_forward : float
            Length of look-forward in days.
        stations_train : str
            Seismic stations providing data for training.
        station_test : str
            Seismic station providing data for testing.
        datastreams : list
            Data streams and transforms from which to extract features. Options are 'X', 'diff_X', 'log_X', 'inv_X', and 'stft_X' 
            where X is one of 'rsam', 'mf', 'hf', or 'dsar'. 
        feature_root : str
            Root for feature naming.
        feature_dir : str
            Directory to save feature matrices.
        root : str 
            Naming convention for train file. If not given, will default to 'fm_*Tw*wndw_*eta*ovlp_*Tlf*lkfd_*ds*' where
            Tw is the window length, eta is overlap fraction, Tlf is look-forward and ds are data streams.
        savefile_type : str
            Extension denoting file format for save/load. Options are csv, pkl (Python pickle) or hdf.

        Attributes:
        -----------
        data : TremorData
            Object containing tremor data of testing station
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
        ti_forecast : datetime.datetime
            Beginning of model forecast period (in testing station)
        tf_forecast : datetime.datetime
            End of model forecast period (in testing station)
        n_jobs : int
            Number of CPUs to use for parallel tasks.
        rootdir : str
            Repository location on file system.
        plotdir : str
            Directory to save forecast plots.
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
    '''
    def __init__(self, model_name,rootdir=None,root=None,modeldir=None,
        featdir=None,datadir=None,predicdir=None,plotdir=None,savefile_type='pkl'):
        # load attributes from models meta
        with open(rootdir+os.sep+'models'+os.sep+model_name+os.sep+'meta.txt') as f:
            lines = f.readlines()
            _dic={}
            for line in lines:
                line=line.strip()
                line=line.split('\t')
                _dic[line[0]]=line[1]
        # load 
        self.stations_train = _dic['stations']
        self.data_streams = _dic['datastreams'].split(',')
        self.window = float(_dic['window'])
        self.overlap = float(_dic['overlap'])
        self.dtb = int(_dic['dtb'])
        self.dtf = int(_dic['dtf'])
        self.lab_lb = float(_dic['lab_lb'])
        self.classifier = _dic['classifier']
        self.Ncl = int(_dic['Ncl'])
        self.Nfts = int(_dic['Nfts'])
        self.method = float(_dic['method'])
        self.feat_selc = _dic['features'].split(',')
        #
        self.look_forward = self.window
        #
        self.dtw = timedelta(days=self.window)
        self.dt = timedelta(seconds=600)
        self.dto = (1.-self.overlap)*self.dtw
        self.iw = int(self.window*6*24)         
        self.io = int(self.overlap*self.iw)      
        if self.io == self.iw: self.io -= 1
        self.window = self.iw*1./(6*24)
        #
        self.n_jobs = 6

        # naming convention and file system attributes
        self.savefile_type = savefile_type
        if root is None:
            self.root = 'fm_{:3.2f}wndw_{:3.2f}ovlp_{:3.2f}lkfd'.format(self.window, self.overlap, self.look_forward)
            self.root += '_'+((('{:s}-')*len(self.data_streams))[:-1]).format(*sorted(self.data_streams))
        else:
            self.root = root
        self.rootdir = '/'.join(getfile(currentframe()).split(os.sep)[:-2])
        #self.plotdir = r'{:s}/plots/{:s}'.format(self.rootdir, self.root)
        if modeldir is None:
            self.modeldir = r'{:s}/models/{:s}'.format(self.rootdir, self.root)
        else:
            self.modeldir = modeldir
        if featdir is None:
            self.featdir = r'{:s}/features'.format(self.rootdir)
        else:
            self.featdir = featdir
        if datadir is None:
            self.datadir = r'{:s}/data'.format(self.rootdir)
        else:
            self.datadir = datadir
        self.featfile = lambda ds,yr: (r'{:s}/fm_{:3.2f}w_{:s}_{:s}_{:d}.{:s}'.format(self.featdir,self.window,ds,self.station,yr,self.savefile_type))
        if predicdir is None:
            self.predicdir = r'{:s}/predictions'.format(self.rootdir)
        else:
            self.predicdir = predicdir
        if plotdir is None:
            self.plotdir = r'{:s}/plots'.format(self.rootdir)
        else:
            self.plotdir = plotdir
        #
    def _load_models(self):
        '''read model
        '''
        # logic to determine which models need to be run and which to be 
        # read from disk
        model = get_classifier(self.classifier)[0]
        pref = type(model).__name__
        models = glob('{:s}'.format(self.modeldir)+os.sep+'{:s}'.format(self.root)+os.sep+'{:s}'.format(pref)+'_*.pkl')
        return models
    def _load_feat_pred(self, ti=None, tf=None):
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

        for i,datastream in enumerate(self.data_streams):
            fl_nm='FM_'+str(int(self.window))+'w_'+datastream+'_'+self.station_test+'_'+str(self.dtb)+'dtb_'+str(self.dtf)+'dtf'+'.csv'
            if os.path.isfile(os.sep.join([self.featdir ,fl_nm])):
                # load feature matrix
                FM.append(load_dataframe(os.sep.join([self.featdir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None))
            else: 
                #
                feat_selc=[ft for ft in self.feat_selc if datastream in ft]
                print('Creating feature matrix:'+fl_nm+'\n . Will be saved in: '+self.featdir)
                feat_stas = FeaturesMulti(stations=[self.station_test], window = self.window, datastream = datastream, feat_dir=self.featdir, 
                    dtb=self.dtb, dtf=self.dtf, lab_lb=self.lab_lb,tes_dir=self.datadir, 
                        noise_mirror=None,data_dir=self.datadir, dt=10,feat_selc=feat_selc)
                feat_stas.save()#fl_nm=fl_nm)
                # load feature matrix
                FM.append(load_dataframe(os.sep.join([self.featdir,fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None))
                del feat_stas
        # currently just importing features around eruptions 
        # horizontal concat on column
        FM = pd.concat(FM, axis=1, sort=False)
        # load labels 
        _=fl_nm.find('.')
        _fl_nm=fl_nm[:_]+'_labels'+fl_nm[_:]
        YS = load_dataframe(os.sep.join([self.featdir,_fl_nm]), index_col=0, parse_dates=False, infer_datetime_format=False, header=0, skiprows=None, nrows=None)
        YS['time'] = pd.to_datetime(YS['time'])
        #
        return FM, YS
    def forecast(self, station_test, ti_forecast=None, tf_forecast=None, use_model=None, recalculate=False,  n_jobs=None, yr=None):
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
        #
        #os.makedir(self.predicdir+os.sep+self.root) if not os.path.isdir(self.predicdir+os.sep+self.root) else pass
        self.station_test=station_test
        _=self.root.split('_')
        _.insert(6,self.station_test)
        self.root_pred=('_').join(_)
        if os.path.isdir(self.predicdir+os.sep+self.root_pred):
            pass
        else:
            makedir(self.predicdir+os.sep+self.root_pred)
        #
        #self.data = TremorData(self.station_test, parent=self, data_dir=self.datadir)
        self.data = TremorData(self.station_test, parent=self, data_dir=self.datadir)
        # if any(['_' in ds for ds in self.data_streams]):
        #     self.data._compute_transforms()
        # if any([d not in self.data.df.columns for d in self.data_streams]):
        #     raise ValueError("data restricted to any of {}".format(self.data.df.columns))
        #
        ti=datetimeify(ti_forecast)
        tf=datetimeify(tf_forecast)
        self.ti_forecast=datetimeify(ti_forecast)
        self.tf_forecast=datetimeify(tf_forecast)
        # special case of high resolution forecast where multiple feature matrices exist
        # if yr is None: 
        #     forecast = []
        #     fr = copy(self.featdir)
        #     # use hires feature matrices for each year
        #     for yr in list(range(ti.year, tf.year+1)):
        #         a=_data.ti
        #         t0 = np.max([datetime(yr,1,1,0,0,0),ti,self.data.ti+self.dtw])
        #         t1 = np.min([datetime(yr+1,1,1,0,0,0),tf,self.data.tf])
        #         forecast_i = self.forecast(t0,t1,recalculate,use_model,n_jobs,yr)    
        #         forecast.append(forecast_i)

        #     # merge the individual forecasts and ensure that original limits are respected
        #     forecast = pd.concat(forecast, sort=False)
        #     return forecast[(forecast.index>=ti)&(forecast.index<=tf)]

        model_path = self.modeldir + os.sep + self.root +os.sep
        # if use_model is not None:
        #     self._detect_model()
        #     model_path = self._use_model+os.sep

        models = self._load_models()
        yr_str = '_{:d}'.format(yr) if yr is not None else ''
        if n_jobs is not None: 
            self.n_jobs = n_jobs 
        confl = '{:s}/consensus{:s}'.format(self.predicdir,'{:s}.{:s}'.format(yr_str, self.savefile_type))
        # self.ti_forecast = self.ti_model if ti is None else datetimeify(ti)
        # self.tf_forecast = self.tf_model if tf is None else datetimeify(tf)
        # if self.tf_forecast > self.data.tf:
        #     self.tf_forecast = self.data.tf
        # if self.ti_forecast - self.dtw < self.data.ti:
        #     self.ti_forecast = self.data.ti+self.dtw
        #
        run_predictions = []
        ys = [] 
        _ys = []        
        tis = []
        # create a prediction for each model
        for model in models:
            # change location
            #pred = model.replace(model_path, self.preddir+os.sep)
            pred = model.replace(self.modeldir, self.predicdir)
            pred = pred.replace(self.root, self.root_pred)
            # update filetype
            pred = pred.replace('.pkl','{:s}.{:s}'.format(yr_str, self.savefile_type))                

            # check if prediction already exists
            if False:#os.path.isfile(pred):
                if recalculate:
                    # delete predictions to be recalculated
                    os.remove(pred)
                    run_predictions.append([model, pred])  
                    tis.append(self.ti_forecast)
                else:
                    # load an existing prediction
                    y = load_dataframe(pred, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                    # check if prediction spans the requested interval
                    if y.index[-1] < self.tf_forecast:
                        run_predictions.append([model, pred])
                        tis.append(y.index[-1])
                    else:
                        ys.append(y)
            else:
                run_predictions.append([model, pred])  
                tis.append(self.ti_forecast)

        # generate new predictions
        if len(run_predictions)>0:
            # load feature matrix
            fM,_ = self._load_feat_pred(self.ti_forecast, self.tf_forecast)
            #
            fM = fM.fillna(1.e-8)
            if fM.shape[0] == 0: return pd.DataFrame([],columns=['consensus'])

            # # setup predictor
            # if self.n_jobs > 1:
            #     p = Pool(self.n_jobs)
            #     mapper = p.imap
            # else:
            #     ys = predict_models(fM, model_path, run_predictions)
            # f = partial(predict_one_model, fM, model_path)

            # run models with glorious progress bar
            #f(run_predictions[0])
            # predict_models(fM, model_path, run_predictions)
            # not parallelized for now
            ys += predict_models(fM, model_path, run_predictions, yr)
            _ys.append(_)
            # if False:
            #     for i, y in enumerate(mapper(f, run_predictions)):
            #         cf = (i+1)/len(run_predictions)
            #         if yr is None:
            #             print(f'forecasting: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
            #         else:
            #             print(f'forecasting {yr:d}: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
            #         ys.append(y)
            # else:
            #     ys = p.imap(f, run_predictions)
            
            # if self.n_jobs > 1:
            #     p.close()
            #     p.join()
        
        # condense data frames and write output
        ys = pd.concat(ys, axis=1, sort=False)
        _ys = pd.concat(_ys, axis=1, sort=False)
        consensus = np.mean([ys[col].values for col in ys.columns if 'pred' in col], axis=0)
        forecast = pd.DataFrame(consensus, columns=['consensus'], index=_ys['time'])
        #
        #forecast['time']=_ys['time']
        #forecast.rename(columns={'time': 'ref'}, inplace=True)
        #forecast.set_index('time')
        #
        save_dataframe(forecast, confl, index=True, index_label='time')
        
        # memory management
        if len(run_predictions)>0:
            del fM
            gc.collect()

        return forecast
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
        # check directories
        if os.path.isdir(self.plotdir):
            pass
        else:
            makedir(self.plotdir)
        if os.path.isdir(self.plotdir+os.sep+self.root_pred):
            pass
        else:
            makedir(self.plotdir+os.sep+self.root_pred)
        #
        self.plotdir=self.plotdir+os.sep+self.root_pred
        #
        if save is None:
            save = '{:s}/forecast.png'.format(self.plotdir)
        # set up figures and axes
        f = plt.figure(figsize=(24,15))
        N = 10
        dy1,dy2 = 0.05, 0.05
        dy3 = (1.-dy1-(N//2)*dy2)/(N//2)
        dx1,dx2 = 0.37,0.04
        axs = [plt.axes([0.10+(1-i//(N/2))*(dx1+dx2), dy1+(i%(N/2))*(dy2+dy3), dx1, dy3]) for i in range(N)][::-1]
        
        for i,ax in enumerate(axs[:-1]):
            ti,tf = [datetime.strptime('{:d}-01-01 00:00:00'.format(2011+i), '%Y-%m-%d %H:%M:%S'),
                datetime.strptime('{:d}-01-01 00:00:00'.format(2012+i), '%Y-%m-%d %H:%M:%S')]
            ax.set_xlim([ti,tf])
            ax.text(0.01,0.95,'{:4d}'.format(2011+i), transform=ax.transAxes, va='top', ha='left', size=16)
            
        ti,tf = [datetimeify(x) for x in xlim]
        axs[-1].set_xlim([ti, tf])
        
        # model forecast is generated for the END of each data window
        t = ys.index

        # average individual model responses
        ys = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
        for i,ax in enumerate(axs):

            ax.set_ylim([-0.05, 1.05])
            ax.set_yticks([0,0.25,0.5, 0.75, 1.0])
            if i//(N/2) == 0:
                ax.set_ylabel('alert level')
            else:
                ax.set_yticklabels([])

            # shade training data
            # ax.fill_between([self.ti_train, self.tf_train],[-0.05,-0.05],[1.05,1.05], color=[0.85,1,0.85], zorder=1, label='training data')            
            # for exclude_date_range in self.exclude_dates:
            #     t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
            #     ax.fill_between([t0, t1],[-0.05,-0.05],[1.05,1.05], color=[1,1,1], zorder=2)            
            
            # consensus threshold
            ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

            # modelled alert
            ax.plot(t, ys, 'c-', label='modelled alert', zorder=4)

            # eruptions
            for te in self.data.tes:
                ax.axvline(te, color='k', linestyle='-', zorder=5)
            ax.axvline(te, color='k', linestyle='-', label='eruption')

        # for tii,yi in zip(t, ys):
        #     if yi > threshold:
        #         i = (tii.year-2011)
        #         axs[i].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
        #         j = (tii+self.dtf).year - 2011
        #         if j != i:
        #             axs[j].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                
        #         if tii > ti:
        #             axs[-1].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                
        for ax in axs:
            ax.fill_between([], [], [], color='y', label='eruption forecast')
        axs[-1].legend()
        
        plt.savefig(save, dpi=400)
        #
        if False:
            # Save just the portion _inside_ the second axis's boundaries
            import matplotlib as mpl
            extent = axs[-1].get_window_extent().transformed(mpl.dpi_scale_trans.inverted())
            plt.savefig(save, bbox_inches=extent)
        #
        plt.close(f)
    pass

# testing
if __name__ == "__main__":
    datadir=r'U:\Research\EruptionForecasting\eruptions\data' 
    featdir=r'U:\Research\EruptionForecasting\eruptions\features'
    modeldir=r'U:\Research\EruptionForecasting\eruptions\models'
    predicdir=r'U:\Research\EruptionForecasting\eruptions\predictions'
    plotdir=r'U:\Research\EruptionForecasting\eruptions\plots'
    
    fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts'
    #
    if False: # ForecastModel class
        #
        n_jobs=0
        data_streams = ['zsc2_rsamF']#,'zsc2_dsarF','zsc2_hfF','zsc2_mfF']
        fm = ForecastModel(station = 'WIZ', ti='2019-11-01', tf='2019-12-31', window=2., overlap=0.75, 
            look_forward=2., data_streams=data_streams, root='test', feature_dir=featdir, 
                data_dir=datadir,savefile_type='csv')
        # drop features 
        drop_features = ['linear_trend_timewise','agg_linear_trend']
        drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
            '*attr_"angle"*']  
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
        # train
        te = fm.data.tes[-1]
        fm.train(ti='2019-11-01', tf='2019-12-31', drop_features=drop_features, exclude_dates=[[te-month/2,te+month/2],], 
            retrain=True, n_jobs=n_jobs, Nfts=5, Ncl=5) #  use_only_features=use_only_features, exclude_dates=[[te-month,te+month],]
        # forecast
        ys = fm.forecast(ti='2012-01-01', tf='2019-12-31', recalculate=True, n_jobs=n_jobs)    
        # plot
        fm.plot_forecast(ys, threshold=0.75, xlim = [te-month/4., te+month/15.], 
            save=r'{:s}/forecast.png'.format(fm.plotdir))
        pass
    
    if True: # ForecastTransLearn class
        #
        datastream = ['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF']#['zsc_rsamF','zsc_mfF','zsc_hfF','zsc_dsarF', 'log_zsc2_rsamF', 'diff_zsc2_rsamF']
        stations=['WIZ','KRVZ']
        dtb = 30
        dtf = 0
        win=2.
        #
        # load feature matrices for WIZ and FWVZ
        #rootdir='/'.join(getfile(currentframe()).split(os.sep)[:-2])
        if True:
            rootdir=r'U:\Research\EruptionForecasting\eruptions'
        root='FM_'+str(int(win))+'w_'+'-'.join(datastream)+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'
        # model to be use
        model_name=root
        fm0 = ForecastTransLearn(model_name,rootdir=rootdir,root=root,datadir=datadir,
            modeldir=modeldir,featdir=featdir,predicdir=predicdir, plotdir=plotdir, savefile_type='csv') # 
        # run forec
        station_test='FWVZ'
        ti_forecast='2006-07-01'
        tf_forecast='2009-12-31'#'2012-12-31'
        ys = fm0.forecast(station_test=station_test,ti_forecast=ti_forecast, tf_forecast=tf_forecast, 
            yr=2012, recalculate=False)
        fm0.plot_forecast(ys, threshold=0.75, xlim = [ti_forecast, tf_forecast])




