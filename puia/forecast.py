"""Forecast objects and performance metrics."""

__author__ = """Alberto Ardid, David Dempsey"""
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
from .utilities import datetimeify, load_dataframe, save_dataframe
# from .data import SeismicData
# from .features import FeaturesSta, FeaturesMulti

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

Todo:
- format to save dataframe with metadata
- add a method to forecast accuracy 
= methods to implement: get_performance, _compute_CI, plot_performance
'''
# Forecast class
class Forecast(object):
    def __init__(self, y, y0, iy, ilf, tes, tl):
        '''
            y - forecast
            y0 - label
            iy - index
            ilf - number of indices in look forward period
            tes - list of eruption dates
            tl - training labels (0=out-of-sample, 1=pseudo out-of-sample, 2=in-sample)
        '''
        self.df=pd.DataFrame(zip(y,y0,tl), columns=['consensus','label','training label'], index=iy)
        self.ilf=ilf
        tes=[pd.Timestamp(te) for te in tes]
        self.tes=[te for te in tes if iy[0]<te<=iy[-1]]        # eruption dates
    def save(self, fl):
        ''' save the forecast

            fl - save file name
        '''
        save_dataframe(self, fl, index=True, index_label='time')
    def clip(self, ti=None, tf=None):
        ''' clip date range of forecast

            ti - starting time
            tf - ending time
        '''
        if ti is None:
            ti=self.iy[0]
        if tf is None:
            tf=self.iy[-1]
        self.df=self.df[(self.iy>=ti)&(self.iy<=tf)]
        self.tes=[te for te in self.tes if ti<te<=tf]   
    def alert_model(self, threshold):      
        ''' return a binary warning model for the forecast

            threshold - forecast value at which alert period starts

            returns AlertModel
        '''
        # an alert model object
        am=AlertModel(self.y, self.iy, self.ilf, threshold)

        # create contiguous alert windows
        inds=np.where(self.y>threshold)[0]
        if len(inds) == 0:
            am.false_negative=len(self.tes)
            am.inalert=0.
            return am

        # create contiguous self-extending alert windows
        dinds=np.where(np.diff(inds)>self.ilf)[0]
        am.alert_window_indices=list(zip(
            [inds[0],]+[inds[i+1] for i in dinds],
            [inds[i]+self.ilf for i in dinds]+[inds[-1]+self.ilf]
            ))
        
        # alerts cannot extend beyond the end of the forecast period 
        if am.alert_window_indices[-1][-1]>=self.y.shape[0]:
            am.alert_window_indices[-1]=[am.alert_window_indices[-1][0],self.y.shape[0]-1]

        # convert indexed windows to dates and durations
        am.alert_windows=[[self.iy[i0],self.iy[i1]] for i0,i1 in am.alert_window_indices]
        am.alert_durations=[np.diff(aw) for aw in am.alert_windows]
        
        # compute true/false positive/negative rates
        tes=copy(self.tes)
        am.inalert=0.

        for t0,t1 in am.alert_windows:
            am.inalert+=(t1-t0).total_seconds()/(24*3600)
            
            if len(tes)==0:
                # no eruptions left to classify, only misclassifications here
                am.false_positive+=1
                continue

            while tes[0]<t0:
                # an eruption has been missed (false_negative)
                tes.pop(0)
                am.false_negative+=1
                if len(tes)==0:
                    break
            if len(tes)==0:
                continue

            if not (t0<tes[0]<=t1):
                # alert does not contain eruption
                am.false_positive+=1
                continue

            # alert contains eruption
            while t0<tes[0]<=t1:
                tes.pop(0)
                am.true_positive+=1
                if len(tes) == 0:
                    break

        # any remaining eruptions after alert windows have cleared must have been missed
        am.false_negative += len(tes)
        am.fraction_inalert=am.inalert/am.total_time
        
        return am
    def roc(self, thresholds=None, save=None):
        if thresholds is None:
            thresholds=np.linspace(0,1,101)[::-1]
        else:
            thresholds=np.sort(thresholds)[::-1]
        return ROC(self, thresholds)
    def _get_y(self):
        return self.df['consensus'].values
    y=property(_get_y)
    def _get_y0(self):
        return self.df['label'].values
    y0=property(_get_y0)
    def _get_iy(self):
        return self.df.index
    iy=property(_get_iy)
    def _get_tl(self):
        return self.df['training label'].values
    tl=property(_get_tl)
class MultiVolcanoForecast(object):
    def __init__(self, forecasts):
        self.forecasts=forecasts
    def roc(self, thresholds=None, save=None):
        if thresholds is None:
            thresholds=np.linspace(0,1,101)[::-1]
        else:
            thresholds=np.sort(thresholds)[::-1]
        self.rocs=[ROC(forecast, thresholds) for forecast in self.forecasts]
        return MultiVolcanoROC(self.rocs)        

class AlertModel(object):
    def __init__(self, y, iy, ilf, threshold):        
        self.y=y
        self.iy=iy
        self.ilf=ilf
        self.threshold=threshold
        self.false_positive=0
        self.false_negative=0
        self.true_positive=0
        self.fraction_inalert=0.
        self.total_time=(self.iy[-1]-self.iy[0]).total_seconds()/(24*3600)
    def isalert(self, t):
        ''' return whether time t falls within an alert

            t - datetime 

            returns bool
        '''
        dt=self.iy[1]-self.iy[0]
        inds=np.where((self.iy<t)&(self.iy>(t-self.ilf*dt)))
        if any(self.y[inds]>=self.threshold):
            return True
        else:
            return False
class ROC(object):
    def __init__(self, fcst, thresholds):
        self.thresholds=thresholds
        self._setup(fcst)
    def _setup(self, fcst):
        self.alert_models=[fcst.alert_model(th) for th in self.thresholds]
        atts=['fraction_inalert','inalert','true_positive','false_positive','false_negative']
        for att in atts:
            arr=[am.__getattribute__(att) for am in self.alert_models]
            arr=np.array(arr)
            self.__setattr__(att, arr)
        self.number_eruptions=len(fcst.tes)
        self.auc=np.trapz(self.true_positive/self.number_eruptions, self.fraction_inalert)
        self.forecast_days=(fcst.iy[-1]-fcst.iy[0])/_DAY
    def plot(self, save=None, reference=None):
        ''' plot ROC curve

            save - file name to save output
            reference - other ROC curve(s) to plot. Can be single ROC object, or dictionary of ROC objects
        '''
        f,ax=plt.subplots(1,1)
        ax.plot(self.fraction_inalert*100, self.true_positive/self.number_eruptions*100, 'k-', label=f'ROC ({1.-self.auc})')
        ax.set_xlabel('time in warning [%]')
        ax.set_ylabel('eruptions in warning [%]')

        if reference is not None:
            if type(reference) is ROC:
                ax.plot(reference.fraction_inalert*100, reference.true_positive/reference.number_eruptions*100, 'k:', 
                        label=f'reference ({1.-reference.auc})')
            if type(reference) is dict:                
                for k,c in zip(reference.keys(),['b','g','r','c','y','m']):
                    v=reference[k]
                    ax.plot(v.fraction_inalert*100, v.true_positive/v.number_eruptions*100, c+'-', label=k+f' ({1.-v.auc})')
            ax.legend()
        ax.set_xscale('log')

        if save is not None:
            plt.savefig(save, dpi=400)
        else:            
            plt.show()        
class MultiVolcanoROC(object):
    def __init__(self, rocs):
        self.rocs=rocs
    def plot(self, save=None, reference=None, xscale='log'):
        f,ax=plt.subplots(1,1)
        inalert=np.sum([r.inalert for r in self.rocs], axis=0)
        total_forecast_days=np.sum([r.forecast_days for r in self.rocs])
        fraction_inalert=inalert/total_forecast_days
        
        true_positive=np.sum([r.true_positive for r in self.rocs], axis=0)
        total_eruptions=np.sum([r.number_eruptions for r in self.rocs])
        auc=np.trapz(true_positive/total_eruptions, fraction_inalert)
        percentage_inalert=fraction_inalert*100
        percentage_true_positive=true_positive/total_eruptions*100
        
        ax.plot(percentage_inalert, percentage_true_positive, 'k-', label=f'ROC ({1.-auc})')
        ax.set_xlabel('time in warning [%]')
        ax.set_ylabel('eruptions in warning [%]')
        pmin,pmax=percentage_inalert[np.where(percentage_inalert>0)[0][0]],percentage_inalert[-1]
        
        if xscale=='log':
            ax.set_xscale('log')
            p=10**np.linspace(np.log10(pmin),np.log10(pmax),100)
        elif xscale=='linear':
            p=np.linspace(pmin,pmax,100)
        ax.plot(p,p,'k:',lw=0.5)
        ax.legend()

        if save is not None:
            plt.savefig(save, dpi=400)
        else:            
            plt.show()    

def merge_forecasts(forecasts, priority='last'):
    ''' Merge multiple forecasts into combined forecast.

        parameters:
            forecasts - List of Forecast objects to merge.

            priority - Rule for merging. Default is 'last' where duplicates occurring
                later in the forecasts list are kept. Alternatives are 'first', 'sample' and 'multi-volcano'. 
                For 'sample', out-of-sample duplicates are kept and in-sample are dropped. For 'multi-volcano',
                forecasts are simply concatenated with 

        returns:
            merged Forecast object

        notes:
            forecasts must have same sample spacing 

    '''
    if len(forecasts)==0:
        raise ValueError('forecast list is empty')
    # check errors
    ilfs=[fcst.ilf for fcst in forecasts]
    if len(np.unique(ilfs))>1:
        raise ValueError('forecasts have different look forward periods')
    # check edge cases
    if len(forecasts)==1:
        return forecasts[0]
    
    # pre-processing for forecast merge
    dts=[np.unique(np.diff(fcst.df.index)) for fcst in forecasts]
    for dt in dts:
        if np.timedelta64(0,'ns') in dt:
            raise ValueError('input forecast has duplicate indices')
    dts=np.unique([dt[0] for dt in dts])
    if len(dts)>1:
        raise ValueError('input forecasts do not have the same sample spacing')
    dt0=dts[0]

    if priority=='multi-volcano':
        return MultiVolcanoForecast(forecasts)
    
    # merge dataframes into first list entry
    df=pd.concat([fcst.df for fcst in forecasts], sort=False)

    # check overlap problems to be addressed
    dts=np.unique(np.diff(df.index.sort_values()))
    if np.timedelta64(0,'ns') in dts:
        has_duplicates=True
    else: 
        has_duplicates=False
    if has_duplicates and len(dts)>2:
        has_noncoincident_overlap=True
    elif not has_duplicates and len(dts)>1:
        has_noncoincident_overlap=True
    else:
        has_noncoincident_overlap=False
    
    if has_noncoincident_overlap:
        overlap_score=np.zeros(len(forecasts))
        for i,fcst_i in enumerate(forecasts):
            for j,fcst_j in enumerate(forecasts):
                if i==j:
                    continue
                ndt=(fcst_i.iy[0]-fcst_j.iy[0])/dt0 % 1

                if ndt != 0:
                    overlap_score[i]+=1
        priority_score=[o_s*fcst.iy.shape[0] for o_s,fcst in zip(overlap_score,forecasts)]
        
        i=np.argmax(priority_score)
        ti=np.min([fcst.iy[0] for fcst in forecasts])
        tf=np.max([fcst.iy[-1] for fcst in forecasts])
        ndti=int((forecasts[i].iy[0]-ti)/dt0)

        ti0=forecasts[i].iy[0]-ndti*dt0
        tf0=ti0+int((tf-ti)/dt0)*dt0
        idx=pd.date_range(ti0,tf0,freq=timedelta(seconds=dt0/np.timedelta64(1,'ns')/1.e9))
        dfs=[]
        for fcst in forecasts:
            t0,t1=fcst.iy[0],fcst.iy[-1]
            idxi=idx[(idx>=t0)&(idx<t1)]
            # df=fcst.df.reindex().interpolate(method='linear')
            df=fcst.df.reindex(fcst.df.index.union(idxi)).interpolate(method='linear').reindex(idxi)
            
            dfs.append(df)
        df=pd.concat(dfs, sort=False)


    if priority in ['first', 'last']:
        # rule for keeping overlaps
        df=df.loc[~df.index.duplicated(keep=priority)]
    elif priority=='sample':
        # when duplicates present, retain with value minimum out-of-sample label
        df=df.rename_axis('time').sort_values(by=['time','training label'], ascending=[True, True])
        df=df.loc[~df.index.duplicated(keep='first')]
    else:
        raise ValueError(f'unrecognized priority flag \'{priority}\'')

    tess=[]
    for fcst in forecasts:
        tess += fcst.tes

    forecasts[0].df=df.sort_index()
    forecasts[0].tes=list(set(tess))
    
    return forecasts[0]
    
def load_forecast(fl):
    ''' load forecast 

        fl - file name of forecast

        returns Forecast object
    '''
    return load_dataframe(fl)
