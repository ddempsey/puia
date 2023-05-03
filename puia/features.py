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

class Feature(object):
    def __init__(self, parent, window, overlap, look_forward, sample_spacing, feature_dir):
        self.parent=parent
        self.window=window
        self.overlap=overlap
        self.look_forward=look_forward

        self.compute_only_features=[]
        
        # self.feature_root=feature_root
        self.feat_dir=feature_dir if feature_dir else f'{self.parent.root_dir}/features'
        self.featfile=lambda ds,yr,st: (f'{self.feat_dir}/fm_{self.window:3.2f}w_{ds}_{st}_{yr:d}.{self.parent.savefile_type}')
  
        # time stepping variables
        self.dtw=timedelta(days=self.window)            # window period
        self.dtf=timedelta(days=self.look_forward)      # look forward period
        self.dt=timedelta(seconds=sample_spacing/np.timedelta64(1,'s'))                          # data sample period
        self.dto=(1.-self.overlap)*self.dtw             # non-overlap period
        self.iw=int(self.dtw/self.dt)                   # index count in window
        self.io=int(self.overlap*self.iw)               # index count in overlap
        if self.io == self.iw: 
            self.io -= 1
        self.window=(self.iw*self.dt).total_seconds()/(24*3600)
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
