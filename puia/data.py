"""Data package for puia."""

__author__ = """David Dempsey"""
__email__ = ''
__version__ = '0.1.0'

# general imports
import os, shutil, warnings
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
import pandas as pd
from multiprocessing import Pool
from time import sleep
from scipy.integrate import cumtrapz
from scipy.stats import mode

from .utilities import datetimeify, load_dataframe, save_dataframe, DummyClass

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
    failedobspyimport=False
except:
    failedobspyimport=True

RATIO_NAMES=['vlar','lrar','rmar','dsar']
BANDS=['vlf','lf','rsam','mf','hf']
FBANDS=[[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
FRS=[200,200,200, 100, 50]
_DAY=timedelta(days=1.)
_DAYSEC=24*3600

STATIONS={
    'WIZ':{
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'channel':'HHZ',
        'network':'NZ'
        },
    'KRVZ':{
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'channel':'EHZ',
        'network':'NZ'
        },
    'FWVZ':{
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'channel':'HHZ',
        'network':'NZ'
        },
    'PVV':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'EHZ',
        'network':'AV'
        },
    'PV6':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'EHZ',
        'network':'AV'
        },
    'OKWR':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'EHZ',
        'network':'AV'
        },
    'VNSS':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'EHZ',
        'network':'AV'
        },
    'SSLW':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'EHZ',
        'network':'AV'
        },
    'REF':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'EHZ',
        'network':'AV'
        },
    'BELO':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'HHZ',
        'network':'YC'
        },
    'CRPO':{
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'channel':'HHZ',
        'network':'OV'
        },
    'IVGP':{
            'client_name':'https://webservices.ingv.it',
            'nrt_name':'https://webservices.ingv.it',
            'channel':'HHZ',
            'network':'IV',
            'location':'*'
            },
    'AUS':{
            'client_name':'IRIS',
            'nrt_name':'https://service.iris.edu',
            'channel':'EHZ',
            'network':'AV',
            'location':'*',
            },
    'GSTD':{
            'client_name':'IRIS',
            'nrt_name':'https://service.iris.edu',
            'channel':'EHZ',
            'network':'AV',
            'location':'*',
            }
    }

class Data(object):
    """ Object to manage acquisition and processing of seismic data.
        
        Constructor arguments:
        ----------------------
        station : str
            Name of station to download seismic data from.
        Attributes:
        -----------
        df : pandas.DataFrame
            Time series of seismic data and transforms.
        ti : datetime.datetime
            Beginning of data range.
        tf : datetime.datetime
            End of data range.
        Methods:
        --------
        update
            Download latest GeoNet data.
        get_data
            Return seismic data in requested date range.
        plot
            Plot seismic data.
    """
    def __init__(self, station, parent=None, data_dir=None, file=None, transforms=None):
        self._station=Station(station)
        self.n_jobs=6
        self.parent=parent
        self.data_dir=data_dir
        if self.data_dir is not None:
            self._wd=lambda x: os.sep.join([self.data_dir,x])
        else:
            self._wd=lambda x: os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data',x])
        self.file=self._match_file(file)
        self._df_loaded=False
        self.ti=None
        self.tf=None
        self.dt=None
        self.transforms=transforms
        self._assess()
    def __repr__(self):
        if self.exists:
            tm=[self.ti.year, self.ti.month, self.ti.day, self.ti.hour, self.ti.minute]
            tm += [self.tf.year, self.tf.month, self.tf.day, self.tf.hour, self.tf.minute]
            return 'SeismicData:{:d}/{:02d}/{:02d} {:02d}:{:02d} to {:d}/{:02d}/{:02d} {:02d}:{:02d}'.format(*tm)
        else:
            return 'no data'
    def _all_cols(self):
        return RATIO_NAMES+[r+'F' for r in RATIO_NAMES]+BANDS+[b+'F' for b in BANDS]
    def _assess(self):
        """ Load existing file and check date range of data.
        """
        # get eruptions
        eruptionfile=self._wd(self.station+'_eruptive_periods.txt')        
        if not os.path.isfile(eruptionfile):
            self.eruption_record=None
        else:
            self.eruption_record=EruptionRecord(eruptionfile)
        # compute transforms
        if self.transforms is not None:
            self.df
            if self.parent is None:
                ds=[]
                for tf in self.transforms:
                    ds+=['{:s}_{:s}'.format(tf,d) for d in self.df.columns]
                self.parent=DummyClass(data_streams=ds)
            self._compute_transforms()
    def _match_file(self, file):
        # check for specified file name
        if file is not None:
            wfl=self._wd(file)
            if not os.path.isfile(wfl):
                raise FileNotFoundError('could not find {:s}'.format(wfl))
            else:
                return wfl
        from glob import glob

        # look for generic match using station identifier
        fls=glob(self._wd('{:s}_*.csv'.format(self.station)))
        if len(fls)>1:
            raise FileNotFoundError('file name ambiguity - found '+(len(fls)*'{:s}, ').format(*fls)[:-2] +' - use \'file\' keyword to specify')
        else:
            return fls[0]            
    def _load(self):
        # load data
        self._df=load_dataframe(self.file, index_col=0, parse_dates=[0,], infer_datetime_format=True)
        self._df_loaded=True
        self.data_streams=list(self.df.columns)
        
        # guard clause: empty dataframe
        if len(self.df.index)==0:
            return

        # assess data
        self.ti=self.df.index[0]
        self.tf=self.df.index[-1]
        md,cnt=mode(self.df.index[1:]-self.df.index[:-1])
        self.dt=md[0]
        if cnt[0] != (self.df.shape[0]-1):
            warnings.warn('non-uniform sampling detected in {:s}'.format(self.file)) 
        
        # remove timezone awareness
        self._tz_unaware()

        if _check_data(self._df):
            raise ValueError('data contains inf or nans that will prevent feature extraction - use \'repair_dataframe\' to fix')
    def _tz_unaware(self):
        # guard clauses: timezone unaware
        if not hasattr(self.df.index.dtype,'tz'):
            return
        if self.df.index.dtype is None:
            return
        
        # convert to utc then make unaware
        from datetime import timezone
        utc=timezone.utc
        iname=self.df.index.name+'_'
        if self.df.index.dtype.tz is not utc:
            self.df[iname]=[i.tz_convert(utc).replace(tzinfo=None) for i in self.df.index]
        else:
            self.df[iname]=[i.replace(tzinfo=None) for i in self.df.index]
        self.df.set_index(iname, inplace=True)        
    def _check_transform(self, name):
        if name not in self.df.columns and name in self.parent.data_streams:
            return True
        else: 
            return False
    def _compute_transforms(self):
        """ Compute data transforms.
            Notes:
            ------
            Naming convention is *transform_type*_*data_type*, so for example
            'inv_mf' is "inverse medium frequency or 1/mf. Other transforms are
            'diff' (derivative), 'log' (base 10 logarithm), 'stft' (short-time
            Fourier transform averaged across 40-45 periods), 'zsc' (z-score), 
            'zsc2' (z-score with 2-sample moving minimum).
        """
        from transforms import transform_functions
        for col in self.df.columns:
            # inverse
            for tf in transform_functions.keys():
                tf_col='{:s}_{:s}'.format(tf, col)
                if self._check_transform(tf_col):
                    self._df[tf_col]=transform_functions[tf](self.df[col])
    def _is_eruption_in(self, days, from_time):
        """ Binary classification of eruption imminence.
            Parameters:
            -----------
            days : float
                Length of look-forward.
            from_time : datetime.datetime
                Beginning of look-forward period.
            Returns:
            --------
            label : int
                1 if eruption occurs in look-forward, 0 otherwise
            
        """
        for te in self.tes:
            if 0 < (te-from_time).total_seconds()/(3600*24) < days:
                return 1.
        return 0.
    def get_data(self, ti=None, tf=None):
        """ Return seismic data in requested date range.
            Parameters:
            -----------
            ti : str, datetime.datetime
                Date of first data point (default is earliest data).
            tf : str, datetime.datetime
                Date of final data point (default is latest data).
            Returns:
            --------
            df : pandas.DataFrame
                Data object truncated to requsted date range.
        """
        # set date range defaults
        if ti is None:
            ti=self.ti
        if tf is None:
            tf=self.tf

        # convert datetime format
        ti=datetimeify(ti)
        tf=datetimeify(tf)

        # subset data
        inds=(self.df.index>=ti)&(self.df.index<tf)
        return self.df.loc[inds]   
    def plot(self, data_streams='rsam', save='seismic_data.png', ylim=None):
        """ Plot seismic data.

            Parameters:
            -----------
            save : str
                Name of file to save output.
            data_streams : str, list
                String or list of strings indicating which data or transforms to plot (see below). 
            ylim : list
                Two-element list indicating y-axis limits for plotting.
                
            data type options:
            ------------------
            rsam - 2 to 5 Hz (Real-time Seismic-Amplitude Measurement)
            mf - 4.5 to 8 Hz (medium frequency)
            hf - 8 to 16 Hz (high frequency)
            dsar - ratio of mf to hf, rolling median over 180 days

            transform options:
            ------------------
            inv - inverse, i.e., 1/
            diff - finite difference derivative
            log - base 10 logarithm
            stft - short-time Fourier transform at 40-45 min period

            Example:
            --------
            data_streams=['dsar', 'diff_hf'] will plot the DSAR signal and the derivative of the HF signal.
        """
        if type(data_streams) is str:
            data_streams=[data_streams,]
        if any(['_' in ds for ds in data_streams]):
            self._compute_transforms()

        # set up figures and axes
        f=plt.figure(figsize=(24,15))
        N=10
        dy1,dy2=0.05, 0.05
        dy3=(1.-dy1-(N//2)*dy2)/(N//2)
        dx1,dx2=0.43,0.03
        axs=[plt.axes([0.05+(1-i//(N/2))*(dx1+dx2), dy1+(i%(N/2))*(dy2+dy3), dx1, dy3]) for i in range(N)][::-1]
        
        for i,ax in enumerate(axs):
            ti,tf=[datetime.strptime('{:d}-01-01 00:00:00'.format(2011+i), '%Y-%m-%d %H:%M:%S'),
                datetime.strptime('{:d}-01-01 00:00:00'.format(2012+i), '%Y-%m-%d %H:%M:%S')]
            ax.set_xlim([ti,tf])
            ax.text(0.01,0.95,'{:4d}'.format(2011+i), transform=ax.transAxes, va='top', ha='left', size=16)
            ax.set_ylim(ylim)
            
        # plot data for each year
        data=self.get_data()
        xi=datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols=['c','m','y','g',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        for i,ax in enumerate(axs):
            if i//(N/2) == 0:
                ax.set_ylabel('data [nm/s]')
            else:
                ax.set_yticklabels([])
            a=ax.get_xlim()
            x0,x1 =[xi+timedelta(days=xl)-_DAY for xl in ax.get_xlim()]
            #testing
            inds=(data.index>=x0)&(data.index<=x1)
            inds=(data.index>=datetimeify(x0))&(data.index<=datetimeify(x1))
            #
            for data_stream, col in zip(data_streams,cols):
                ax.plot(data.index[inds], data[data_stream].loc[inds], '-', color=col, label=data_stream)
            for te in self.tes:
                ax.axvline(te, color='k', linestyle='--', linewidth=2)
            ax.axvline(te, color='k', linestyle='--', linewidth=2, label='eruption')
        axs[-1].legend()
        
        plt.savefig(save, dpi=400)
    def plot_zoom(self, data_streams='rsamF', save=None, range=None):
        """ Plot seismic data.

            Parameters:
            -----------
            save : str
                Name of file to save output.
            data_streams : str, list
                String or list of strings indicating which data or transforms to plot (see below). 
            range : list
                Two-element list indicating time range boundary
                
            data type options:
            ------------------
            rsam - 2 to 5 Hz (Real-time Seismic-Amplitude Measurement)
            mf - 4.5 to 8 Hz (medium frequency)
            hf - 8 to 16 Hz (high frequency)
            dsar - ratio of mf to hf, rolling median over 180 days

            transform options:
            ------------------
            inv - inverse, i.e., 1/
            diff - finite difference derivative
            log - base 10 logarithm
            stft - short-time Fourier transform at 40-45 min period

            Example:
            --------
            data_streams=['dsar', 'diff_hf'] will plot the DSAR signal and the derivative of the HF signal.
        """
        if type(data_streams) is str:
            data_streams=[data_streams,]
        if any(['_' in ds for ds in data_streams]):
            self._compute_transforms()

        # adding multiple Axes objects  
        fig, ax=plt.subplots(1, 1, figsize=(15,5))
        #ax.set_xlim(*range)
        # plot data for each year
        data=self.get_data(*range)
        xi=datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols=['c','m','y','g',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        inds=(data.index>=datetimeify(range[0]))&(data.index<=datetimeify(range[1]))
        for data_stream, col in zip(data_streams,cols):
            ax.plot(data.index[inds], data[data_stream].loc[inds], '-', color=col, label=data_stream)
        for te in self.tes:
            if [te>=datetimeify(range[0]) and te<=datetimeify(range[1])]:
                ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder=0)
        #
        ax.plot([], color='k', linestyle='--', linewidth=2, label='eruption')
        ax.set_xlim(*range)
        ax.legend()
        ax.grid()
        ax.set_ylabel('rsam')
        ax.set_xlabel('Time [year-month]')
        ax.title.set_text('Station '+self.station+': Seismic data')
        #plt.show()
        if not save:
            save='../data/plots/'+self.station+'_seismic_data_zoom.png'
        plt.savefig(save, dpi=400)
    def plot_intp_data(self, save=None, range_dates=None):
        """ Plot interpolated seismic data

            Parameters:
            -----------
            save : str
                Name of file to save output.
            range_dates : list
                Two-element list indicating time range boundary

            Example:
            --------
        """
        month=timedelta(days=365.25/12)
        # import interpolated data
        df_intp=load_dataframe(self.file, index_col=0, parse_dates=[0,], infer_datetime_format=True)
        # import non-interpolated data 
        df_non_intp=load_dataframe(self.file[:-4]+'_nitp'+self.file[-4:], index_col=0, parse_dates=[0,], infer_datetime_format=True)
        # % of interpolated data 
        p_intp=df_non_intp.shape[0] / df_intp.shape[0] * 100.
        
        # distribution of interpolated data 
        fig, (ax, ax2)=plt.subplots(2, 1, figsize=(12,5),gridspec_kw={'height_ratios': [1, 3]})
        _aux=df_intp['rsamF'] - df_non_intp['rsamF'] # ceros (point) and nans (interpolated)
        for i in range(_aux.size): 
            if _aux[i]:
                ax.plot([df_intp.index[i], df_intp.index[i]], [0, 1.], '-', color='red',  alpha=0.7, linewidth=0.3)#, label='data points')
        ax.plot([], [], '-', color='red',  alpha=0.5, linewidth=0.3, label='Location interpolated data points')
        ax.set_xlim([df_intp.index[0]-month, df_intp.index[-1]+month])
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #
        ax2.plot(df_non_intp.index, df_non_intp['rsamF'], '*', color='b', label='data points', markersize=1)
        ax2.plot(df_intp.index, df_intp['rsamF'], '-', color='k', alpha=0.5, linewidth=0.5, label='interpolated data')
        #
        for te in self.tes:
            #if [te>=datetimeify(range[0]) and te<=datetimeify(range[1])]:
            ax.axvline(te, color='k', linestyle='--', linewidth=.7, alpha=0.7, zorder=0)
            ax2.axvline(te, color='k', linestyle='--', linewidth=.7, alpha=0.7, zorder=0)
        ax.plot([], color='k', linestyle='--', linewidth=.7, label='eruption', alpha=0.7)
        ax2.plot([], color='k', linestyle='--', linewidth=.7, label='eruption', alpha=0.7)
        #
        ax2.set_xlim([df_intp.index[0]-month, df_intp.index[-1]+month])
        ax2.set_ylabel('rsam')
        ax2.set_xlabel('Time [year-month]')
        #
        if range_dates:
            range_dates=[datetimeify(range_dates[0]),datetimeify(range_dates[1])]
            ax.set_xlim([range_dates[0]-month, range_dates[1]+month])
            ax2.set_xlim([range_dates[0]-month, range_dates[1]+month])
        #
        ax.legend()
        ax2.legend()
        ax2.grid()
        ax.title.set_text('Station '+self.station+': interpolated data points '+str(round(p_intp,2))+'%')
        #fig.tight_layout()      
        #plt.show()
        plt.savefig('../data/plots/'+self.station+'_data_itp.png', dpi=400)
    def _get_tes(self):
        if self.eruption_record is None:
            return None
        return [e.date for e in self.eruption_record.eruptions]
    tes=property(_get_tes)
    def _get_df(self):
        if not self._df_loaded:
            self._load()
        return self._df
    df=property(_get_df)
    def _get_station(self):
        return self._station.name
    station=property(_get_station)

class SeismicData(Data):
    def __init__(self, station, parent=None, data_dir=None, transforms=None):
        file='{:s}_seismic_data.csv'.format(station)
        super(SeismicData,self).__init__(station, parent, data_dir, file, transforms)
    def update(self, ti=None, tf=None, n_jobs=None):
        """ Download latest GeoNet data.
            Parameters:
            -----------
            ti : str, datetime.datetime
                First date to retrieve data (default is first date data available).
            tf : str, datetime.datetime
                Last date to retrieve data (default is current date).
            n_jobs : int
                Number of CPUs to use.
        """
        if failedobspyimport:
            raise ImportError('ObsPy import failed, cannot update data.')

        if self._station._undefined:
            raise ValueError('no station download info for \'{:s}\''.format(self.station))

        os.makedirs('_tmp', exist_ok=True)

        # default data range if not given 
        if ti is None:
            if self.tf is not None:
                ti=datetime(self.tf.year,self.tf.month,self.tf.day,0,0,0)
            else:
                ti=self._probe_start()
                
        tf=tf or datetime.today() + _DAY
        
        ti=datetimeify(ti)
        tf=datetimeify(tf)

        ndays=(tf-ti).days

        # parallel data collection - creates temporary files in ./_tmp
        pars=[[i,ti,self.station] for i in range(ndays)]
        n_jobs=self.n_jobs if n_jobs is None else n_jobs   
        if n_jobs == 1: # serial 
            print('Station '+self.station+': Downloading data in serial')
            for par in pars:
                print(str(par[0]+1)+'/'+str(len(pars)))
                #print(str(par))
                get_data_for_day(*par)
        else: # parallel
            print('Station '+self.station+': Downloading data in parallel')
            print('From: '+ str(ti))
            print('To: '+ str(tf))
            print('\n')
            p=Pool(n_jobs)
            p.starmap(get_data_for_day, pars)
            p.close()
            p.join()

        # read temporary files in as dataframes for concatenation with existing data
        cols=self._all_cols()
        dfs=[self.df[cols]]
        for i in range(ndays):
            fl='_tmp/_tmp_fl_{:05d}.csv'.format(i)
            if not os.path.isfile(fl): 
                continue
            dfs.append(load_dataframe(fl, index_col=0, parse_dates=[0,], infer_datetime_format=True))
        shutil.rmtree('_tmp')
        self._df=pd.concat(dfs)

        # impute missing data using linear interpolation and save file
        self._df=self.df.loc[~self.df.index.duplicated(keep='last')]
        if True: #save non-interpolated data
            save_dataframe(self.df, self.file[:-4]+'_nitp'+self.file[-4:], index=True)

        self._df=self._df.resample('10T').interpolate('linear')

        save_dataframe(self.df, self.file, index=True)
        self.ti=self.df.index[0]
        self.tf=self.df.index[-1]
    def _probe_start(self, before=None):
        ''' Tries to figue out when the first available data for a station is.
        '''  
        s=STATIONS[self.station]
        client=FDSNClient(s['client_name'])    
        site=client.get_stations(station=self.station, level="response", channel=s['channel'])
        return site.networks[0].stations[0].start_date

class GeneralData(Data):
    def __init__(self, station, name, parent=None, data_dir=None, transforms=None):
        file=f'{station}_{name}_data.csv'
        super(GeneralData,self).__init__(station, parent, data_dir, file, transforms)

class Eruption(object):
    def __init__(self, date):
        self.date=datetimeify(date)

class EruptionRecord(object):
    def __init__(self, filename):
        self.filename=filename
        with open(filename,'r') as fp:
            self.eruptions=[Eruption(ln.rstrip()) for ln in fp.readlines()]

class Station(object):
    def __init__(self, name):
        self.name=name
        self._assign_attrs()
    def _assign_attrs(self):
        try:
            self._undefined=False
            STATIONS[self.name]
        except KeyError:
            self._undefined=True
            return
        _=[self.__setattr__(k,v) for k,v in STATIONS[self.name].items()]
        
        try:
            STATIONS[self.name]['location']
        except KeyError:
            self.location=None
    def get_waveform(self, t0, t1):
        """ Retrieve waveform data for the nominated time window.
        """
        t0=UTCDateTime(t0)
        t1=UTCDateTime(t1)
        tnow=UTCDateTime(datetime.utcnow().date())
        if not t0<t1:
            raise ValueError('endtime must be after starttime')
        
        attempts=0    
        while True:
            try:
                client=FDSNClient(self.client_name)
                client_nrt=FDSNClient(self.nrt_name)
                break
            except FDSNException:
                sleep(30)
                attempts += 1
                pass
            if attempts > 10:
                raise FDSNException('timed out after 10 attempts, couldn\'t connect to FDSN service')

        # download data
        try:
            site=client.get_stations(starttime=t0, endtime=t1, station=self.name, 
                level="response", channel=self.channel)
            F=site.networks[0].stations[0].channels[0].sample_rate
        except (FDSNNoDataException, FDSNException):
            return None


        try:
            # padded request
            st=client.get_waveforms(self.network, self.name, self.location,
                self.channel, t0, t1)# t0-_PADSEC, t1+_PADSEC)
            # if not the expected amount of data, try different client
            data=get_data_from_stream(st, site)
            if data is None:
                raise FDSNNoDataException('')
            data.F=F
            fraction=data.shape[0]/((t1-t0)*F)
        except (ValueError,ObsPyMSEEDFilesizeTooSmallError,FDSNNoDataException,FDSNException) as e:
            st=None
            data=None
            fraction=0.

        # sufficient data has been recovered and can be returned
        if not (st is None or ((tnow-t1)/(24*3600) < 5. and fraction<1.)):
            return data

        # check if there is more recent data on NRT
        try:
            st=client_nrt.get_waveforms(self.network, self.name, self.location, 
                self.channel, t0, t1)# t0-_PADSEC, t1+_PADSEC)
            data0=get_data_from_stream(st, site)
            if data0 is None:
                raise FDSNNoDataException('')
            data0.F=F
            fraction0=data0.shape[0]/((t1-t0)*F)
        except (FDSNNoDataException,ValueError,FDSNException):
            return None

        # determine which data to return
        if fraction0>fraction:
            return data0
        else:
            return data
    def get_seismic(self, t0, t1, pad=0.05, pad_f=0.05):
        t0=UTCDateTime(t0)
        t1=UTCDateTime(t1)
        padsec=pad*_DAYSEC
        padsecf=pad_f*_DAYSEC
        data=self.get_waveform(t0-padsec, t1+padsecf)
        N=int(600*data.F)                             # 10 minute windows in seconds
        if data is None:
            return None

        # round start time back to nearest 10 mins
        t010=UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(t0.year, t0.month, t0.day))
        dN=int(np.floor((t0.datetime-t010.datetime).total_seconds()/600))
        t0=t010+dN*600

        # in case start time now before data begins, round start time forward to nearest 10 mins
        if t0.datetime<data.index[0]:
            t0=data.index[0]
            t010=UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(t0.year, t0.month, t0.day))
            dN=int(np.ceil((t0-t010.datetime).total_seconds()/600))
            t0=t010+dN*600
        i0=abs(data.index-t0.datetime).argmin()
    
        # round end time back to nearest 10 mins
        t110=UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(t1.year, t1.month, t1.day))
        dN=int(np.floor((t1.datetime-t110.datetime).total_seconds()/600))
        t1=t110+dN*600

        # in case end time after data ends, round end time backward to nearest 10 mins
        if t1.datetime>data.index[-1]:
            t1=data.index[-1]
            t110=UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(t1.year, t1.month, t1.day))
            dN=int(np.floor((t1-t110.datetime).total_seconds()/600))
            t1=t110+dN*600
        i1=abs(data.index-t1.datetime).argmin()
        
        # process frequency bands
        dataI=cumtrapz(data['raw'], dx=1./data.F, initial=0)
        dataI -= dataI[i0]
        m=(i1-i0)//N
        inds=pd.date_range(t0.datetime,t1.datetime,m+1)[1:]
        
        # apply filters
        _datas=[]; _dataIs=[]
        for fmin,fmax in FBANDS:
            _data=abs(bandpass(data['raw'], fmin, fmax, data.F)[i0:i1])*1.e9
            _dataI=abs(bandpass(dataI, fmin, fmax, data.F)[i0:i1])*1.e9
            _datas.append(_data)
            _dataIs.append(_dataI)
        
        # find outliers in each 10 min window
        outliers=[]
        maxIdxs=[] 
        for k in range(m):
            outlier, maxIdx=outlierDetection(_datas[2][k*N:(k+1)*N])
            outliers.append(outlier)
            maxIdxs.append(maxIdx)

        # compute rsam and other bands (w/ EQ filter)
        f=0.1 # Asymmetry factor
        numSubDomains=4
        subDomainRange=N//numSubDomains # No. data points per subDomain    
        columns=[]
        datas=[]
        for _data,name in zip(_datas, BANDS):
            dr=[]
            df=[]
            for k, outlier, maxIdx in zip(range(m), outliers, maxIdxs):
                domain=_data[k*N:(k+1)*N]
                dr.append(np.mean(domain))
                if outlier: # If data needs filtering
                    Idx=wrapped_indices(maxIdx, f, subDomainRange, N)
                    domain=np.delete(domain, Idx) # remove the subDomain with the largest peak
                df.append(np.mean(domain))

            datas.append(np.array(dr)); columns.append(name)
            datas.append(np.array(df)); columns.append(name+'F')

        # compute dsar (w/ EQ filter)
        for j,rname in enumerate(RATIO_NAMES):
            dr=[]
            df=[]
            for k, outlier, maxIdx in zip(range(m), outliers, maxIdxs):
                domain_mf=_dataIs[j][k*N:(k+1)*N]
                domain_hf=_dataIs[j+1][k*N:(k+1)*N]
                dr.append(np.mean(domain_mf)/np.mean(domain_hf))
                if outlier: # If data needs filtering
                    Idx=wrapped_indices(maxIdx, f, subDomainRange, N)
                    domain_mf=np.delete(domain_mf, Idx) 
                    domain_hf=np.delete(domain_hf, Idx) 
                df.append(np.mean(domain_mf)/np.mean(domain_hf))
            datas.append(np.array(dr)); columns.append(rname)
            datas.append(np.array(df)); columns.append(rname+'F')

        # write out temporary file
        datas=np.array(datas)
        df=pd.DataFrame(zip(*datas), columns=columns, index=inds)
        return df

def repair_dataframe(broken, fixed, percentiles=[0.5,99.5]):
    '''
    Replace inf and nan in dataframe.

    '''
    df=load_dataframe(broken, index_col=0, parse_dates=[0,], infer_datetime_format=True)
    
    # check and correct monotonicity
    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    # check and delete outliers
    for col in df.columns:
        data=df[col]
        log_dsar=np.log10(data)
        dmin=np.percentile(log_dsar,percentiles[0])
        dmax=np.percentile(log_dsar,percentiles[1])
        dmid=0.5*(dmin+dmax)
        drange=dmax-dmin
        dmin=dmid-drange
        dmax=dmid+drange
        data[(dmin>log_dsar)|(log_dsar>dmax)]=np.nan
        data.interpolate(inplace=True)

        if _check_data(data):
            data[(-1e-30>data)|(data>1e30)]=np.nan
            data.interpolate(inplace=True)

        df[col]=data

    save_dataframe(df, fixed, index=True)

def get_data_from_stream(st, site):  
    if len(st.traces) == 0:
        raise
    elif len(st.traces) > 1:
        try:
            st.merge(fill_value='interpolate')
        except Exception:
            st.interpolate(100).merge(fill_value='interpolate')
              
    st.remove_sensitivity(inventory=site)
    t1,t0=(st.traces[0].meta['endtime'],st.traces[0].meta['starttime'])
    return pd.DataFrame(np.array([st.traces[0].data,]).T, columns=['raw'], 
        index=pd.date_range(t0.datetime,t1.datetime,st.traces[0].data.shape[0])) 

def _check_data(df):
        # check for infs or nans which will cause problems during feature extraction
        ts=[df.max()] if type(df)==pd.Series else df.max()
        if not any([t>1.e30 for t in ts]):
            # check +inf
            pass
        else:
            print('contains +inf')
            return True
        ts=[df.max()] if type(df)==pd.Series else df.min()
        if not any([t<-1.e30 for t in ts]):
            # check -inf
            pass
        else:
            print('contains -inf')
            return True
        ts=[df.max()] if type(df)==pd.Series else df.mean()
        if not any([t!=t for t in ts]):
            # check nan
            pass
        else:
            print('contains nan')
            return True
        if not df.index.has_duplicates:
            # check indices in order
            pass
        else:
            print('there are duplicated indices')
            return True
        if df.index.is_monotonic:
            # check indices in order
            pass
        else:
            print('indices are not monotonic')
            return True
        return False

def get_data_for_day(i,t0,station):
    """ Download WIZ data for given 24 hour period, writing data to temporary file.
        Parameters:
        -----------
        i : integer
            Number of days that 24 hour download period is offset from initial date.
        t0 : datetime.datetime
            Initial date of data download period.
        
    """
    t0=UTCDateTime(t0)
    
    daysec=24*3600
    fbands=[[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
    names=BANDS
    frs=[200,200,200, 100, 50]

    F=100 # frequency
    D=1   # decimation factor
    S=STATIONS[station]
    try:
        S['location']
    except KeyError:
        S['location']=None

    attempts=0    
    while True:
        try:
            client=FDSNClient(S['client_name'])
            client_nrt=FDSNClient(S['nrt_name'])
            break
        except FDSNException:
            sleep(30)
            attempts += 1
            pass
        if attempts > 10:
            raise FDSNException('timed out after 10 attempts, couldn\'t connect to FDSN service')

    # download data
    datas=[]
    columns=[]
    try:
        site=client.get_stations(starttime=t0+i*daysec, endtime=t0 + (i+1)*daysec, station=station, level="response", channel=S['channel'])
    except (FDSNNoDataException, FDSNException):
        return

    pad_f=0.1
    try:
        st=client.get_waveforms(S['network'],station, S['location'], S['channel'], t0+(i-pad_f)*daysec, t0 + (i+1+pad_f)*daysec)
        
        # if less than 1 day of data, try different client
        data=get_data_from_stream(st, site)
        if data is None: return
        if len(data) < 600*F:
            raise FDSNNoDataException('')
    except (ValueError,ObsPyMSEEDFilesizeTooSmallError,FDSNNoDataException,FDSNException) as e:
        try:
            st=client_nrt.get_waveforms(S['network'],station, S['location'], S['channel'], t0+(i-pad_f)*daysec, t0 + (i+1+pad_f)*daysec)
            data=get_data_from_stream(st, site)
        except (FDSNNoDataException,ValueError,FDSNException):
            return

    if D>1:
        st.decimate(D)
        F=F//D
    data=st.traces[0]
    i0=int((t0+i*daysec-st.traces[0].meta['starttime'])*F)+1
    if i0<0:
        return
    if i0 >= len(data):
        return
    i1=int(24*3600*F)
    if (i0+i1)>len(data):
        i1=len(data)
    else:
        i1 += i0
    # process frequency bands
    dataI=cumtrapz(data, dx=1./F, initial=0)
    dataI -= dataI[i0]
    ti=st.traces[0].meta['starttime']+timedelta(seconds=(i0+1)/F)
        # round start time to nearest 10 min increment
    tiday=UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti=tiday+int(np.round((ti-tiday)/600))*600
    N=600*F                             # 10 minute windows in seconds
    m=(i1-i0)//N
    
    # apply filters and remove filter response
    _datas=[]; _dataIs=[]
    for (fmin,fmax),fr in zip(fbands,frs):
        _data=abs(bandpass(data, fmin, fmax, F)[i0:i1])*1.e9
        _dataI=abs(bandpass(dataI, fmin, fmax, F)[i0:i1])*1.e9
        _datas.append(_data)
        _dataIs.append(_dataI)
    
    # find outliers in each 10 min window
    outliers=[]
    maxIdxs=[] 
    for k in range(m):
        outlier, maxIdx=outlierDetection(_datas[2][k*N:(k+1)*N])
        outliers.append(outlier)
        maxIdxs.append(maxIdx)

    # compute rsam and other bands (w/ EQ filter)
    f=0.1 # Asymmetry factor
    numSubDomains=4
    subDomainRange=N//numSubDomains # No. data points per subDomain    
    for _data,name in zip(_datas, names):
        dr=[]
        df=[]
        for k, outlier, maxIdx in zip(range(m), outliers, maxIdxs):
            domain=_data[k*N:(k+1)*N]
            dr.append(np.mean(domain))
            if outlier: # If data needs filtering
                Idx=wrapped_indices(maxIdx, f, subDomainRange, N)
                domain=np.delete(domain, Idx) # remove the subDomain with the largest peak
            df.append(np.mean(domain))
        datas.append(np.array(dr)); columns.append(name)
        datas.append(np.array(df)); columns.append(name+'F')

    # compute dsar (w/ EQ filter)
    for j,rname in enumerate(RATIO_NAMES):
        dr=[]
        df=[]
        for k, outlier, maxIdx in zip(range(m), outliers, maxIdxs):
            domain_mf=_dataIs[j][k*N:(k+1)*N]
            domain_hf=_dataIs[j+1][k*N:(k+1)*N]
            dr.append(np.mean(domain_mf)/np.mean(domain_hf))
            if outlier: # If data needs filtering
                Idx=wrapped_indices(maxIdx, f, subDomainRange, N)
                domain_mf=np.delete(domain_mf, Idx) 
                domain_hf=np.delete(domain_hf, Idx) 
            df.append(np.mean(domain_mf)/np.mean(domain_hf))
        datas.append(np.array(dr)); columns.append(rname)
        datas.append(np.array(df)); columns.append(rname+'F')

    # write out temporary file
    datas=np.array(datas)
    time=[(ti+j*600).datetime for j in range(datas.shape[1])]
    df=pd.DataFrame(zip(*datas), columns=columns, index=pd.Series(time))
    save_dataframe(df, '_tmp/_tmp_fl_{:05d}.csv'.format(i), index=True, index_label='time')

def wrapped_indices(maxIdx, f, subDomainRange, N):
    ''' helper function for computing filtering masks
    '''
    startIdx=int(maxIdx-np.floor(f*subDomainRange)) # Compute the index of the domain where the subdomain centered on the peak begins
    endIdx=startIdx+subDomainRange # Find the end index of the subdomain
    if endIdx >= N: # If end index exceeds data range
        Idx=list(range(endIdx-N)) # Wrap domain so continues from beginning of data range
        end=list(range(startIdx, N))
        Idx.extend(end)
    elif startIdx < 0: # If starting index exceeds data range
        Idx=list(range(endIdx))
        end=list(range(N+startIdx, N)) # Wrap domains so continues at end of data range
        Idx.extend(end)
    else:
        Idx=list(range(startIdx, endIdx))
    return Idx

def outlierDetection(data, outlier_degree=0.5):
    """ Determines whether a given data interval requires earthquake filtering
        Parameters:
        -----------
        data : list
            10 minute interval of a processed datastream (rsam, mf, hf, mfd, hfd).
        outlier_degree : float
            exponent (base 10) which determines the Z-score required to be considered an outlier.
        
        Returns:
        --------
        outlier : boolean
            Is the maximum of the data considered an outlier?
        maxIdx : int
            Index of the maximum of the data
    """
    mean=np.mean(data)
    std=np.std(data)
    maxIdx=np.argmax(data)
    # compute Z-score
    Zscr=(data[maxIdx]-mean)/std
    # Determine if an outlier
    if Zscr > 10**outlier_degree:
        outlier=True
    else:
        outlier=False
    return outlier, maxIdx

##

class SeismicDataCombined(SeismicData):
    def __init__(self, stations, parent=None):
        self.stations = stations
        self.station = '_'.join(sorted(self.stations))
        self._datas = []
        self.tes = []
        self.df = []
        for station in stations:
            self._datas.append(SeismicData(station, parent))
            #fl = os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','{:s}_eruptive_periods.txt'.format(station)])
            fl = '..\\data\\'+'{:s}_eruptive_periods.txt'.format(station)
            with open(fl,'r') as fp:
                self._datas[-1].tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            self.tes += self._datas[-1].tes
            self.df.append(self._datas[-1].df)
        self.df = pd.concat(self.df)
        #self.tes = sorted(list(set(self.tes))) # ## testing: need to be checked
        self.ti = np.min([station.ti for station in self._datas])
        self.tf = np.max([station.tf for station in self._datas])
    def _compute_transforms(self):
        [station._compute_transforms() for station in self._datas]
        self.df = pd.concat([station.df for station in self._datas])
    def update(self, ti=None, tf=None, n_jobs = None):
        [station.update(ti,tf,n_jobs) for station in self._datas]
    def get_data(self, ti=None, tf=None):
        return pd.concat([station.get_data(ti,tf) for station in self._datas])
    def plot(self):
        raise NotImplementedError('method not implemented')