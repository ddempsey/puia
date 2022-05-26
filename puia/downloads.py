
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
    
STATIONS = {
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
            }
    }

class Downloader(object):
    def __init__(self):

        
        os.makedirs('_tmp', exist_ok=True)

        # default data range if not given 
        if ti is None:
            if self.tf is not None:
                ti = datetime(self.tf.year,self.tf.month,self.tf.day,0,0,0)
            else:
                ti = self._probe_start()
                
        tf = tf or datetime.today() + _DAY
        
        ti = datetimeify(ti)
        tf = datetimeify(tf)

        ndays = (tf-ti).days

        # parallel data collection - creates temporary files in ./_tmp
        pars = [[i,ti,self.station] for i in range(ndays)]
        n_jobs = self.n_jobs if n_jobs is None else n_jobs   
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
            p = Pool(n_jobs)
            p.starmap(get_data_for_day, pars)
            p.close()
            p.join()

        # read temporary files in as dataframes for concatenation with existing data
        cols = self._all_cols()
        dfs = [self.df[cols]]
        for i in range(ndays):
            fl = '_tmp/_tmp_fl_{:05d}.csv'.format(i)
            if not os.path.isfile(fl): 
                continue
            dfs.append(load_dataframe(fl, index_col=0, parse_dates=[0,], infer_datetime_format=True))
        shutil.rmtree('_tmp')
        self._df = pd.concat(dfs)

        # impute missing data using linear interpolation and save file
        self._df = self.df.loc[~self.df.index.duplicated(keep='last')]
        if True: #save non-interporlated data
            save_dataframe(self.df, self.file[:-4]+'_nitp'+self.file[-4:], index=True)

        self.df = self.df.resample('10T').interpolate('linear')

        save_dataframe(self.df, self.file, index=True)
        self.ti = self.df.index[0]
        self.tf = self.df.index[-1]

def get_data_from_stream(st, site):  
    if len(st.traces) == 0:
        raise
    elif len(st.traces) > 1:
        try:
            st.merge(fill_value='interpolate').traces[0]
        except Exception:
            st.interpolate(100).merge(fill_value='interpolate').traces[0]
              
    st.remove_sensitivity(inventory=site)
    # st.detrend('linear')
    return st.traces[0].data

def get_data_for_day(i,t0,station):
    """ Download WIZ data for given 24 hour period, writing data to temporary file.
        Parameters:
        -----------
        i : integer
            Number of days that 24 hour download period is offset from initial date.
        t0 : datetime.datetime
            Initial date of data download period.
        
    """
    t0 = UTCDateTime(t0)
    
    daysec = 24*3600
    fbands = [[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
    names = BANDS
    frs = [200,200,200, 100, 50]

    F = 100 # frequency
    D = 1   # decimation factor
    S = STATIONS[station]
    try:
        S['location']
    except KeyError:
        S['location']=None

    attempts = 0    
    while True:
        try:
            client = FDSNClient(S['client_name'])
            client_nrt = FDSNClient(S['nrt_name'])
            break
        except FDSNException:
            sleep(30)
            attempts += 1
            pass
        if attempts > 10:
            raise FDSNException('timed out after 10 attempts, couldn\'t connect to FDSN service')


    # download data
    datas = []
    columns = []
    try:
        site = client.get_stations(starttime=t0+i*daysec, endtime=t0 + (i+1)*daysec, station=station, level="response", channel=S['channel'])
    except (FDSNNoDataException, FDSNException):
        return

    pad_f=0.1
    try:
        st = client.get_waveforms(S['network'],station, S['location'], S['channel'], t0+(i-pad_f)*daysec, t0 + (i+1+pad_f)*daysec)
        
        # if less than 1 day of data, try different client
        data = get_data_from_stream(st, site)
        if data is None: return
        if len(data) < 600*F:
            raise FDSNNoDataException('')
    except (ValueError,ObsPyMSEEDFilesizeTooSmallError,FDSNNoDataException,FDSNException) as e:
        try:
            st = client_nrt.get_waveforms(S['network'],station, S['location'], S['channel'], t0+(i-pad_f)*daysec, t0 + (i+1+pad_f)*daysec)
            data = get_data_from_stream(st, site)
        except (FDSNNoDataException,ValueError,FDSNException):
            return

    # st.taper(max_percentage=0.05, type="hann")
    if D>1:
        st.decimate(D)
        F=F//D
    data = st.traces[0]
    i0=int((t0+i*daysec-st.traces[0].meta['starttime'])*F)+1
    if i0<0:
        return
    if i0 >= len(data):
        return
    i1=int(24*3600*F)
    if (i0+i1)>len(data):
        i1 = len(data)
    else:
        i1 += i0
    # process frequency bands
    dataI = cumtrapz(data, dx=1./F, initial=0)
    dataI -= dataI[i0]
    ti = st.traces[0].meta['starttime']+timedelta(seconds=(i0+1)/F)
        # round start time to nearest 10 min increment
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti = tiday+int(np.round((ti-tiday)/600))*600
    N = 600*F                             # 10 minute windows in seconds
    m = (i1-i0)//N
    Nm = N*m       # number windows in data
    
    # apply filters and remove filter response
    _datas = []; _dataIs = []
    for (fmin,fmax),fr in zip(fbands,frs):
        _data = abs(bandpass(data, fmin, fmax, F)[i0:i1])*1.e9
        _dataI = abs(bandpass(dataI, fmin, fmax, F)[i0:i1])*1.e9
        _datas.append(_data)
        _dataIs.append(_dataI)
    
    # find outliers in each 10 min window
    outliers = []
    maxIdxs = [] 
    for k in range(m):
        outlier, maxIdx = outlierDetection(_datas[2][k*N:(k+1)*N])
        outliers.append(outlier)
        maxIdxs.append(maxIdx)

    # compute rsam and other bands (w/ EQ filter)
    f = 0.1 # Asymmetry factor
    numSubDomains = 4
    subDomainRange = N//numSubDomains # No. data points per subDomain    
    for _data,name in zip(_datas, names):
        dr = []
        df = []
        for k, outlier, maxIdx in zip(range(m), outliers, maxIdxs):
            domain = _data[k*N:(k+1)*N]
            dr.append(np.mean(domain))
            if outlier: # If data needs filtering
                Idx = wrapped_indices(maxIdx, f, subDomainRange, N)
                domain = np.delete(domain, Idx) # remove the subDomain with the largest peak
            df.append(np.mean(domain))
        datas.append(np.array(dr)); columns.append(name)
        datas.append(np.array(df)); columns.append(name+'F')

    # compute dsar (w/ EQ filter)
    for j,rname in enumerate(RATIO_NAMES):
        dr = []
        df = []
        for k, outlier, maxIdx in zip(range(m), outliers, maxIdxs):
            domain_mf = _dataIs[j][k*N:(k+1)*N]
            domain_hf = _dataIs[j+1][k*N:(k+1)*N]
            dr.append(np.mean(domain_mf)/np.mean(domain_hf))
            if outlier: # If data needs filtering
                Idx = wrapped_indices(maxIdx, f, subDomainRange, N)
                domain_mf = np.delete(domain_mf, Idx) 
                domain_hf = np.delete(domain_hf, Idx) 
            df.append(np.mean(domain_mf)/np.mean(domain_hf))
        datas.append(np.array(dr)); columns.append(rname)
        datas.append(np.array(df)); columns.append(rname+'F')

    # write out temporary file
    datas = np.array(datas)
    time = [(ti+j*600).datetime for j in range(datas.shape[1])]
    df = pd.DataFrame(zip(*datas), columns=columns, index=pd.Series(time))
    save_dataframe(df, '_tmp/_tmp_fl_{:05d}.csv'.format(i), index=True, index_label='time')
