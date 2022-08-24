import os, sys
sys.path.insert(0, os.path.abspath('..'))
from puia.data import TremorData, Data, Station, repair_dataframe
from puia.utilities import *
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
import time
from functools import partial
from multiprocessing import Pool

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
from obspy.signal.filter import bandpass

# constants
month = timedelta(days=365.25/12)
day = timedelta(days=1)

def import_data():
    if False: # plot raw vel data
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        from obspy import UTCDateTime
        #t = UTCDateTime("2012-02-27T00:00:00.000")
        starttime = UTCDateTime("2014-01-28")
        endtime = UTCDateTime("2014-01-30")
        inventory = client.get_stations(network="AV", station="SSLW", starttime=starttime, endtime=endtime)
        st = client.get_waveforms(network = "AV", station = "SSLW", location = None, channel = "EHZ", starttime=starttime, endtime=endtime)
        st.plot()  
        asdf

    t0 = "2012-01-01"
    t1 = "2013-07-01"
    td = TremorData(station = 'KRVZ')
    td.update(ti=t0, tf=t1)
    #td.update()

def data_Q_assesment():
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    station = 'KRVZ'
    # read raw data
    td = TremorData(station = station)
    #t0 = "2007-08-22"
    #t1 = "2007-09-22"
    #td.update(ti=t0, tf=t1)

    # plot data 
    #td.plot( data_streams = ['rsamF'])#(ti=t0, tf=t1)
    t0 = "2012-08-01"
    t1 = "2012-08-10"
    #td.update(ti=t0, tf=t1)
    data_streams = ['rsamF','mfF','hfF']
    td.plot_zoom(data_streams = data_streams, range = [t0,t1])

    # interpolated data 
    #t0 = "2015-01-01"
    #t1 = "2015-02-01"
    td.plot_intp_data(range_dates = None)
    #td.plot_intp_data(range_dates = [t0,t1])

def calc_feature_pre_erup():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''

    ## data streams
    #ds = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
    #    'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    ## stations
    ss = ['PVV','VNSS','SSLW','OKWR','REF','BELO','CRPO','VTUN','KRVZ','FWVZ','WIZ','AUR']
    ss = ['PVV','VNSS','KRVZ','FWVZ','WIZ','BELO'] # ,'SSLW'
    ## days looking backward from eruptions 
    lbs = [30]
    ## Run parallelization 
    ps = []
    #
    if True: # serial
        for s in ss:
            print(s)
            for d in ds:
                for lb in lbs:
                    p = [lb, s, d]
                    calc_one(p)
    else: # parallel
        for s in ss:
            for d in ds:
                for lb in lbs:
                    ps.append([lb,s,d])
        n_jobs = 4 # number of cores
        p = Pool(n_jobs)
        p.map(calc_one, ps)

def calc_one(p):
    ''' p = [weeks before eruption, station, datastream] 
    Load HQ data (by calculating features if need) before (p[0] days) every eruption in station given in p[1] for datastreams p[2]. 
    (auxiliary function for parallelization)
    '''
    lb,s,d = p
    #fm = ForecastModel(window=w, overlap=1., station = s,
    #    look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl') 
    fm = ForecastModel(window=2., overlap=1., station = s,
        look_forward=2., data_streams=[d], savefile_type='csv')
    a = fm.data.tes
    for etime in fm.data.tes:
        ti = etime - lb*day
        tf = etime 
        fm._load_data(ti, tf, None)

def corr_ana_feat():
    ''' Correlation analysis between features calculated for multiple volcanoes 
        considering 1 month before their eruptions.
    '''
    # load objects
    ## stations (volcanoes)
    ss = ['PVV','VNSS','KRVZ','FWVZ','WIZ','BELO'] # ,'SSLW'
    ## data streams
    ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    ## days looking backward from eruptions 
    lbs = [30]
    ## write a .csv file where rows and times 1-month before each eruption and columns are the eruptions considered
    fts = ['linear_trend_timewise','agg_linear_trend']
    # create a pandas dataframe for each feature where the columns are the eruptions consider
    pass # under developement 
    
def ruapehu_2009():
    
    q,rsam=np.genfromtxt(r'P:\My Documents\papers\ardid_lakes_grl\plot-data.csv', skip_header=1, delimiter=',').T

    f,ax=plt.subplots(1,1)
    ax.plot(np.log10(q), np.log10(rsam), 'kx')
    # ax.plot(q, rsam, 'kx')
    def y(x,m,c): return m*x+c
    from scipy.optimize import curve_fit
    p=curve_fit(y, np.log10(q), np.log10(rsam), [1,0])[0]
    print(p)
    plt.show()


    tes=['2021-03-04 13:20:00',
        '2016-11-13 11:00:00',
        '2013-09-04 12:00:00',
        '2007-11-06 22:10:00',
        '2006-10-04 09:20:00',
        '2015-04-24 03:30:00',
        '2012-07-03 10:30:00',
        '2007-10-03 19:10:00',
        '2007-09-25 08:20:00',
        '2016-12-29 02:30:00',
        '2007-10-26 18:20:00',
        '2009-07-13 06:30:00',
        '2010-09-03 16:30:00',
        '2009-04-14 02:10:00',
        '2015-10-12 08:00:00']

    #df=load_dataframe(r'U:\Research\EruptionForecasting\eruptions\data'+os.sep+'RU001_temp_data.csv', index_col=0, parse_dates=[0,], infer_datetime_format=True)
    #save_dataframe(df, r'U:\Research\EruptionForecasting\eruptions\data'+os.sep+'RU001A_level_data_repaired.csv', index=True)

    zd=Data(station='RU001A', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    td=TremorData(station='FWVZ', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    Td=Data(station='RU001', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    
    #td.set_timezone()

    if False:
        f,axs=plt.subplots(5,3, figsize=(20,10))
        axs=[axi for ax in axs for axi in ax]
    else:
        #td.update()
        f,ax=plt.subplots(1,1, figsize=(10,5))
        axs=[ax]
        tes=[(td.df.index[-1]-5*day).strftime('%Y-%m-%d %H:%M:%S')]

    ds=td.df['dsarF'].rolling(2*24*6).median()
    ds0=td.df['dsarF']
    
    plt.tight_layout()
    for ax,tei in zip(axs,tes):
        te=datetimeify(tei)
        t0,t1=[te-5*day,te+5*day]
        inds=(td.df.index>t0)&(td.df.index<t1)

        ax.plot(td.df.index[inds], td.df['rsam'][inds], 'k-', alpha=0.5)
        ax.plot(td.df.index[inds], td.df['rsamF'][inds], 'k-', label='rsam')
        ax.set_ylim([0, 3*td.df['rsamF'][inds].mean()])
        
        ax_=ax.twinx()
        # inds=(zd.df.index>t0)&(zd.df.index<t1)
        # ax_.plot(zd.df.index[inds], zd.df[' z (m)'][inds], 'b-')
        # ax.plot([],[], 'b-',label='lake level')
        #ax_.plot(ds.index[inds], ds[inds], 'b-')
        ax_.plot(ds0.index[inds], ds0[inds], 'b-', lw=0.5)
        ax.plot([],[], 'b-',label='dsar')
        inds=(Td.df.index>t0)&(Td.df.index<t1)
        ax__=ax.twinx()
        ax__.plot(Td.df.index[inds], Td.df[' t (C)'][inds], 'g-')
        #ax.plot([],[], 'g-',label='lake temp.')
        # ax.set_ylabel('rsam')
        # ax_.set_ylabel('lake level')
        ax.set_xlim([t0,t1])
        ax.text(0.95,0.95,tei[:10],transform=ax.transAxes,ha='right',va='top')
    ax.legend()
    
    if len(axs) == 1:
        plt.savefig('ruapehu_now.png',dpi=400)
    else:
        plt.savefig('ruapehu_events.png',dpi=400)

def ruapehu_2016_discharge():
    
    q,rsam=np.genfromtxt(r'P:\My Documents\papers\ardid_lakes_grl\plot-data.csv', skip_header=1, delimiter=',').T

    # f,ax=plt.subplots(1,1)
    # ax.plot(np.log10(q), np.log10(rsam), 'kx')
    # # ax.plot(q, rsam, 'kx')
    # def y(x,m,c): return m*x+c
    # from scipy.optimize import curve_fit
    # p=curve_fit(y, np.log10(q), np.log10(rsam), [1,0])[0]
    # print(p)
    # plt.show()

    tes=['2021-03-04 13:20:00',
        '2016-11-13 11:00:00',
        '2009-07-13 06:50:00',
        '2010-09-03 16:30:00',]

    td=TremorData(station='FWVZ', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    
    f,axs=plt.subplots(2,2, figsize=(10,5))
    axs=[axi for ax in axs for axi in ax]
    
    plt.tight_layout()
    for ax,tei in zip(axs,tes):
        te=datetimeify(tei)
        t0,t1=[te,te+day]
        inds=(td.df.index>t0)&(td.df.index<t1)

        ax.plot(np.arange(td.df['rsam'][inds].shape[0])/6, td.df['rsam'][inds], 'k-', lw=0.5)
        # ax.set_ylim([0, 3*td.df['rsamF'][inds].mean()])
        
        # ax.set_xlim([t0,t1])
        ax.text(0.95,0.95,tei[:10],transform=ax.transAxes,ha='right',va='top')
        ax.set_yscale('log')
        ax.set_xlabel('time since release [hours]')
        ax.set_ylabel('RSAM $\propto$ fluid release rate')
    ax.legend()
    
    plt.savefig('ruapehu_2016_discharge.png',dpi=400)
    
def whakaari_dsar():
       
    td=TremorData(station='FWVZ', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    #td.eruption_record.eruptions[2].date=datetimeify('2013-10-11 07:09:00')
    f,axs=plt.subplots(3,1, figsize=(7,7))
    #axs=[axi for ax in axs for axi in ax]
    
    ds=td.df['dsarF'].rolling(2*24*6).median()
    ds0=td.df['dsarF']

    # f,ax=plt.subplots()
    # ax.hist(ds0.values,bins=np.linspace(0,15,31),label='DSAR')
    # ax.axvline(np.percentile(ds0.values,50),color='r',linestyle='-',label='50pct')
    # ax.axvline(np.percentile(ds0.values,95),color='r',linestyle='--',label='90pct')
    # ax.set_xlabel('DSAR')
    # ax.legend()
    # plt.show()
    
    import matplotlib.dates as mdates
    plt.tight_layout()
    for ax,tei in zip(axs,td.tes):
        te=datetimeify(tei)
        t0,t1=[te-30*day,te+1*day]
        inds=(td.df.index>t0)&(td.df.index<t1)

        # ax.plot(td.df.index[inds], td.df['rsamF'][inds], 'k-', lw=0.5, label='rsam',alpha=0.5)
        # ax.set_ylim([0, 3*td.df['rsamF'][inds].mean()])
        ax.axvline(te,color='r',linestyle=':',label='eruption')
        
        # ax_=ax.twinx()
        ax_=ax
        ax_.plot(ds0.index[inds], ds0[inds], 'b-', alpha=0.3, lw=0.5)
        ax_.plot(ds.index[inds], ds[inds], 'b-')
        ax.plot([],[], 'b-',label='2-day dsar')
        ax.set_xlim([t0,t1])
        ax.text(0.03,0.95,tei.strftime('%b-%Y'),transform=ax.transAxes,ha='left',va='top')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[2].legend(loc=9)

    plt.savefig('whakaari_dsar.png',dpi=400)

def whakaari_rsam():
       
    td=TremorData(station='WIZ', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    #td.eruption_record.eruptions[2].date=datetimeify('2013-10-11 07:09:00')
    # f,axs=plt.subplots(3,1, figsize=(7,7))
    #axs=[axi for ax in axs for axi in ax]
    
    ds0=td.df['rsam']

    f,ax=plt.subplots()
    ts=[ds0.index[0],]
    ts+=[datetimeify(ti) for ti in ['2011-01-01','2014-01-01','2015-01-01','2018-01-01','2021-01-01']]
    bins=np.linspace(1,4,21)
    cs=['b','g','r','k']
    for ti,tf,c in zip(ts[:2],ts[1:3],cs[:2]):
        lbl=ti.strftime('%Y')+'-'+tf.strftime('%Y')
        ax.hist(np.log10(ds0[(ds0.index>ti)&(ds0.index<tf)].values),bins=bins,label=lbl,alpha=0.5,color=c)
    for ti,tf,c in zip(ts[3:-1],ts[4:],cs[2:]):
        lbl=ti.strftime('%Y')+'-'+tf.strftime('%Y')
        ax.hist(np.log10(ds0[(ds0.index>ti)&(ds0.index<tf)].values),bins=bins,label=lbl,alpha=0.5,color=c)
    ax.set_xlabel('log10(RSAM)')
    ax.set_ylabel('frequency')
    ax.legend()
    # plt.show()
    
    plt.savefig('whakaari_rsam.png',dpi=400)

def looking_for_vlps_ruapehu():

    f,ax=plt.subplots(1,1,figsize=(10,5))
    td=TremorData(station='FWVZ', data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    te=td.tes[1]
    
    te=datetimeify('2007 09 25 08 26 00')
    st=Station('FWVZ')
    t0,t1=(te-0.5*day,te+0.5*day)
    wv=st.get_waveform(t0,t1)

    
    inds=(td.df.index<t1)&(td.df.index>t0)
    df=td.df[inds]
    ax.axvline(te, color='r', linestyle=':')
    ax.plot(wv.index,wv['raw'],'k-')

    wv['vlp']=bandpass(wv['raw'].values, 1./25, 1./6.6, 100)
    wv['vlf']=bandpass(wv['raw'].values, 1./100, 1./10, 100)
    wv['lf']=bandpass(wv['raw'].values, 1./10, 1./0.5, 100)

    # ax_=ax.twinx()
    # ax_.plot(wv.index,wv['vlf'],'m-', lw=0.5)
    # ax_.plot(wv.index,wv['lf'],'c-', lw=0.5)
    
    ax__=ax.twinx()
    ax__.plot(df.index,df['vlf'],'g-')
    ax__.plot(df.index,df['vlfF'],'m-')
    # ax.set_xlim([te-day/24/3, te+day/24/6])
    
    plt.show()
    #plt.savefig('vlps.png',dpi=400)
def looking_for_vlps_whakaari():

    f,ax=plt.subplots(1,1,figsize=(10,5))
    # td=TremorData(station='WIZ', transforms=['log_zsc'],
    #     data_dir=r'U:\Research\EruptionForecasting\eruptions\data')
    
    te=datetimeify('2019-05-24 01:41:52')
    st=Station('WIZ')
    t0,t1=(te-0.25*day,te+0.25*day)
    #wv=st.get_waveform(t0,t1)
    wv=st.get_tremor(t0,t1,pad_f=0.)    
    wv1=st.get_tremor(t0,t1+day/24/6,pad_f=0.)

    # inds=(td.df.index<t1)&(td.df.index>t0)
    # df=td.df[inds]
    ax.axvline(te, color='r', linestyle=':')
    #ax.plot(wv.index,wv['raw'],'k-')

    # wv['vlp']=bandpass(wv['raw'].values, 1./25, 1./6.6, 100)
    # wv['vlf']=bandpass(wv['raw'].values, 1./100, 1./10, 100)
    #wv['lf']=bandpass(wv['raw'].values, 1./10, 1./0.5, 100)

    # ax_=ax.twinx()
    ax.plot(wv.index,wv['vlf'],'k-', lw=0.5)
    ax.plot(wv1.index,wv1['vlf'],'c-', lw=0.5)
    #ax_.plot(wv.index,wv['lf'],'c-', lw=0.5)
    
    # ax__=ax.twinx()
    # ax__.plot(df.index,df['log_zsc_vlf'],'g-')
    # ax__.plot(df.index,df['log_zsc_vlfF'],'m-')
    # ax.set_xlim([te-day/24/3, te+day/24/6])
    
    plt.show()
    #plt.savefig('vlps.png',dpi=400)

if __name__ == "__main__":
    # whakaari_dsar()
    # whakaari_rsam()
    # ruapehu_2009()
    ruapehu_2016_discharge()
    # looking_for_vlps_ruapehu()
    # looking_for_vlps_whakaari()
    #import_data()
    #data_Q_assesment()
    #calc_feature_pre_erup()
