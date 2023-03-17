
from datetime import timedelta
from puia.tests import run_tests
from puia.model import ForecastModel,MultiVolcanoForecastModel,MultiDataForecastModel
from puia.data import SeismicData, GeneralData
from puia.utilities import datetimeify, load_dataframe
from glob import glob
from sys import platform
import pandas as pd
import numpy as np

_MONTH=timedelta(days=365.25/12)

# set path depending on OS
if platform == "linux" or platform == "linux2":
    root=r'/media/eruption_forecasting/eruptions'
elif platform == "win32":
    root=r'U:\Research\EruptionForecasting\eruptions'
    # root=r'C:\Users\dde62\code\alberto\EruptionForecasting'

DATA_DIR=f'{root}/data'
FEAT_DIR=f'{root}/features'
MODEL_DIR=f'{root}/models'
FORECAST_DIR=f'{root}/forecasts'

TI=datetimeify('2011-01-03')
TF=datetimeify('2019-12-31')

def reliability(root, data_streams, eruption, Ncl, eruption2=None):
    # setup forecast model
    n_jobs=6 
    root='{:s}_e{:d}'.format(root, eruption)
    if eruption2 is not None:
        root += '_p{:d}'.format(eruption2)
    data='WIZ'
    Model=ForecastModel
    if 'zsc2_Qm' in data_streams:
        data={'WIZ':['seismic','inversion']}
        Model=MultiDataForecastModel
    elif 'zsc2_2019' in data_streams:
        data={'WIZ':['seismic','dsarTemplate']}
        Model=MultiDataForecastModel
    fm=Model(data=data, window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root=root, savefile_type='pkl', feature_dir=FEAT_DIR, data_dir=DATA_DIR, model_dir=MODEL_DIR,
        forecast_dir=FORECAST_DIR)   

    # train-test split on five eruptions to compute model confidence of an eruption
        # remove duplicate linear features (because correlated), unhelpful fourier compoents
        # and fourier harmonics too close to Nyquist
    drop_features=['linear_trend_timewise','agg_linear_trend']  
    if root is not 'benchmark':
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max=fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # train a model with data from that eruption excluded
    te=fm.data.tes[eruption-1]
    exclude_dates=[[te-_MONTH, te+_MONTH]]
    if eruption2 is not None:
        te=fm.data.tes[eruption2-1]
        exclude_dates.append([te-_MONTH, te+_MONTH])
    fm.train(TI, TF, drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    tf=te+_MONTH/28.
    if eruption==3:
        tf=te+_MONTH/28.*15
    fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=tf, recalculate=False, n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)
    pass

def discriminability(root, data_streams, Ncl, eruption=None):
    # setup forecast model
    n_jobs=6
    if eruption is not None:
        root='{:s}_e{:d}_p0'.format(root, eruption)
    else:
        root='{:s}_e0'.format(root)
    
    data='WIZ'
    Model=ForecastModel
    if 'zsc2_Qm' in data_streams:
        data={'WIZ':['seismic','inversion']}
        Model=MultiDataForecastModel
    elif 'zsc2_2019' in data_streams:
        data={'WIZ':['seismic','dsarTemplate']}
        Model=MultiDataForecastModel
    fm=Model(data=data, window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root=root, savefile_type='pkl', feature_dir=FEAT_DIR, data_dir=DATA_DIR, model_dir=MODEL_DIR,
        forecast_dir=FORECAST_DIR)   

    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    drop_features=['linear_trend_timewise','agg_linear_trend']  
    if root is not 'benchmark':
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max=fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # construct hires model over entire dataset to compute false alarm rate
    exclude_dates=[]
    if eruption is not None:
        te=fm.data.tes[eruption-1]
        exclude_dates=[[te-_MONTH, te+_MONTH]]
    fm.train(TI, TF, drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # forecast over whole dataset
    fm.hires_forecast(TI, TF, recalculate=False, n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)    

def get_forecast(root, assumeUpdating=False):
    from glob import glob
    d=r'C:\Users\dde62\code\puia\forecasts'
    # Recursion L1
    # read in data for eruption
    if '_e' in root:
        fls=glob(d+f'\{root}_hires\consensus*.pkl')        
        ys=[]
        for fl in fls:
            ys.append(load_dataframe(fl))
        y0=pd.concat(ys).sort_index()
        return y0[(y0.index>TI)&(y0.index<TF)]

    # Recursion L0
    # read in data for all eruptions
    y0=get_forecast(f'{root}_e0')
    y0s=[get_forecast(f'{root}_e{i+1:d}') for i in range(5)]
    if assumeUpdating:
        # Special case: OCT 2013 eruption sequence
        # throws out model results after the first eruption, in which case subsequent
        # eruptions will be predicted assuming training on first in squence
        te=datetimeify('2013-10-03 12:35:00')
        y0s[2]=y0s[2][(y0s[2].index<te)]
    for i,yi in enumerate(y0s):
        inds=(y0.index>yi.index[0])&(y0.index<yi.index[-1])        
        y0.loc[inds,'consensus']=np.interp(y0.index[inds], xp=yi.index, fp=yi['consensus'])
    
    # y0=pd.concat(y0s)
    
    # Allow eruption out-of-sample simulations (i>0) to overwrite non-eruption (i=0)
    # y0=y0[~y0.index.duplicated(keep='last')].sort_index()
    return y0[(y0.index>TI)&(y0.index<TF)]

def performance(root):
    y=get_forecast(f'{root}', assumeUpdating=True)
    td=SeismicData(station='WIZ', data_dir=DATA_DIR)
    rsam=td.df['rsamF']
    rsam2=None
    if 'template' in root:
        td=GeneralData(station='WIZ', name='dsarTemplate', data_dir=DATA_DIR)
        rsam2=td.df['cc2019']
    elif 'physics' in root:
        td=GeneralData(station='WIZ', name='inversion', data_dir=DATA_DIR)
        rsam=td.df['Qm']
    tes=list(td.tes)[:2]+[datetimeify(te) for te in ['2013-10-08 02:05:00','2013-10-11 07:09:00']]+td.tes[2:]

    from matplotlib import pyplot as plt
    f,axs=plt.subplots(5,1,figsize=(8,8))
    for te,ax in zip(td.tes, axs):
        ti=te-_MONTH/6
        tf=te+_MONTH/15
        if td.tes.index(te)==2:
            tf=te+_MONTH/2.
        inds=(rsam.index>ti)&(rsam.index<tf)
        ax.plot(rsam.index[inds], rsam[inds].values,'k-',lw=0.5,label='rsam')
        ax_=ax.twinx()
        inds=(y.index>ti)&(y.index<tf)
        ax_.plot(y.index[inds], y[inds].values,'c-',lw=0.5)
        ax.plot([],[],'c-',lw=0.5,label='model')
        if rsam2 is not None:
            inds=(rsam2.index>ti)&(rsam2.index<tf)
            ax_.plot(rsam2.index[inds], rsam2[inds].values,'g-',lw=0.5)
            ax.plot([],[],'g-',lw=0.5,label='med_dsar CC')
        ax.set_title(f'{te.strftime("%b %Y")}')

        for te in tes:
            if te>ti and te<tf:
                ax.axvline(te, color='r',linestyle='--')
        ax_.set_ylim(-0.1,1.1)
        ax.set_xlim([ti,tf])
        ax_.axhline(1, color='k',alpha=0.5,linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{root}_forecasts.png',dpi=400)

def run_models(root, data_streams, Ncl=100):
    performance(root)
    return
    # assess reliability by cross validation on five eruptions
    for eruption in range(1,6):
        reliability(root, data_streams, eruption, Ncl)
        
    # assess discriminability by high-resoultion simulation across dataset
    discriminability(root, data_streams, Ncl)

    # summarise forecast performance
    performance(root)

def test_multi_volcano_forecast():
    # one volcano, multiple data types - dictionary: station string and list of data types
    data={'WIZ':['2019-12-01','2020-01-01'],
          'FWVZ':['2006-10-01','2006-11-01']}
    fm=MultiVolcanoForecastModel(data=data, window=2., overlap=0.75, look_forward=2., data_streams=['zsc2_rsamF','zsc2_dsarF'], root='seismic_WIZ_FWVZ',
                                 feature_dir=FEAT_DIR, data_dir=DATA_DIR, model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR)
    
    drop_features=['linear_trend_timewise','agg_linear_trend']  
    # train a model with the following data excluded
    i_WIZ=4         # 2019 Whakaari eruption
    te1=fm.data['WIZ'].tes[i_WIZ]
    i_FWVZ=0        # 2006 Ruapehu eruption
    te2=fm.data['FWVZ'].tes[i_FWVZ]
    
    # input format for exclude dates: dictionary keyed by station (eruptions not excluded in this test)
    exclude_dates={'WIZ':[[te1+_MONTH/12, te1+_MONTH/6]],
                   'FWVZ':[[te2+_MONTH/12, te2+_MONTH/6]]}
    
    # train combined model
    fm.train(drop_features=drop_features, retrain=True, Ncl=10, n_jobs=2, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for eruptions
        # 2019 Whakaari eruption
    tf=te1+_MONTH/28.
    fm.hires_forecast(station='WIZ', ti=te1-2*fm.ft.dtw-fm.ft.dtf, tf=tf, recalculate=True, n_jobs=2, 
                      root=f'WIZ_e{i_WIZ+1:d}_hires', threshold=1.0)
        # 2006 Ruapehu eruption
    tf=te2+_MONTH/28.
    fm.hires_forecast(station='FWVZ', ti=te2-2*fm.ft.dtw-fm.ft.dtf, tf=tf, recalculate=True, n_jobs=2, 
                      root=f'FWVZ_e{i_FWVZ+1:d}_hires', threshold=1.0)    

def test_cross_validation_multi_volcano():
    pass
    # define pool of volcanoes and record times 
    data={'WIZ':['2011-01-03','2019-12-31'],
            'FWVZ':['2005-01-01','2015-12-31'], 
                'KRVZ':['2010-01-01','2019-12-31'],
                    'CRPO':['2014-02-01','2017-04-22'],#['2014-07-02','2014-11-19'],
                        'GOD':['2010-03-06','2010-05-29'],#:['2019-12-01','2020-01-01'],
                            'VNSS':['2013-01-01','2019-12-31'],
                                'BELO':['2007-08-22','2010-07-10'],
                                    'VONK':['2014-01-02','2015-07-15'],
                                        'VTUN':['2014-08-05','2015-12-30'],
                                            'KINC':['2011-07-01','2013-01-15']}
    # define eruptions to use per volcano 
    eruptions={'WIZ':[0,2,3,4],
            'FWVZ':[0,1,2], 
                'KRVZ':[0],
                    'CRPO':[2],#['2014-07-02','2014-11-19'],
                        'GOD':[1],#:['2019-12-01','2020-01-01'],
                            'VNSS':[0,1],
                                'BELO':[0,1,2],
                                    'VONK':[0],
                                        'VTUN':[3],
                                            'KINC':[0]}                         
    # loop over eruptions (stattions and eruptions)
    for sta in eruptions:
        pass
        for erup in sta:
            pass
            # define training model
            fm=MultiVolcanoForecastModel(data=data, window=2., overlap=0.75, look_forward=2., data_streams=['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF'], root='seismic_WIZ_FWVZ',
                feature_dir=FEAT_DIR, data_dir=DATA_DIR, model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR)
            drop_features=['linear_trend_timewise','agg_linear_trend']
            # exclude data from eruption (train a model with the following data excluded)
            te1=fm.data[sta].tes[erup]
            exclude_dates={'sta':[[te1+_MONTH/12, te1+_MONTH/6]]}
            # train
            fm.train(drop_features=drop_features, retrain=True, Ncl=300, n_jobs=12, exclude_dates=exclude_dates)        
            # compute forecast over whole station period (and plot)
            tf=te1+_MONTH*.1#/28.
            ti=te1-_MONTH*.5#e2-2*fm.ft.dtw-fm.ft.dtf
            fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=12, 
                root=f+sta+'_e{erup+1:d}_hires', threshold=1.0, save='_fc_eruption.png')
            # plot forecast around eruption
            tf=datetimeify(data[sta][1])#te9+_MONTH*.1#/28.
            ti=datetimeify(data[sta][0])#te9-_MONTH*.5#e2-2*fm.ft.dtw-fm.ft.dtf
            fm.hires_forecast(station='VTUN', ti=ti, tf=tf, recalculate=True, n_jobs=12, 
                root=f'VTUN_e{i_VTUN+1:d}_hires', threshold=1.0, save='_fc_whole_period.png')
            # save models and forecast
    
def test_single_data_forecast():
    fm=ForecastModel(data='TEST', window=2., overlap=0.75, look_forward=2., data_streams=['zsc2_rsamF','zsc2_dsarF'],
        root='test', feature_dir=FEAT_DIR, data_dir=DATA_DIR, model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR)   

    drop_features=['linear_trend_timewise','agg_linear_trend']  
    # train a model with data from that eruption excluded
    te=fm.data.tes[-1]
    exclude_dates=[[te+_MONTH/6, te+_MONTH/3]]
    fm.train(drop_features=drop_features, retrain=False, Ncl=10, n_jobs=2, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    tf=te+_MONTH/28.
    fm.hires_forecast(ti=te-2*fm.ft.dtw-fm.ft.dtf, tf=tf, recalculate=True, n_jobs=2, root=r'{:s}_hires'.format(fm.root), threshold=1.0)
    

def test_multi_data_forecast():
    # data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    # run_models(root='seismic_reference',data_streams=data_streams, Ncl=500)

    data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF',
                    'zsc2_2019']
    run_models(root='seismic_template',data_streams=data_streams, Ncl=500)

    # data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF',
    #                 'zsc2_Qm', 'zsc2_Dm']
    # run_models(root='seismic_physics',data_streams=data_streams, Ncl=500)

def main():
    # test_single_data_forecast()
    # test_multi_data_forecast()
    test_multi_volcano_forecast()
    #test_cross_validation_multi_volcano()
    pass

if __name__=='__main__':
    main()
