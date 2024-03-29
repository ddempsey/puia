
from datetime import timedelta
from puia.tests import run_tests
from puia.model import ForecastModel,MultiVolcanoForecastModel,MultiDataForecastModel
from puia.data import SeismicData, GeneralData
from puia.utilities import datetimeify, load_dataframe
from glob import glob
from sys import platform
import pandas as pd
import numpy as np
import os

_MONTH=timedelta(days=365.25/12)
_DAY=timedelta(days=1.)
_MIN=timedelta(days=1/24/60)

# set path depending on OS
if platform == "linux" or platform == "linux2":
    root=r'/media/eruption_forecasting/eruptions'
elif platform == "win32":
    root=r'U:\Research\EruptionForecasting\eruptions'
    root=r'C:\Users\dde62\code\alberto\EruptionForecasting'

DATA_DIR=f'{root}/data'
FEAT_DIR=f'{root}/features'
MODEL_DIR=f'{root}/models'
FORECAST_DIR=f'{root}/forecasts'

TI=datetimeify('2011-01-03')
TF=datetimeify('2019-12-29')

def reliability(root, data_streams, eruption, Ncl, eruption2=None):
    # setup forecast model
    n_jobs=10
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
    if True:
        if root is not 'benchmark':
            drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
            #freq_max=fm.dtw//fm.dt//4
            freq_max=int((2.*24*60)//(10.)//4)
            drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # train a model with data from that eruption excluded
    te=fm.data.tes[eruption-1]
    exclude_dates=[[te-_MONTH, te+_MONTH]]
    if eruption2 is not None:
        te=fm.data.tes[eruption2-1]
        exclude_dates.append([te-_MONTH, te+_MONTH])
    fm.train(TI, TF, drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    tf=te+_MONTH
    #if eruption==3:
    #    tf=te+_MONTH/28.*15
    #fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=tf, recalculate=True, n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)
    fm.hires_forecast(ti=te-_MONTH, tf=tf, recalculate=True, n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)
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
    if False:
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
    #performance(root)
    #return
    # assess reliability by cross validation on five eruptions
    for eruption in range(1,6):
        reliability(root, data_streams, eruption, Ncl)
        
    # assess discriminability by high-resoultion simulation across dataset
    discriminability(root, data_streams, Ncl)

    # summarise forecast performance
    performance(root)

def test_multi_data_forecast():
    # data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    # run_models(root='seismic_reference',data_streams=data_streams, Ncl=500)

    #data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF',
    #                'zsc2_2019']
    #run_models(root='seismic_template',data_streams=data_streams, Ncl=500)

    data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF',
                     'zsc2_Qm', 'zsc2_Dm', 'zsc2_Nm']
    run_models(root='seismic_physics',data_streams=data_streams, Ncl=500)

def main():
    test_multi_data_forecast()
    pass

if __name__=='__main__':
    main()