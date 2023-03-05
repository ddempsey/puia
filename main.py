
from datetime import timedelta
from puia.tests import run_tests
from puia.model import ForecastModel,CombinedModel,MultiDataForecastModel
from puia.utilities import datetimeify

_MONTH = timedelta(days=365.25/12)
DATA_DIR=r'U:\Research\EruptionForecasting\eruptions\data'
# FEAT_DIR=r'U:\Research\EruptionForecasting\eruptions\features'
DATA_DIR=r'C:\Users\dde62\code\alberto\EruptionForecasting\data'
FEAT_DIR=r'C:\Users\dde62\code\alberto\EruptionForecasting\features'
TI = datetimeify('2011-01-03')
TF = datetimeify('2019-12-31')

def reliability(root, data_streams, eruption, Ncl, eruption2=None):
    # setup forecast model
    n_jobs = 6 
    root = '{:s}_e{:d}'.format(root, eruption)
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
    fm = Model(data=data, window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root=root, savefile_type='pkl', feature_dir=FEAT_DIR, data_dir=DATA_DIR)   

    # train-test split on five eruptions to compute model confidence of an eruption
        # remove duplicate linear features (because correlated), unhelpful fourier compoents
        # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend']  
    if root is not 'benchmark':
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # train a model with data from that eruption excluded
    te = fm.data.tes[eruption-1]
    exclude_dates = [[te-_MONTH, te+_MONTH]]
    if eruption2 is not None:
        te = fm.data.tes[eruption2-1]
        exclude_dates.append([te-_MONTH, te+_MONTH])
    fm.train(TI, TF, drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # test the model by constructing a hires forecast for excluded eruption
    tf = te+_MONTH/28.
    if eruption==3:
        tf = te+_MONTH/28.*15
    fm.hires_forecast(ti=te-2*fm.dtw-fm.dtf, tf=tf, recalculate=False, n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)
    pass

def discriminability(root, data_streams, Ncl, eruption=None):
    # setup forecast model
    n_jobs=6
    if eruption is not None:
        root = '{:s}_e{:d}_p0'.format(root, eruption)
    else:
        root = '{:s}_e0'.format(root)
    
    data='WIZ'
    Model=ForecastModel
    if 'zsc2_Qm' in data_streams:
        data={'WIZ':['seismic','inversion']}
        Model=MultiDataForecastModel
    elif 'zsc2_2019' in data_streams:
        data={'WIZ':['seismic','dsarTemplate']}
        Model=MultiDataForecastModel
    fm = Model(data=data, window=2., overlap=0.75, look_forward=2., data_streams=data_streams,
        root=root, savefile_type='pkl', feature_dir=FEAT_DIR, data_dir=DATA_DIR, ti=datetimeify('2011-01-01'))   

    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    drop_features = ['linear_trend_timewise','agg_linear_trend']  
    if root is not 'benchmark':
        drop_features += ['*attr_"imag"*','*attr_"real"*','*attr_"angle"*']
        freq_max = fm.dtw//fm.dt//4
        drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    
    # construct hires model over entire dataset to compute false alarm rate
    exclude_dates = []
    if eruption is not None:
        te = fm.data.tes[eruption-1]
        exclude_dates = [[te-_MONTH, te+_MONTH]]
    fm.train(TI, TF, drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
    
    # forecast over whole dataset
    fm.hires_forecast(TI, TF, recalculate=False, n_jobs=n_jobs, root=r'{:s}_hires'.format(fm.root), threshold=1.0)    

def run_models(root, data_streams, Ncl=100):
    # assess reliability by cross validation on five eruptions
    for eruption in range(1,6):
        reliability(root, data_streams, eruption, Ncl)
        
    # assess discriminability by high-resoultion simulation across dataset
    discriminability(root, data_streams, Ncl)

def test_multi_data_load():
    # invocation types
        # one volcano, seismic data - station string
    a=ForecastModel(data='WIZ', data_dir=DATA_DIR)
        # multiple volcanoes, seismic data - list of station strings
    b=CombinedModel(data=['WIZ','FWVZ'], data_dir=DATA_DIR)
        # one volcano, multiple data types - dictionary: station string and list of data types
    c=MultiDataForecastModel(data={'WIZ':['seismic','inversion','dsarTemplate']}, data_dir=DATA_DIR)
    pass

def test_single_data_forecast():
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    run_models(root='test',data_streams=data_streams, Ncl=10)

def test_multi_data_forecast():
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    run_models(root='seismic_reference',data_streams=data_streams, Ncl=500)

    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF',
                    'zsc2_2019']
    run_models(root='seismic_template',data_streams=data_streams, Ncl=500)

    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF',
                    'zsc2_Qm', 'zsc2_Dm']
    run_models(root='seismic_physics',data_streams=data_streams, Ncl=500)

def main():
    #test_multi_data_load()
    # test_single_data_forecast()
    test_multi_data_forecast()
    pass

if __name__=='__main__':
    main()