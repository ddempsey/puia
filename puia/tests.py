# general imports
import os, ast
from inspect import getfile, currentframe
from utilities import DummyClass
from model import TrainModelCombined
from forecast import ForecastTransLearn
from datetime import datetime, timedelta
from utilities import datetimeify, load_dataframe, save_dataframe
from data import TremorData
import matplotlib.pyplot as plt
import numpy as np


# constants
month = timedelta(days=365.25/12)
day = timedelta(days=1)
minute = timedelta(minutes=1)

def test_data():
    from data import TremorData
    wd=os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data'])
    td=TremorData(station='TEST', data_dir=wd)
    rs=td.df['rsam']
    td.parent=DummyClass(data_streams=['inv_rsam', 'diff_dsar'])
    td._compute_transforms()

def run_tests(testname='all'):
    # runs selected tests
    testfile = getfile(currentframe())
    with open(testfile,'r') as fp:
        tree = ast.parse(fp.read(), filename=testfile)
        
    fs = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]
    fs.pop(fs.index('run_tests'))

    if testname is 'all':
        for f in fs:
            eval('{:s}()'.format(f))
        return
    
    testname = 'test_'+testname
    if testname not in fs:
        raise ValueError('test function \'{:s}\' not found'.format(testname))

    eval('{:s}()'.format(testname))


# testing
if __name__ == "__main__":
    if False:
        test_data()
    if False: # test download data
        from obspy import UTCDateTime 
        from datetime import timedelta
        #    stations = ['FWVZ','BELO','REF','SSLW','VNSS','OKWR']
        stations=['GSTD']
        dt = timedelta(days=20)
        for station in stations:
            try:
                td = TremorData(station=station)
                ti = td._probe_start()
                if station == 'WIZ': ti = UTCDateTime(datetimeify('2008-01-01'))
                if station == 'OKWR': ti = UTCDateTime(datetimeify('2008-01-01'))
                if station == 'VNSS': ti = UTCDateTime(datetimeify('2013-01-01'))
                if station == 'AUS': ti = UTCDateTime(datetimeify('2005-11-01'))
                if station == 'IVGP': ti = UTCDateTime(datetimeify('2019-08-10'))
                if station == 'IVUG': ti = UTCDateTime(datetimeify('2020-08-10'))
                if station == 'ISTR': ti = UTCDateTime(datetimeify('2021-08-10'))
                if station == 'FWVZ': ti = UTCDateTime(datetimeify('2005-06-01'))
                if station == 'PVV': ti = UTCDateTime(datetimeify('2014-01-01'))
                if station == 'GSTD': ti = UTCDateTime(datetimeify('2021-05-10'))
                if td.tf is not None:
                    from copy import deepcopy
                    ti = UTCDateTime(deepcopy(td.tf))
                N = int(np.ceil((datetime.today()-ti._get_datetime())/dt))
                for i in range(N):
                    t0=ti+i*dt
                    t1=ti+(i+1)*dt
                    if t1>datetime.today():
                        t1 = datetime.today()
                    td.update(t0, t1, n_jobs=4)
            except:
                with open('{:s}_download.err'.format(station),'w') as fp:
                    fp.write(str(traceback.format_exc())+'\n')
                    fp.write(str(sys.exc_info()[0]))
                try:
                    shutil.rmtree('_tmp')
                except:
                    pass
                pass
        asdf
    if False: # testing TL forecasting 
        if False:
            rootdir=r'U:\Research\EruptionForecasting\eruptions'
            datadir=r'U:\Research\EruptionForecasting\eruptions\data' 
            featdir=r'U:\Research\EruptionForecasting\eruptions\features'
            modeldir=r'U:\Research\EruptionForecasting\eruptions\models'
            predicdir=r'U:\Research\EruptionForecasting\eruptions\predictions'
            plotdir=r'U:\Research\EruptionForecasting\eruptions\plots'
        if True:
            rootdir=r'E:\EruptionForecasting'
            datadir=r'E:\EruptionForecasting\data' 
            featdir=r'E:\EruptionForecasting\features'
            modeldir=r'E:\EruptionForecasting\models'
            predicdir=r'E:\EruptionForecasting\predictions'
            plotdir=r'E:\EruptionForecasting\plots'
        # feat selection
        fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts' 
        #fl_lt = None
        # 
        datastream = ['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF']
        stations=['WIZ']#,'FWVZ'] # this could contain multiple stations
        dtb = 30 # looking back from eruption times
        dtf = 0  # looking forward from eruption times
        win=2.   # window length
        lab_lb=4.# days to label as eruptive before the eruption times 
        #
        root='FM_'+str(int(win))+'w_'+'-'.join(datastream)+'_'+'-'.join(stations)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'
        #
        ## (1) training models with 'stations'
        # combined features matrices are save in featdir in a folder named by the attributes 
        fm0 = TrainModelCombined(stations=stations,window=win, overlap=1., dtb=dtb, dtf=dtf, datastream=datastream,
            rootdir=rootdir,root=root,feat_dir=featdir, data_dir=datadir, tes_dir=datadir,feat_selc=fl_lt, model_dir=modeldir,
                lab_lb=lab_lb,noise_mirror=True, savefile_type='csv') # 
        #
        fm0.train(Nfts=2, Ncl=2, retrain=True, classifier="DT", random_seed=0, method=0.75, n_jobs=4)

        ## (2) Forecasting on test station 
        # usgin model created from 'stations' to predict on a diferent station called 'station_test'
        model_name=root # model to be use
        fm0 = ForecastTransLearn(model_name,rootdir=rootdir,root=root,datadir=datadir,
            modeldir=modeldir,featdir=featdir,predicdir=predicdir, plotdir=plotdir, 
                savefile_type='csv') # 
        # run forec
        station_test='FWVZ'         # only one station
        ti_forecast='2007-09-01'   
        tf_forecast='2007-10-31'
        #
        ys = fm0.forecast(station_test=station_test,ti_forecast=ti_forecast, tf_forecast=tf_forecast, 
            recalculate=True) # generate a consensus file in predicdir folder
        # Need to check that the model.predict in line 153 in predict_models function (forecast.py) is working fine. 
        '''
        To do Feature class
            - method to mark rows with imcomplete data 
            - check sensitivity to random number seed for noise mirror (FeaturesMulti._load_tes())
            - criterias for feature selection (see FeaturesSta.reduce()

        To do TrainModelCombined class"
            - test that is working correctly

        To do ForecastTransLearn class:
            - test that is working correctly
            - format to save dataframe with metadata
            - add a method to forecast accuracy 
            - methods to implement: get_performance, _compute_CI, plot_performance
        '''
    if True: # testing TL cross validation
        if False: 
            rootdir=r'U:\Research\EruptionForecasting\eruptions'
            datadir=r'U:\Research\EruptionForecasting\eruptions\data' 
            featdir=r'U:\Research\EruptionForecasting\eruptions\features'
            modeldir=r'U:\Research\EruptionForecasting\eruptions\models'
            predicdir=r'U:\Research\EruptionForecasting\eruptions\predictions'
            plotdir=r'U:\Research\EruptionForecasting\eruptions\plots'
        if True: 
            rootdir=r'E:\EruptionForecasting'
            datadir=r'E:\EruptionForecasting\data' 
            featdir=r'E:\EruptionForecasting\features'
            modeldir=r'E:\EruptionForecasting\models'
            predicdir=r'E:\EruptionForecasting\predictions'
            plotdir=r'E:\EruptionForecasting\plots'
            #rootdir='../../EruptionForecasting'
            #datadir='../../EruptionForecasting/data' 
            #featdir='../../EruptionForecasting/features'
            #modeldir='../../EruptionForecasting/models'
            #predicdir='../../EruptionForecasting/predictions'
            #plotdir='../../EruptionForecasting/plots'
        # feat selection
        # fl_lt = r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\test_codes\pca\feature_selection\whakaari\models\test\all.fts'#'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\models\test\all.fts' 
        # fl_lt = featdir+r'\all.fts'
        fl_lt = None
        # 
        ## cross-validation 
        stations=['WIZ','WIZ']#,'KRVZ'] # this could contain multiple stations. If want to forecast in the same stations, repeat staion
        no_erup = None#['WIZ',4] # remove eruption from training (e.g., ['WIZ',4]; eruption number, as 4, start counting from 0)
        datastream =  ['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF']
        dtb = 180 # looking back from eruption times
        dtf = 30  # looking forward from eruption times
        win=2.   # window length
        lab_lb=2.# days to label as eruptive before the eruption times 
        noise_mirror=True
        #
        if True: # training and forecasting loop
            for i,sta_test in enumerate(stations):
                sta_train=stations.copy()
                sta_train.remove(sta_test)
                #
                if True: # Verbose 
                    # get eruption times for test station
                    fl_nm = os.sep.join([datadir,sta_test+'_eruptive_periods.txt'])
                    with open(fl_nm,'r') as fp:
                        tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                    print(str(i+1)+'/'+str(len(stations)))
                    print('Training: '+str(sta_train))
                    if (no_erup and no_erup[0] in sta_train):
                        print('Eruption not in training: '+no_erup[0] +'  '+str(tes[no_erup[1]])+'   ')
                    print('Testing: '+sta_test)
                #
                root='FM_'+str(int(win))+'w_'+'-'.join(datastream)+'_'+'-'.join(sta_train)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf'
                #
                ## (1) training models with 'stations'
                # combined features matrices are saved in featdir in a folder named by the attributes 
                # DED changed overlap from 1. to 0.75
                fm0 = TrainModelCombined(stations=sta_train,window=win, overlap=0.75, dtb=dtb, dtf=dtf, datastream=datastream,
                    rootdir=rootdir,root=root,feat_dir=featdir, data_dir=datadir, tes_dir=datadir,feat_selc=fl_lt, model_dir=modeldir,
                        lab_lb=lab_lb,noise_mirror=noise_mirror, savefile_type='csv', no_erup = no_erup) # 
                # DED change to 50 classifiers and Decision Trees
                fm0.train(Nfts=20, Ncl=50, retrain=False, classifier="DT", random_seed=0, method=0.75, n_jobs=6) #20,500
                ## (2) Forecasting on test station 
                # usgin model created from 'stations' to predict on a diferent station called 'station_test'
                model_name=root # model to be use
                fm0 = ForecastTransLearn(model_name,rootdir=rootdir,root=root,datadir=datadir,
                    modeldir=modeldir,featdir=featdir,predicdir=predicdir, plotdir=plotdir, 
                        savefile_type='csv') # 
                # get eruption times for test station
                fl_nm = os.sep.join([datadir,sta_test+'_eruptive_periods.txt'])
                with open(fl_nm,'r') as fp:
                    tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                ti_forecast=tes[-2]-12*month  #'2007-09-01'   
                tf_forecast=tes[-2]+.5*month #'2007-09-01' 
                #
                ys = fm0.forecast(station_test=sta_test,ti_forecast=ti_forecast, tf_forecast=tf_forecast, 
                    recalculate=True) # generate a consensus file in predicdir folder
                #
                if stations[i] == stations[i+1]: 
                    break    # break here
                # Need to check that the model.predict in line 153 in predict_models function (forecast.py) is working fine. 

        if True: # plot results 
            pass
            import pandas as pd 
            #
            for i,sta_test in enumerate(stations):
                sta_train=stations.copy()
                sta_train.remove(sta_test)
                #
                root='FM_'+str(int(win))+'w_'+'-'.join(datastream)+'_'+'-'.join(sta_train)+'_'+str(dtb)+'dtb_'+str(dtf)+'dtf_'+sta_test
                # read concensus 
                _con = pd.read_csv(predicdir+os.sep+root+os.sep+'consensus.csv', index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0)
                _con['consensus']=_con['consensus'].rolling(6*24*2).median()
                _con.plot()  
                #
                fl_nm = os.sep.join([datadir,sta_test+'_eruptive_periods.txt'])
                with open(fl_nm,'r') as fp:
                    tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                for _tes in tes:
                    plt.axvline(x = _tes, color = 'k')#, label = 'axvline - full height')
                plt.axvline(x = _tes, color = 'k', label = 'eruption')#
                plt.title(sta_test)
                plt.show()    
      
              