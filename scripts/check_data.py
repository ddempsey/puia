from datetime import timedelta
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from puia.data import TremorData, repair_dataframe
from puia.utilities import DummyClass
from glob import glob

from matplotlib import pyplot as plt
import numpy as np

def main():
    wd = r'C:\Users\dde62\code\eruptions_data'+os.sep
    fls=glob(wd+'*_eruptive_periods.txt')
    stations = []
    eruptions = []
    total_years = 0.
    f,ax = plt.subplots(1,1)
    cnt = 0
    for fl in fls:
        station = fl.split(os.sep)[-1].split('_')[0]
        
        try:
            td=TremorData(station, data_dir=wd)
            td.df
        except ValueError:
            try:
                print('repairing {:s}'.format(station))
                repair_dataframe(wd+'{:s}_tremor_data.csv'.format(station),wd+'{:s}_tremor_data.csv'.format(station))
            except ValueError:
                print('could not repair {:s}'.format(station))
                continue
            td = TremorData(station, data_dir=wd)
        td.parent=DummyClass(data_streams=['log_zsc2_dsar'])
        td._compute_transforms()

        N = int(np.sqrt(td.df.shape[0])/2.)
        h,e = np.histogram(td.df['log_zsc2_dsar'], bins=N)
        I = np.sum(h*(e[1:]-e[:-1]))
        h=h/I
        ax.plot(0.5*(e[1:]+e[:-1]), h, '-', label=station)
        stations.append(station)
    print(stations)
    ax.legend()
    plt.show()
        
        
        # if any([t>1.e30 for t in td.df.max()]):
        #     print(station)

    #     eruptions += td.tes
    #     stations.append(station)
    #     total_years += (td.df.index[-1]-td.df.index[0]).total_seconds()/(24*3600*365.25)
    #     print(station)
    #     print(td.df.max())
    # print(stations)
    # print(len(eruptions),'eruptions,',len(stations),'volcanoes, {:d} years'.format(int(total_years)))

if __name__ == "__main__":
    main()