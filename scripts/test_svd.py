from datetime import timedelta
import os, sys
sys.path.insert(0, os.path.abspath('..'))
from puia.features import FeaturesSta


from matplotlib import pyplot as plt
import numpy as np

def main():
    feat_dir=r'U:\Research\EruptionForecasting\eruptions\features'
    tes_dir=r'U:\Research\EruptionForecasting\eruptions\data' 
    feat_sta = FeaturesSta(station='WIZ', window = 2., datastream = 'zsc2_dsarF', feat_dir=feat_dir, 
        ti='2019-12-07', tf='2019-12-10', tes_dir = tes_dir)
    feat_sta.fM=feat_sta.fM.iloc[:,:20]
    from sklearn.decomposition import PCA
    pca=PCA(n_components=5)
    pca.fit(feat_sta.fM)
    Y=pca.transform(feat_sta.fM)

    print(feat_sta)
    
if __name__ == "__main__":
    main()