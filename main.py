
from puia.tests import run_tests
from puia.model import CombinedModel

def test_multi_data_load():
    data_dir=r'U:\Research\EruptionForecasting\eruptions\data'
    # invocation types
        # one volcano, seismic data - station string
    a=CombinedModel(data='WIZ', data_dir=data_dir)
        # multiple volcanoes, seismic data - list of station strings
    b=CombinedModel(data=['WIZ','FWVZ'], data_dir=data_dir)
        # one volcano, multiple data types - dictionary: station string and list of data types
    c=CombinedModel(data={'WIZ':['seismic','inversion','dsar_template']}, data_dir=data_dir)
    pass

def main():
    #test_multi_data_load()
    pass

if __name__=='__main__':
    main()