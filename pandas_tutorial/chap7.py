import pandas as pd
import __init__
import matplotlib.pyplot as plt
import src.load_data.loader as data_loader
import numpy as np

if __name__ == "__main__":
    path = data_loader.dataset_path("non-emergency_calls", "311-service-requests.csv")
    na_values = ['NO CLUE', 'N/A', '0']
    requests = pd.read_csv(path, na_values=na_values, dtype={'Incident Zip': str})

    requests['Incident Zip'] = requests['Incident Zip'].str.slice(0,5)

    zero_zips = requests['Incident Zip'] == '00000'
    requests.loc[zero_zips, 'Incident Zip'] = np.nan

    unique_zips = requests['Incident Zip'].unique().astype(str)
    unique_zips.sort()
    #print(unique_zips)

    zips = requests['Incident Zip']
    is_close = zips.str.startswith('0') | zips.str.startswith('1')
    is_far = ~(is_close) & zips.notnull()

    print(zips[is_far].unique())
    print(zips[is_close].unique())

    print(requests[is_far][['Incident Zip', 'Descriptor', 'City']].sort_values('Incident Zip'))
    print(requests['City'].str.upper().value_counts())