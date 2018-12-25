import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import __init__
import src.load_data.loader as data_loader

if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/weather_2012.csv'
    dest_folder = 'weather/'
    #data_loader.fetch_data(url, dest_folder)
    data_file = 'weather_2012.csv'
    local_path = data_loader.dataset_path(dest_folder, data_file)
    weather_2012_final = pd.read_csv(local_path, index_col='Date/Time', parse_dates=True)
    temperatures = weather_2012_final[[u'Temp (C)']].copy()
    print(temperatures.head())
    temperatures.loc[:,'Hour'] = weather_2012_final.index.hour
    print(temperatures.head())
    temperatures.groupby('Hour').aggregate(np.median).plot()
    plt.show()
    temperatures.groupby('Hour').aggregate(np.mean).plot()
    plt.show()


