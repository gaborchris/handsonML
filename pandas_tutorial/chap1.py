import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import src.load_data.loader as data_loader


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (15, 10)
    bike_url ='https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/bikes.csv'
    dest_folder = 'bikes/'

    #fetch data from github
    #data_loader.fetch_data(bike_url, dest_folder)

    data_path = data_loader.dataset_path(dest_folder, 'bikes.csv')
    print("Loading data from:\n" + data_path)
    df = pd.read_csv(data_path, sep=';',encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
    print(df.head(10))

    print(df['Berri 1'].head(10))

    #df['Berri 1'].plot()
    #plt.show()
    df.plot()
    plt.show()





