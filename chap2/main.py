import sys
import os
home_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_folder)

from src.load_data import download_housing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_standard_dist(maxsize=7):
    #standard normal plot
    numbers = (10**exp for exp in range(1, maxsize+1))
    for power in numbers:
        counts = pd.DataFrame(np.round(np.random.randn(power), 1)).loc[:,0].value_counts()
        df = pd.DataFrame(counts.values, index=counts.index)
        df = pd.DataFrame(counts.index, columns=['x'])
        df['y'] = pd.Series(counts.values)
        df.plot(kind='scatter', x='x', y='y')
        plt.title("i = " + str(power))
        plt.show()

    df = pd.DataFrame(np.random.randn(10000000))
    df.plot(kind='hist', bins=1000)
    plt.show()



if __name__ == "__main__":
    #print("About to start downloading california dataset")
    #download_housing.fetch_housing_data()

    df = pd.read_csv(os.path.join(home_folder, "datasets/housing/housing.csv"))
    print(df.head())
    df.plot(kind='scatter', x='longitude', y='latitude')
    plt.show()



    
