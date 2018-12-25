import pandas as pd
import matplotlib.pyplot as plt
import __init__
import src.load_data.loader as data_loader

if __name__ == "__main__":
    csv_path = data_loader.dataset_path("bikes", "bikes.csv")
    bikes = pd.read_csv(csv_path, sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
    berri_bikes = bikes[['Berri 1']].copy()
    berri_bikes.loc[:,'weekday'] = berri_bikes.index.weekday
    weekday_counts = berri_bikes.groupby('weekday').aggregate(sum)
    weekday_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts.plot(kind='bar')
    plt.show()

