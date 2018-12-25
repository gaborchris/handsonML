import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import __init__
import src.load_data.loader as data_loader

if __name__ == "__main__":
    local_path = data_loader.dataset_path("weather", "weather_2012.csv")
    weather_2012 = pd.read_csv(local_path, parse_dates=True, index_col='Date/Time')
    weather_description = weather_2012['Weather']
    is_snowing = weather_description.str.contains("Snow")
    is_snowing = is_snowing.astype(float).resample('D').apply(np.max)
    snow_totals = is_snowing.astype(float).resample('M').apply(np.sum)
    #snow_totals.plot(kind='bar')
    avg_temperature = weather_2012['Temp (C)'].resample('M').apply(np.mean)
    min_temperature = weather_2012['Temp (C)'].resample('M').apply(np.min)
    min_temperature.name = 'Min Temp'
    snow_totals.name = 'Day\'s snowed'
    avg_temperature.name = 'Average Temp'
    stats = pd.concat([snow_totals, min_temperature], axis=1)
    print(stats)
    stats.plot(kind='bar', subplots=True)
    plt.axhline(0,color='black')

    plt.subplots_adjust(bottom=0.3)
    plt.show()