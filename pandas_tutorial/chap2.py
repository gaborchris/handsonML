import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import src.load_data.loader as data_loader

if __name__ == "__main__":
    src_path = "https://raw.githubusercontent.com/jvns/pandas-cookbook/master/data/311-service-requests.csv"
    dataset_folder = "non-emergency_calls"
    print("You may need to download data if this dir is empty: "+dataset_folder)
    #load data into datasets
    #data_loader.fetch_data(src_path, dataset_folder)
    datafile_name = "311-service-requests.csv"
    saved_data_path = data_loader.dataset_path(dataset_folder, datafile_name)
    complaints = pd.read_csv(saved_data_path)
    #print(complaints.head())
    print(complaints.columns)
    #print(complaints["Borough"].value_counts())
    #top_complaints = complaints['Complaint Type'].value_counts().head(10)
    #top_complaints.plot(kind='bar')
    #plt.subplots_adjust(bottom=0.3)
    #plt.show()
    xcoord = 'X Coordinate (State Plane)'
    ycoord = 'Y Coordinate (State Plane)'
    complaints.plot(kind='scatter', x=xcoord, y=ycoord)
    plt.show()
