import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import src.load_data.loader as data_loader

if __name__=="__main__":
    src_dir_name = 'non-emergency_calls'
    datafile_name = '311-service-requests.csv'
    data_path = data_loader.dataset_path(src_dir_name, datafile_name)
    complaints = pd.read_csv(data_path)

    noise_complaints = complaints[complaints['Complaint Type'] == "Noise - Street/Sidewalk"]
    noise_complaints_counts = noise_complaints['Borough'].value_counts()
    total_complaint_counts = complaints['Borough'].value_counts()
    print(noise_complaints_counts)
    print(total_complaint_counts)
    (noise_complaints_counts/total_complaint_counts).plot(kind='bar')
    plt.show()


