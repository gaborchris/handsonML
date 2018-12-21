import sys
import os
import tarfile
from six.moves import urllib

#relative sys.path import for root directory of project
#must change if file moves from project/src/load_data/
if __name__ == "__main__":
    from os.path import dirname as udir
    from os.path import abspath as upath
    home_folder = udir(udir(udir(upath(__file__))))
    sys.path.append(home_folder)

import datasets


def fetch_data(url, save_dir, filename="", extract=False):
    if save_dir == "":
        print("Select a folder to save data in")
        return
    else:
       data_folder_path =  os.path.join(os.path.dirname(datasets.__file__), save_dir)
    if not os.path.isdir(data_folder_path):
        print("Making new data directory:\n" + data_folder_path)
        os.makedirs(data_folder_path)
    if filename == "":
        download_path = os.path.join(data_folder_path, os.path.basename(url))
    else:
        download_path = os.path.join(data_folder_path, filename)
    urllib.request.urlretrieve(url, download_path)
    if extract == True:
        data_tgz = tarfile.open(download_path)
        data_tgz.extractall(path=data_folder_path)
        data_tgz.close()

def dataset_path(dataset, filename):
    return os.path.join(os.path.join(os.path.dirname(datasets.__file__), dataset),filename)
        
    




if __name__ == "__main__":
    download_src = "https://raw.githubusercontent.com/ageron/handson-ml/master"
    download_src +=  "/datasets/housing/housing.tgz"
    fetch_data(download_src, "bikes", filename="frog.csv", extract=True)
    
