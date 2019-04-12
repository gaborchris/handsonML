import os
import datasets
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master"
HOUSING_PATH = os.path.join(os.path.dirname(datasets.__file__), "housing")
HOUSING_URL = DOWNLOAD_ROOT + "/datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    tgz_path = os.path.join(housing_path, "housing.tgz")
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


if __name__ == "__main__":
    print("Starting download...")
    print(HOUSING_URL)
    fetch_housing_data()
