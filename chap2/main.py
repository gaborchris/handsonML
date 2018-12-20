import sys
import os
home_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_folder)

from src.load_data import download_housing


if __name__ == "__main__":
    print("About to start downloading")
    download_housing.fetch_housing_data()
