import kaggle
import os
"""
Pruba de programa para importar dataset de COVID-QU-Ex
"""

path_save = "Data_Source/COVID-QU-Ex Dataset"
path_kaggle = "anasmohammedtahir/covidqu"
os.makedirs(path_save, exist_ok=True)
kaggle.api.dataset_download_files(path_kaggle, path= path_save, unzip=True)