# Importación de librerías
from Utils import CreaCarpetas
import os
import shutil
# Genera las carpetas para el conjunto de datos organizados
dir_carpetas = CreaCarpetas("Segmentacion")

# Direcciónes de las carpetas de entrenamient, validación y evaluación
path_train = "Data_Source/COVID-QU-Ex dataset/Lung Segmentation Data/Lung Segmentation Data/Train"
path_validation = "Data_Source/COVID-QU-Ex dataset/Lung Segmentation Data/Lung Segmentation Data/Val"
path_test = "Data_Source/COVID-QU-Ex dataset/Lung Segmentation Data/Lung Segmentation Data/Test"

# Copiando las imagenes del conjunto de entranmiento

print("-------- Copia de archivos de entrenamiento --------- ")
for label in os.listdir(path_train):  # Recorre las carpetas COVID-19, Non-COVID-19 y Normal
    path_rx = os.path.join(path_train,label,"images")
    path_mask = os.path.join(path_train,label,"lung masks")
    list_img = os.listdir(path_mask)
    print(f"Copiando archivos de la carpeta: {label} ...")
    for i,image in enumerate(list_img):
        path_origen_img = os.path.join(path_rx,image)
        path_destino_img = dir_carpetas.get('PathTrain').get('Inputs')
        shutil.copy(path_origen_img,path_destino_img)
        
        path_origen_mask = os.path.join(path_mask,image)
        path_destino_mask = dir_carpetas.get('PathTrain').get('Targets')
        shutil.copy(path_origen_mask,path_destino_mask) 
print("Copia de archivos de entrenamiento finalizada \n")

# Copiando las imagenes del conjunto de validacion

print("-------- Copia de archivos de validación --------- ")
for label in os.listdir(path_validation):  # Recorre las carpetas COVID-19, Non-COVID-19 y Normal
    path_rx = os.path.join(path_validation,label,"images")
    path_mask = os.path.join(path_validation,label,"lung masks")
    list_img = os.listdir(path_mask)
    print(f"Copiando archivos de la carpeta: {label} ...")
    for i,image in enumerate(list_img):
        path_origen_img = os.path.join(path_rx,image)
        path_destino_img = dir_carpetas.get('PathValidation').get('Inputs')
        shutil.copy(path_origen_img,path_destino_img)
        
        path_origen_mask = os.path.join(path_mask,image)
        path_destino_mask = dir_carpetas.get('PathValidation').get('Targets')
        shutil.copy(path_origen_mask,path_destino_mask) 
print("Copia de archivos de validación finalizada \n")

# Copiando las imagenes del conjunto de evaluacion

print("-------- Copia de archivos de evaluación --------- ")
for label in os.listdir(path_test):  # Recorre las carpetas COVID-19, Non-COVID-19 y Normal
    path_rx = os.path.join(path_test,label,"images")
    path_mask = os.path.join(path_test,label,"lung masks")
    list_img = os.listdir(path_mask)
    print(f"Copiando archivos de la carpeta: {label} ...")
    for i,image in enumerate(list_img):
        path_origen_img = os.path.join(path_rx,image)
        path_destino_img = dir_carpetas.get('PathTest').get('Inputs')
        shutil.copy(path_origen_img,path_destino_img)
        
        path_origen_mask = os.path.join(path_mask,image)
        path_destino_mask = dir_carpetas.get('PathTest').get('Targets')
        shutil.copy(path_origen_mask,path_destino_mask)   
print("Copia de archivos de evaluación finalizada \n")

# Creación de archivo txt con la información del conjunto de datos
my_file = open("Data/Segmentacion/Data_info.txt", "w")
my_file.write('\nBase de datos original \n')

len_train = len(os.listdir(dir_carpetas.get('PathTrain').get('Inputs')))
len_validation = len(os.listdir(dir_carpetas.get('PathValidation').get('Inputs')))
len_test = len(os.listdir(dir_carpetas.get('PathTest').get('Inputs')))
len_total = len_train + len_validation + len_test

print('Cantidad de imagenes en la base de datos : ', len_total)
my_file.write(f'   Cantidad de imagenes en la base de datos : {len_total}\n')

my_file.write('\nBase de datos organizada \n')

print('Imagenes en set de entrenamiento : ',len_train)
my_file.write(f'   Imagenes en set de entrenamiento : {len_train}\n')

print('Imagenes en set de validación : ',len_validation)
my_file.write(f'   Imagenes en set de validación : {len_validation}\n')

print('Imagenes en set de evaluación : ',len_test)
my_file.write(f'   Imagenes en set de evaluación : {len_test}')
my_file.close()