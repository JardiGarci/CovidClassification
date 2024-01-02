from Utils import img2boxes, unet_segmenta
import os
"""
version = 0, versión reducida (infection Segmentation Data)
version = 1, versión de datos totales (Lung Segmentation Data)
"""
version = 0

if version == 0:
    path_covid_origen = "Data_Source/COVID-QU-Ex dataset/Infection Segmentation Data/Infection Segmentation Data/Train/COVID-19/images"
    path_normal_origen = "Data_Source/COVID-QU-Ex dataset/Infection Segmentation Data/Infection Segmentation Data/Train/Normal/images"
    path_noncovid_origen = "Data_Source/COVID-QU-Ex dataset/Infection Segmentation Data/Infection Segmentation Data/Train/Non-COVID/images"
else: 
    path_covid_origen = "Data_Source\COVID-QU-Ex dataset\Lung Segmentation Data\Lung Segmentation Data\Train\COVID-19\images"
    path_normal_origen = "Data_Source\COVID-QU-Ex dataset\Lung Segmentation Data\Lung Segmentation Data\Train\Non-COVID\images"
    path_noncovid_origen = "Data_Source\COVID-QU-Ex dataset\Lung Segmentation Data\Lung Segmentation Data\Train\Normal\imagess"

path_covid_destino = "Data\\Boxes\\Train\\COVID-19"
path_normal_destino = "Data\\Boxes\\Train\\Normal"
path_noncovid_destino = "Data\\Boxes\\Train\\Non-COVID-19"


os.makedirs(path_covid_destino, exist_ok=True)
os.makedirs(path_normal_destino, exist_ok=True)
os.makedirs(path_noncovid_destino, exist_ok=True)

model = unet_segmenta()

print("Inicianado obtención de cajas ... ")
img2boxes( path_inputs = path_covid_origen , path_outputs = path_covid_destino , model = model )
print("Proceso finalizado")

print("Inicianado obtención de cajas ... ")
img2boxes( path_inputs = path_normal_origen , path_outputs = path_normal_destino , model = model )
print("Proceso finalizado")

print("Inicianado obtención de cajas ... ")
img2boxes( path_inputs = path_noncovid_origen , path_outputs = path_noncovid_destino , model = model )
print("Proceso finalizado")