from IPython.display import clear_output
import keras
import cv2
import os
import numpy as np

def CreaCarpetas(name, path = "Data", validation = True, targets = True):
    '''
    Crea el siguiente conjunto de carpetas
    name
        train
            inputs
            targets
        test
            inputs
            targets
    '''
    
    path_dataset = os.path.join(path,name)
    
    # Creación de carpetas

    os.makedirs(path_dataset, exist_ok=True)
    
    path_train = os.path.join(path_dataset,'train')
    path_validation = os.path.join(path_dataset,'validation')
    path_test = os.path.join(path_dataset,'test')
    
    direcciones = {'PathTrain' : {'Inputs': os.path.join(path_train,'inputs')}, #Guarda direcciones de inputs
                'PathTest' : {'Inputs': os.path.join(path_test,'inputs')}}
    
    os.makedirs(path_train, exist_ok=True)      # Crea carpeta Train   
    os.makedirs(os.path.join(direcciones['PathTrain']['Inputs']), exist_ok=True) # Crea carpeta de inputs
    if targets == True: 
        direcciones['PathTrain']['Targets'] = os.path.join(path_train,'targets')  # Guarda dirección de targets
        os.makedirs(os.path.join(direcciones['PathTrain']['Targets']), exist_ok=True) # Crea carpeta de targets
        
    os.makedirs(path_test, exist_ok=True)      # Crea carpeta Test
    os.makedirs(os.path.join(direcciones['PathTest']['Inputs']), exist_ok=True) # Crea carpeta de Inputs
    if targets == True: 
        direcciones['PathTest']['Targets'] = os.path.join(path_test,'targets')   # Guarda dirección de targets
        os.makedirs(os.path.join(direcciones['PathTest']['Targets']), exist_ok=True) # Crea carpeta de tagets

    if validation == True:
        os.makedirs(path_validation, exist_ok=True)      # Crea carpeta Train 
        direcciones['PathValidation'] = {}
        direcciones['PathValidation']['Inputs'] = os.path.join(path_validation,'inputs')  # Guarda dirección de inputs
        os.makedirs(os.path.join(direcciones['PathValidation']['Inputs']), exist_ok=True) # Crea carpeta de inputs
        if targets == True: 
            direcciones['PathValidation']['Targets'] = os.path.join(path_validation,'targets')  # Guarda dirección de targets
            os.makedirs(os.path.join(direcciones['PathValidation']['Targets']), exist_ok=True) # Crea carpeta de targets
    
    return direcciones

class SegGen(keras.utils.Sequence):
    def __init__(self, ids, path_rx,path_mask, batch_size=5, image_size=256, umbral = True):
        self.ids = ids
        self.path_rx = path_rx
        self.path_mask = path_mask
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.umbral = umbral

    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path_rx, id_name)
        mask_path = os.path.join(self.path_mask, id_name)

        ## Reading Image
        image = cv2.imread(image_path, 0)
        try:
            image = cv2.resize(image, (self.image_size, self.image_size))
        except:
            print(id_name)

        ## Reading Masks
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        if self.umbral == True:
            ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)


        ## Normalizing
        image = image/255.0
        mask = mask/255.0

        return image, mask

    def __getitem__(self, index):

        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        image = image.reshape(image.shape[0],image.shape[1],image.shape[2],1)
        mask  = np.array(mask)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)

        return np.float32(image), np.float32(mask)

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

def print_progress(iteracion,total_iteraciones,tamano = 50):
    clear_output(wait = True)
    barra = '['
    avance = int((iteracion/total_iteraciones)*tamano)
    for N_string in range(tamano):
        if N_string < avance:
            barra += ':'
        elif N_string == avance:
            barra += '>'
        else:
            barra += '.'
    barra += f']   {iteracion + 1}/{total_iteraciones}'
    print(barra)