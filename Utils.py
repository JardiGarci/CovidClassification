from IPython.display import clear_output
import keras
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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
        ## Ruta de imagen de entrada y target
        image_path = os.path.join(self.path_rx, id_name)
        mask_path = os.path.join(self.path_mask, id_name)

        ## Lectura y cambio de tamaño de la imagen 
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (self.image_size, self.image_size))

        ## Lectura y cambio de tamaño de la imagen target
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        if self.umbral == True:
            ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

        ## Normalización de las imagenes
        image = image/255.0
        mask = mask/255.0

        return image, mask

    def __getitem__(self, index):
        ## Secciona imagenes por lotes
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []

        # Recorre las imagenes de cada lote y las agrega a listas
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        # Devuelve un arreglo por cada imagen 
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
    """
    Función para mostrar el avance de un proceso
    """
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

class boxescount():
    def __init__(self, lungs):
        self.lungs = np.array(lungs, dtype = np.uint8)
        self.img_boxinlungs = np.array(self.lungs)
        self.mask = np.array(self.lungs)
        self.mask[self.mask > 0] = 1
        self.contours, hierarchy = cv2.findContours(self.mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(self.contours) > 0:
            self.filter_contornos()

    def filter_contornos(self):
        """
        Acciones de la función:
        - Eliminar objetos diferentes a los pulmones
        - Ordena los contornos de los pulmones de forma: [izquierdo, derecho]
        """
        contornos = list(self.contours)

        while True:
            p_x = []
            shape = []
            for object in contornos:
                x,y,h,w = cv2.boundingRect(object)
                p_x.append(x)
                shape.append(h*w)
            if len(contornos) <= 2:
                break
            contornos.pop(shape.index(min(shape)))

        if p_x.index(min(p_x)) == 1:
            contornos = list(reversed(contornos))

        self.contours = contornos

    
    def boxes(self,pulmon,mask,s, rect, show = False):
        """
        Recorre pixel por pixel la imagen para secciónar los pulmones en cajas de un determinado tamaño S
        """
        mask = np.array(mask)
        y , x = mask.shape
        boxes = []
        j = 0      # Con está variable recorremos la imagen verticalmente
        while ( j + s ) <= y:
            boxesinline = 0     
            for i in range(x):
                if (i+s) < x:
                    mat = mask[ j : j + s , i : i + s ]
                    if np.all(mat == 1):            # Guarda la caja si todos los elementos dentro de la caja contiene información 
                        mask[ j : j + s , i : i + s ] = 0
                        box = np.array(pulmon[ j : j + s , i : i + s ])
                        box = box - np.mean(box)                        # Media y resta (Procedimiento de metodo MDF)
                        boxes.append(box)
                        boxesinline = 1 
                        if show == True:
                            cv2.rectangle(self.img_boxinlungs, (i + rect[0] , j + rect[1] ), (i + rect[0] + s, j + rect[1] + s), (255, 0, 0), 1)              
                else:
                    break
            if boxesinline == 1:
                j += s      # Si se secciona un caja, el avance es del tamaño de la caja
            else:
                j +=1
        return np.array(boxes)

    def dic_boxes(self, limit = False):
        count = 0
        side = ['izquierdo','derecho']
        dic = {}
        for i in self.contours:      # Recorre los pulmones
            dic[side[count]] = {}       # Crea diccionario de cada pulmón
            x,y,w,h = cv2.boundingRect(i)
            rect = [x,y,w,h]
            pulmon = self.lungs[ y : y + h , x : x + w ]
            mask = self.mask[ y : y + h , x : x + w ]
            s = 6
            if limit == False:
                while True:
                    box_lung = self.boxes(pulmon, mask, s, rect)

                    if box_lung.shape[0]>3:       # Limita los tamaños de caja a aquellos en los que ambos pulmones contienen más de 3 cajas
                        dic[side[count]][str(s)] = box_lung 
                        # s += 2
                        # Bineo Logartimico
                        s = int(s * np.sqrt(np.sqrt(2)) ) + 1
                    else:
                        break
            else:
                for n in np.arange(s , limit,2):
                    box_lung = self.boxes(pulmon, mask,n)
                    if box_lung.shape[0]>0:
                        dic[side[count]][str(n)] = box_lung 
            count += 1
        return dic
    
    def show_boxes(self, s, show = True):
        """ 
        Permite mostrar gráficamente la segmentación de pulmones en cajas
        """
        self.img_boxinlungs = np.array(self.lungs)
        for n,i in enumerate(self.contours):
            x,y,w,h = cv2.boundingRect(i)
            rect = [x,y,w,h]
            pulmon = self.lungs[ y : y + h , x : x + w ]
            mask = self.mask[ y : y + h , x : x + w ]
            self.boxes(pulmon, mask, s, rect, show = True)
        if show == True:
            plt.figure()
            plt.imshow(self.img_boxinlungs, cmap="gray")
            plt.axis('off')
            plt.suptitle(f'Boxes  s = {s}')
            plt.show()
        return self.img_boxinlungs

def img2boxes(path_inputs, path_outputs, model, limite_tamano_caja = 6):
    list_img = os.listdir(path_inputs)
    len_list = len(list_img)
    n_imagen = 0
    for i,image_name in enumerate(list_img):
        print_progress(i,len_list)

        image_path = os.path.join(path_inputs, image_name)
        image = cv2.imread(image_path, 0)
        image = load_img(image=image)
        mask = model.predict(image, verbose=0)
        lungs = np.array(image[0,:,:,0])
        lungs[mask[0,:,:,0] < 0.5] = 0
        boxes = boxescount(lungs*255)

        
        if len(boxes.contours) == 2:
            dic = boxes.dic_boxes()
            if len(dic["izquierdo"].keys()) >= limite_tamano_caja and len(dic["derecho"].keys()) >= limite_tamano_caja:
                n_imagen += 1
                image_name = str(n_imagen)
                path_destino = os.path.join(path_outputs,image_name)
                np.savez(path_destino, boxes = dic)    # Guarda el diccionario

def load_img(image,image_size = 256):
    # Cambia el tamaño de la imagen al tamaño con el que está entrenada la red U-Net
    image = cv2.resize(image, (image_size, image_size))
    # Normaliza las imagenes
    image = image/255.0
    # Cambia el tamaño del arreglo para poder ser introducido a la red U-Net
    image = np.array(image, dtype='float32')
    image = image.reshape(1,image.shape[0],image.shape[1],1)
    return image

def unet_segmenta():
    "Función para importar red de segmentación de pulmones"
    path_modelo = 'Models/Sementacion/ModelSegmenta.keras'
    model = keras.models.load_model(path_modelo)
    return model


def multidim_cumsum(a):
	out = a[...,:].cumsum(-1)[...,:]
	for i in range(2,a.ndim+1):
		np.cumsum(out, axis=-i, out=out)
	return out

# Define mathematical function for curve fitting 
def func(x, y, a1, a2, a3, a4, a5, c): 
	return a1*x + a2*y + a3*x*y + a4*x*x + a5*y*y + c

def MF_DFA(dic_boxes, qmin = -5, qmax = 5, dq = 0.25):
    espectros = {}
    for side in dic_boxes.keys(): # Recorre cada lado
        # createFolder('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/')
        M=[];Q=np.arange( qmin-dq , qmax+2*dq , dq );NQ=len(Q);Mpre=[]  # Crea una lista con los valores de Q en función de qmin y qmax

        for j in range(NQ+1):
            M.append([]);Mpre.append([])

        for len_s in dic_boxes[side]: # Recorre cada tamano de caja
            # tx malla de la caja
            tx = np.array(range(int(len_s))); ty = tx; TX, TY = np.meshgrid(tx, ty) #es necesario para hacer el fit
            F=[]
            for box in range(dic_boxes[side][len_s].shape[0]): # Recorre cada caja de un determinado tamaño
                Sub_IM = dic_boxes[side][len_s][box] #renombro la caja a anlizar

                Sub_IM = Sub_IM - np.mean(Sub_IM) #restamos la media
                IM = multidim_cumsum(Sub_IM) #calculo la integracion de la superficie

                x1, y1, z1 = TX.flatten(), TY.flatten(), IM.flatten() 
                x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1
                X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  
                Y_data = z1

                reg = linear_model.LinearRegression().fit(X_data, Y_data)
                a1 = reg.coef_[0]; a2 = reg.coef_[1]; a3 = reg.coef_[2]; a4 = reg.coef_[3]; a5 = reg.coef_[4]; c = reg.intercept_

                ZZ = func(TX, TY, a1, a2, a3, a4, a5, c)
                
                F_matrix = IM-ZZ
                F.append(np.mean(F_matrix**2))

            M[0].append(np.log10(int(len_s)));Mpre[0].append(int(len_s))

            for j in range(NQ):
                if j == int(NQ/2):
                    M[j+1].append(np.log10( np.exp(0.5*np.mean(np.log(np.array(F)))) ))
                    Mpre[j+1].append(np.exp(0.5*np.mean(np.log(np.array(F)))) )
                else:
                    M[j+1].append(np.log10( np.mean(np.array(F)**(Q[j]/2.0))**(1.0/Q[j]) ))
                    Mpre[j+1].append(np.mean(np.array(F)**(Q[j]/2.0))**(1.0/Q[j]) )

        # F_qs = Mpre
        # np.savetxt('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/F_qs.txt',np.matrix(Mpre).transpose(),fmt='%s')

        MHT=[[],[],[]]; MHT[0]=Q
        for i in range(NQ):
            h=np.polyfit(M[0],M[i+1],1)[0]
            MHT[1].append(h);MHT[2].append(Q[i]*h-2)
        
        # h_tau_spectrum = MHT
        # np.savetxt('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/h_tau-spectrum.txt',np.matrix(MHT).transpose(),fmt='%s')

        Maf=[[],[]]
        for k in range(1,NQ-1):
            a=(MHT[2][k+1]-MHT[2][k-1])/(2*dq)
            Maf[0].append(a); Maf[1].append(MHT[0][k]*a-MHT[2][k])
        
        MF_spectrum = Maf
        espectros[side] = MF_spectrum
        # np.savetxt('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/MF-spectrum.txt',np.matrix(Maf).transpose(),fmt='%s')
    return espectros