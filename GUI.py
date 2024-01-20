from tkinter import *
from tkinter import filedialog # Función para escoger archivo de entrada
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Utils import unet_segmenta, load_img, boxescount, MF_DFA
import numpy as np

model = unet_segmenta()
    
def plt_espectros(espectros, rx, lungs,  limits = [1.7, 3.3, 0.3, 2.05]):
    
    # Muestra radiografía
    ax1.set_title("RxT")
    ax1.imshow(rx, cmap= 'gray')
    ax1.axis("off")

    # Muestra pulmones
    ax2.set_title("Lungs")
    ax2.imshow(lungs, cmap= 'gray')
    ax2.axis("off")

    # Muestra espectro pulmón izquierdo
    ax3.clear()
    ax3.set_title("Pulmón izquierdo")
    ax3.set(xlabel = "α", ylabel = "f(α)",  xlim =(1.7, 3.3), ylim =(0.3, 2.05))
    ax3.grid(which='major', color='#DDDDDD', linewidth=0.8)
    ax3.plot(espectros["izquierdo"][0],espectros["izquierdo"][1])
    ax3.scatter(espectros["izquierdo"][0],espectros["izquierdo"][1],alpha = 0.4,s=20)
    
    # Muestra espectro pulmón derecho
    ax4.clear()
    ax4.set_title("Pulmón derecho")
    ax4.set(xlabel = "α", ylabel = "f(α)",  xlim =(1.7, 3.3), ylim =(0.3, 2.05))
    ax4.grid(which='major', color='#DDDDDD', linewidth=0.6)
    ax4.plot(espectros["derecho"][0],espectros["derecho"][1])
    ax4.scatter(espectros["derecho"][0],espectros["derecho"][1],alpha = 0.4,s=20)

    canvas.draw()
    

def elegir_imagen():
    image_path = filedialog.askopenfilename(filetypes= [
        ("image",".jpg"),
        ("image",".jpeg"),
        ("image",".png")
    ])

    if len(image_path)> 0:
        lbl_inf_image.configure(text="Leyendo la imagen ... ")
        root.update()
        try:
            
            # Lectura de la radiografía
            rx = cv2.imread(image_path, 0) # Lectura de la imagen

            # Ajuste de la imagen
            image = load_img(rx) # Preparación de la imagen

            # Segmentación de pulmones
            mask = model.predict(image, verbose = 0) # Obtención de la máscara
            lungs = np.array(image[0,:,:,0])*255 
            lungs[mask[0,:,:,0]<0.5] = 0  # Imagen con información unicamente de los pulomnes

            # Seccionamiento de información en cajas
            boxes = boxescount(lungs=lungs)  
            dic_boxes = boxes.dic_boxes() 

            # Calculo de espectros
            espectros = MF_DFA(dic_boxes)

            plt_espectros(espectros, rx, lungs)

            lbl_inf_image.configure(text=image_path)
            
        except:
            lbl_inf_image.configure(text=" Archivo no reconocido")
    else: 
        lbl_inf_image.configure(text=" Aún no se ha seleccionado una imagen")



root = Tk()


root.title("CovidClassification")

frame = Frame(root)

button = Button(root, text= "Elegir imagen", command= elegir_imagen)

lbl_1 = Label(root, text="Imagen de entrada : ")

lbl_inf_image = Label(root, text=" Aún no se ha seleccionado una imagen")

button.pack()
lbl_1.pack()
lbl_inf_image.pack()
frame.pack()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = [8,8])
canvas = FigureCanvasTkAgg(figure=fig, master=root)
canvas.get_tk_widget().pack()


root.mainloop()    # Crea interfaz

