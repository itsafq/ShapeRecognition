import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog,Text,Label
import tkinter.font as tkFont
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pygame

new_model = tf.keras.models.load_model('saved_model\my_model')


batch_size = 32
img_height = 300
img_width = 300
class_names = ['circle', 'hexagon', 'octagon', 'rectangle', 'triangle']
root = tk.Tk()
root.title('SHAPE RECOGNITION!')
root.geometry("700x650")
filename ="null"
apps = []
def add_App():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",title="Select File",
    filetypes= (("all files","*.*"),("exe","*.exe")))
    apps.append(filename)
    for app in apps:
        label = tk.Label(frame,text=app,bg="cornsilk4" ) 
        label.pack() 
    return filename  



def run_App():
    global filename
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    labeloutput = tk.Label(frame,text=Output,bg="white")
    labeloutput.pack()
    


canvas = tk.Canvas(root,height=700,width=650,bg="antiquewhite3")

canvas.pack()

frame =tk.Frame(root,bg="white")
frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)


fontStyle = tkFont.Font(family="Lucida Grande",size=20)
labeltitle = Label(frame,text="SHAPE RECOGNITION" ,font= fontStyle,bg="antiquewhite3")
labeltitle.pack()

line = tk.Frame(frame, height=1, width=550, bg="grey80", relief='groove')
line.pack()

openFile = tk.Button(frame,text="Select Image",padx=10,pady=5,fg="black",bg="cornsilk1",  command=add_App  )
openFile.pack(pady=10)
RunApp = tk.Button(frame,text="Recognize Shape",padx=10,pady=5,fg="black",bg="cornsilk1",  command=run_App  )
RunApp.pack()




root.mainloop()