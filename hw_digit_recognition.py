import cv2
import numpy as np
from PIL import Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *
import tensorflow as tf

model = tf.keras.models.load_model('mdl.h5')

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
width = 500
height = 500
white = (255, 255, 255)
bkg=(230, 230, 230)



def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=25)
    draw.line([x1, y1, x2, y2], fill="black", width=25)


def recognize():
    filename = "image.png"
    image1.save(filename)
    txt.delete('1.0', END)
    pred = testing()
    print('argmax', np.argmax(pred[0]), '\n',
          pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])
    txt.insert(tk.INSERT,
               "{}\nAccuracy: {}%".format(classes[np.argmax(pred[0])], round(pred[0][np.argmax(pred[0])] * 100, 3)))


def testing():
    img = cv2.imread('image.png', 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0

    pred = model.predict(img)
    return pred


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)


root = Tk()

root.resizable(0, 0)
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt = tk.Text(root, bd=3, exportselection=0, bg='WHITE', font='Helvetica',
              padx=10, pady=10, height=2, width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

btnModel = Button(text="Распознать", command=recognize, font=('Helvetica', '15'), bg='white')
btnClear = Button(text="Очистить", command=clear, font=('Helvetica', '12'), bg='white')
txt.pack(side=RIGHT)
btnClear.pack(side=BOTTOM)
btnModel.pack(side=BOTTOM)
root.title('Проект "Распознование цифр"')
root.configure(bg='#E6E6E6')
root.mainloop()
