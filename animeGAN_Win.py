from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label

model = load_model('C:\\Users\\Igor\\source\\repos\\FaceGenerator\\',
                   'FaceGenerator\\animeGAN.model')


def show_face():
    noise = np.random.normal(0, 1, (1, 256))
    generated_images = model.predict(noise)

    generated_images = 0.5 * generated_images + 0.5
    im = Image.fromarray((generated_images[0]*255).astype('uint8'))

    render = ImageTk.PhotoImage(im)
    img = Label(image=render)
    img.image = render
    img.place(x=20, y=30)


root = tk.Tk()
root.geometry("173x178")
root.title("Face Generator")
frame = tk.Frame(root)
frame.pack()


button = tk.Button(frame,
                   text="Generate Face",
                   command=show_face)
button.pack(side=tk.LEFT)
root.mainloop()
