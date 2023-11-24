import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tkinter import font
from img_util import img_util

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Predictor")
        
        self.canvas = tk.Canvas(self.root, width=448, height=448, bg='white')
        self.canvas.pack()

        button_font = font.Font(family='Helvetica', size=36, weight='bold')

        self.clear_button = Button(self.root, text="Clear", height=2, font=button_font, command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT) 
        
        
        self.save_button = Button(self.root, text="Predict", height=2, font=button_font, command=self.save_drawing)
        self.save_button.pack(side=tk.LEFT)

        self.prediction = Label(self.root, text="0", height=2, font=button_font)
        self.prediction.pack(side=tk.LEFT)

        self.drawing = False
        self.last_x, self.last_y = None, None
        
        self.image = Image.new("L", (448, 448), 255)
        self.draw = ImageDraw.Draw(self.image)

        #Define Model Architecture
        num_classes = 10
        input_shape = (28, 28, 1)

        #The following lines are copied from online, at this website: https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial
        #I do not know how to design model architecture myself, and this architecture worked really well
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.model.load_weights('C:/Users/calid/OneDrive/Documents/Purdue/Freshman/ENGR133/Individual Project/model_weights.h5')
        
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        
    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        
    def draw_line(self, event):
        if self.drawing:
            x, y = event.x, event.y
            circle_radius = 30  # Adjust this value to change the circle size
            self.canvas.create_oval(x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius, fill='black', outline='black')
            self.draw.ellipse([x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius], fill=0, outline=0)

    def end_drawing(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (448, 448), 255)
        self.draw = ImageDraw.Draw(self.image)
        
    def save_drawing(self):
        #Get Numpy array of the image drawn
        img_array = np.array(self.image)

        #Subsample down to 28 by 28
        img_array = img_util.subsample(self, img_array)
        img_array = img_util.subsample(self, img_array)
        img_array = img_util.subsample(self, img_array)
        img_array = img_util.subsample(self, img_array)

        """
        PROBLEMS:
        Thinks 7s are 2s
        Thinks 9s are 4s
        """

        #Reshape array to be compatible with model
        arr_for_prediction = img_util.reshape_for_model(self, img_array)

        #Get prediction tensor
        prediction = self.model.predict(arr_for_prediction)

        index = np.argmax(prediction)

        print(index)
        self.prediction.config(text=f"{index}")

        img = Image.fromarray(img_array)
        img.save("C:/Users/calid/OneDrive/Documents/Purdue/Freshman/ENGR133/Individual Project/drawn_image_subsampled.png")
        
    def subsample(self, image):
        #Use array slicing to delete every other row
        subsampled_image = image[::2, ::2]

        return subsampled_image

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
