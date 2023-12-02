import tkinter as tk
from tkinter import Button, Label, font
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from img_util import img_util
import os

class DigitPredictor:
    def __init__(self, root, path, confidence_counter):
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

        self.model.load_weights(path)

        self.confidence_counter = confidence_counter
        
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

        img = Image.fromarray(img_array)
        #img.save("C:/Users/calid/OneDrive/Documents/Purdue/Freshman/ENGR133/Individual Project/drawn_image.png")

        #Subsample down to 28 by 28
        img_array = img_util.subsample(img_array)
        img_array = img_util.subsample(img_array)
        img_array = img_util.subsample(img_array)
        img_array = img_util.subsample(img_array)

        #Reshape array to be compatible with model
        arr_for_prediction = img_util.reshape_for_model(img_array)

        #Get prediction tensor
        prediction = self.model.predict(arr_for_prediction)

        index = np.argmax(prediction)
        
        print(index)

        if (self.confidence_counter):
                print("Confidence: ", prediction[0, index] * 100, "%")

        self.prediction.config(text=f"{index}")

        img = Image.fromarray(img_array)
        #img.save("C:/Users/calid/OneDrive/Documents/Purdue/Freshman/ENGR133/Individual Project/drawn_image_subsampled.png")

def main():
    #Get path of weights file
    current_dir = os.getcwd()

    #Exit if input is -1
    file_name = input("Enter the name of the weights file: ")
    if (file_name == "-1"):
        return
    
    #Get absolute path
    path = os.path.join(current_dir, file_name)
    print(path[len(path) - 3])
    
    #Handle Error
    while (not os.path.exists(path) or path[(len(path) - 3):] != ".h5"):
        print("Error: File not found")
        file_name = input("Enter the name of the weights file: ")
        if (file_name == "-1"):
            return
        path = os.path.join(current_dir, file_name)
    
    #Check if user wants confidence counter
    confidence_counter = input("Do you want a confidence counter? (y/n): ")
    if (confidence_counter == "-1"):
        return
    
    #Handle Error input
    while (confidence_counter != "y" and confidence_counter != 'n'):
        print("Error. Type y or n")
        confidence_counter = input("Do you want a confidence counter? (y/n): ")
        if (confidence_counter == "-1"):
            return
    confidence_counter = confidence_counter == "y"

    root = tk.Tk()
    app = DigitPredictor(root, path, confidence_counter)
    root.mainloop()


if __name__ == "__main__":
    main()
