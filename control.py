import numpy as np
import os
import matplotlib.pyplot as plt

from functions import *
from models import *
from optimizers import *
from utils import *
from settings import *

class Image:
    def __init__(self, height: int, width: int, size: int = 28):
        self.height = height * size
        self.width = width * size
        self.size = size
        self.image = np.zeros((self.height, self.width)) - 1

    def insert_char(self, char: np.ndarray, i: int, j: int):
        self.image[i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size] = char

    def return_image(self):
        return self.image

class Accumulator:
    def __init__(self):
        self.data = []

    def append_matrix(self, matrix: np.ndarray):
        self.data.append(matrix)

    def clear_data(self):
        self.data = []

class Program:
    def __init__(self, path_models: str, max_per_line: int = 30):
        self.path_models = path_models
        self.accumulator = Accumulator()
        self.max_per_line = max_per_line

    def load_models(self, gen_digits: str, gen_letters: str):
        path_gen_digits = os.path.join(self.path_models, gen_digits)
        path_gen_letters = os.path.join(self.path_models, gen_letters)

        self.gen_digits: Generator = Model.load(path_gen_digits)
        self.gen_letters: Generator = Model.load(path_gen_letters)

    def logic(self, text: str):
        text = self.prepare_text(text)

        height = len(text)
        width = max(len(line) for line in text)
        new_image = Image(height=height, width=width, size=28)

        for i, line in enumerate(text):
            self.accumulator.clear_data()
            self.generate_images(line)
            for j, char_img in enumerate(self.accumulator.data):
                new_image.insert_char(char=char_img, i=i, j=j)

        plt.figure(figsize=(width * 0.35, height * 0.8))
        plt.imshow(new_image.return_image(), cmap="gray")
        plt.axis('off')
        plt.show()

    def prepare_text(self, raw_text: str):
        lines = []
        for paragraph in raw_text.strip().lower().split("\n"):
            words = paragraph.split(" ")
            line = ""
            for word in words:
                if len(line.replace(" ", "")) + len(word) <= self.max_per_line:
                    line += word + " "
                else:
                    lines.append(line.strip())
                    line = word + " "
            if line:
                lines.append(line.strip())
        return lines

    def generate_images(self, text: str):
        for c in text:
            if c == ' ':
                self.accumulator.append_matrix(np.zeros((28, 28)) - 1)
            elif c.isdigit():
                onehot = one_hot(np.array([[DigitsSTOI[c]]]), n_classes=len(DigitsSTOI))
                generation = self.gen_digits.generate(onehot)
                self.accumulator.append_matrix(generation.reshape(28, 28))
            elif c.isalpha():
                onehot = one_hot(np.array([[MinusLetterSTOI[c]]]), n_classes=len(MinusLetterSTOI))
                generation = self.gen_letters.generate(onehot)
                self.accumulator.append_matrix(generation.reshape(28, 28))
            else:
                self.accumulator.append_matrix(np.ones((28, 28)))