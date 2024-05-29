import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = (28, 28)

class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size, output_size)
        self.learning_rate = learning_rate

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            total_error = 0
            for image, label in training_data:
                image_flat = image.flatten()
                prediction = self.predict(image_flat)
                error = label - prediction
                gradient = -error * np.reshape(image_flat, (len(image_flat), 1))
                self.weights -= self.learning_rate * gradient
                total_error += np.abs(error)
            print(f"Epoch {epoch + 1}, Mean Error: {total_error / len(training_data)}")

symbols = ['A', 'B', 'C', 'D']
font_paths = ['Candarai.ttf', 'framd.ttf', 'ariblk.ttf', 'comicz.ttf']

train_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        image = Image.new("L", IMAGE_SIZE, color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)
        train_data.append((np.array(image), symbol_index))

test_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        other_font_path = font_paths[(font_index + 1) % len(font_paths)]
        image = Image.new("L", IMAGE_SIZE, color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(other_font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)
        test_data.append((np.array(image), symbol_index))

input_size = IMAGE_SIZE[0] * IMAGE_SIZE[1]
output_size = len(symbols)
print("Input Size =", input_size)
print("Output Size =", output_size)

learning_rate = 0.1
perceptron = Perceptron(input_size, output_size, learning_rate)

training_data = [(data.flatten(), [1 if i == label else 0 for i in range(output_size)]) for data, label in train_data]

epochs = 100
perceptron.train(training_data, epochs)

correct_predictions = 0
for image, label in test_data:
    prediction = perceptron.predict(image.flatten())
    if np.argmax(prediction) == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)
print("Model Accuracy:", accuracy)

for i, (test_image_array, symbol_index) in enumerate(test_data):
    test_image = Image.fromarray(test_image_array)
    test_image_path = f"test_image_{symbols[symbol_index]}.png"
    test_image.save(test_image_path)

    test_image.show()

random_image = np.random.randint(0, 256, size=IMAGE_SIZE)
prediction = perceptron.predict(random_image.flatten())
print("Predicted class for random image:", np.argmax(prediction))
