from PIL import Image, ImageDraw, ImageFont
import numpy as np

image_size = (28, 28)

symbols = ['A', 'B', 'C', 'D']
font_paths = ['Candarai.ttf', 'framd.ttf', 'ariblk.ttf', 'comicz.ttf']

train_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        image = Image.new("L", image_size, color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)
        train_data.append((np.array(image), symbol_index))

test_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        image = Image.new("L", image_size, color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)
        test_data.append((np.array(image), symbol_index))

for i, (test_image_array, symbol_index) in enumerate(test_data):
    test_image = Image.fromarray(test_image_array)
    test_image_path = f"test_image_{symbols[symbol_index]}.png"
    test_image.save(test_image_path)


    test_image.show()

print("Количество обучающих образов:", len(train_data))
print("Количество тестовых образов:", len(test_data))
print("Количество входных сигналов =", image_size[0] * image_size[1])
print("Количество выходных сигналов =", len(symbols))

class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1, stop_criteria=0.01, weight_range=(-0.5, 0.5)):
        self.weights = np.random.uniform(*weight_range, size=(input_size, output_size))
        self.learning_rate = learning_rate
        self.stop_criteria = stop_criteria

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            total_error = 0
            for image, label in training_data:
                image_flat = image.flatten()
                prediction = self.predict(image_flat)
                error = label - prediction
                self.weights += self.learning_rate * error * np.reshape(image_flat, (len(image_flat), 1))
                total_error += np.abs(error)
            print(f"Эпоха {epoch + 1}, средняя ошибка: {total_error / len(training_data)}")

input_size = image_size[0] * image_size[1]
output_size = 4
learning_rate = 0.1
stop_criteria = 0.01
weight_range = (-0.5, 0.5)

neuron = NeuralNetwork(input_size, output_size, learning_rate, stop_criteria, weight_range)

training_data = [(data.flatten(), label) for data, label in train_data]

epochs = 100
neuron.train(training_data, epochs)

correct_predictions = 0
total_predictions = len(test_data)
for test_image_array, symbol_index in test_data:
    prediction = neuron.predict(test_image_array.flatten())
    print(f"Предсказанный символ для изображения {symbols[symbol_index]} с шрифтом {font_index}:", prediction)
    if np.argmax(prediction) == symbol_index:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("Точность модели:", accuracy)

