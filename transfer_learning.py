# Projeto de Transfer Learning em Python
# Utilizando o ambiente Google Colab

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Carregar o dataset de gatos e cachorros
dataset, metadata = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'].take(3000), dataset['train'].skip(3000).take(1000)

# Pré-processamento
def format_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label

train_dataset = train_dataset.map(format_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(format_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Carregar modelo pré-treinado (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Criar o modelo com Transfer Learning
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinamento
history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

# Avaliação do modelo
loss, accuracy = model.evaluate(test_dataset)
print(f'Loss: {loss}, Accuracy: {accuracy}')
