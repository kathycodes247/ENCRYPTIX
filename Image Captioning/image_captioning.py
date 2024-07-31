import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Example usage
image_path = 'example.jpg'  # Replace with your image path
image = load_image(image_path)

inception_model = InceptionV3(weights='imagenet')
model_new = Model(inception_model.input, inception_model.layers[-2].output)

def extract_features(image):
    feature_vector = model_new.predict(image)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector

# Example usage
features = extract_features(image)

# Replace 'path_to_model' and 'path_to_tokenizer' with actual paths
caption_model = tf.keras.models.load_model('path_to_model')
tokenizer = ...  # Load your tokenizer here
max_length = 34  # Set to the max length used during training

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Example usage
caption = generate_caption(caption_model, tokenizer, features.reshape((1, 2048)), max_length)
print("Generated Caption:", caption)

def caption_image(image_path):
    image = load_image(image_path)
    features = extract_features(image)
    caption = generate_caption(caption_model, tokenizer, features.reshape((1, 2048)), max_length)
    return caption

# Example usage
caption = caption_image('example.jpg')
print("Generated Caption:", caption)

