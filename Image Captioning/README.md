# Image Captioning

This project combines computer vision and natural language processing to build an image captioning AI. The AI uses a pre-trained VGG16 model to extract features from images and a recurrent neural network (RNN) to generate captions for those images.


**Installation**

To get started, clone the repository and install the required dependencies:
```
git clone https://github.com/kathycodes247/image-captioning-ai.git 

cd image-captioning-ai

pip install -r requirements.txt
```

**Usage**

**Preprocess the Dataset**
1. Load and preprocess your dataset (e.g., COCO dataset).
2. Extract features from images using the VGG16 model.

**Feature Extraction**

```python
  from tensorflow.keras.applications import VGG16
  from tensorflow.keras.applications.vgg16 import preprocess_input
  from tensorflow.keras.preprocessing import image
  import numpy as np


# Load VGG16 model
base_model = VGG16(weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Preprocess image
def preprocess_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)

# Extract features
img_path = 'path_to_your_image.jpg'
features = preprocess_image(img_path, model)
```

**Tokenize Captions**

```
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['captions'])

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in data['captions'])

# Convert captions to sequences
sequences = tokenizer.texts_to_sequences(data['captions'])
sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['captions'])

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in data['captions'])

# Convert captions to sequences
sequences = tokenizer.texts_to_sequences(data['captions'])
sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
```

**Model Architecture**

Define the image captioning model combining extracted features and captions.
```
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout, add
from tensorflow.keras.models import Model

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

model = define_model(vocab_size, max_length)
model.summary()
```

**Training**

Train the model using the image features and preprocessed captions.
```
def data_generator(features, captions, tokenizer, max_length):
    while True:
        for img_id, caption in zip(features.keys(), captions):
            img_feature = features[img_id]
            caption_seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(caption_seq)):
                in_seq, out_seq = caption_seq[:i], caption_seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                yield ([img_feature, in_seq], out_seq)

# Create generators
train_generator = data_generator(features, data['captions'], tokenizer, max_length)

# Fit model
model.fit(train_generator, epochs=20, steps_per_epoch=len(data['captions']), verbose=1)
```

**Generating Captions**

Use the trained model to generate captions for new images.
```   
def generate_caption(model, tokenizer, img_feature, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([img_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Load and preprocess a new image
img_path = 'path_to_new_image.jpg'
img_feature = preprocess_image(img_path, model)

# Generate caption
caption = generate_caption(model, tokenizer, img_feature, max_length)
print("Generated Caption:", caption)
```

**Contributing**
If you'd like to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature-name).
3. Make your changes and commit them (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature-name).
5. Open a pull request.

**Contact**

For any questions or suggestions, feel free to reach out:

@GitHub: kathycodes247

