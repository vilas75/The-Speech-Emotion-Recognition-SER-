import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio, display
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import tkinter as tk
from tkinter import filedialog, Label, Button

# Load Dataset
paths = []
labels = []
for dirname, _, filenames in os.walk('Dataset'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
print('Dataset is loaded.')

# Create DataFrame
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df['label'] = pd.Categorical(df['label'])

# Define Functions for Visualization
output_folder = 'spectrograms'
os.makedirs(output_folder, exist_ok=True)

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    plt.savefig(os.path.join(output_folder, f'waveplots_{emotion}.png'))
    plt.close()

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()
    plt.savefig(os.path.join(output_folder, f'spectogram_{emotion}.png'))
    plt.close()

# Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = np.array([x for x in X_mfcc])
X = np.expand_dims(X, -1)

# One-hot Encoding
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()

# Save the OneHotEncoder
os.makedirs('models', exist_ok=True)
with open('models/one_hot_encoder.pkl', 'wb') as file:
    pickle.dump(enc, file)

# Create LSTM Model
model = Sequential([
    LSTM(123, return_sequences=False, input_shape=(40, 1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the Model
history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)

# Save the trained model
model.save('models/emotion_model.keras')

# Load the trained model
model = load_model('models/emotion_model.keras')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the OneHotEncoder
with open('models/one_hot_encoder.pkl', 'rb') as file:
    enc = pickle.load(file)

# Predict Emotion
def predict_emotion(audio_path):
    mfcc_features = extract_mfcc(audio_path)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)
    predictions = model.predict(mfcc_features)
    predicted_label_index = np.argmax(predictions, axis=1)
    predicted_emotion = enc.categories_[0][predicted_label_index][0]
    return predicted_emotion

# GUI Functionality
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        emotion = predict_emotion(file_path)
        result_label.config(text=f'The predicted emotion is: {emotion}')
        play_audio(file_path)

def play_audio(file_path):
    data, sampling_rate = librosa.load(file_path)
    display(Audio(data, rate=sampling_rate))

# Create GUI
root = tk.Tk() #tkinter This creates the main window of the application, known as the root window
root.title("Emotion Recognition from Audio")

open_button = Button(root, text="Open Audio File", command=open_file)
open_button.pack(pady=60)

result_label = Label(root, text="")
result_label.pack(pady=60)

root.mainloop()
