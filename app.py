
from flask import Flask, render_template, request, redirect, url_for, jsonify
from keras.models import load_model
import librosa
import numpy as np
import pickle
import os

app = Flask(__name__)

model = load_model('models/emotion_model.h5')

# Explicitly compile the model to build the metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the OneHotEncoder
with open('models/one_hot_encoder.pkl', 'rb') as file:
    enc = pickle.load(file)

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def predict_emotion(audio_path):
    mfcc_features = extract_mfcc(audio_path)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=-1) # Add channel dimension
    predictions = model.predict(mfcc_features)
    predicted_label_index = np.argmax(predictions, axis=1)
    predicted_emotion = enc.categories_[0][predicted_label_index][0]
    return predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Predict emotion
    emotion = predict_emotion(file_path)

    # Return the result as JSON
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
