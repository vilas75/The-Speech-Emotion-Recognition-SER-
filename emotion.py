import pandas as pd  # data Processing
import numpy as np  #no of operation,vector,numerical operation (linear algebra)
import os           # to add the file
import seaborn as sns   #visualisation of audio file  
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt # for generate pyplot
import librosa          # librosa is used for music and audio analysis 
import librosa.display   # the function constructs a plot which adaptively switches between
#a raw samples-based view of the signal (matplotlib.pyplot.step) and an amplitude envelope view of the signal 
## librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later
from IPython.display import Audio, display # used for interactive computing # to display the audio
import warnings  # import warnings library 
import pickle #a powerful tool for saving and loading Python objects in a binary format.
warnings.filterwarnings('ignore')

##Load Dataset---------------

paths = []
labels = []
for dirname, _, filenames in os.walk('Dataset'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1] # to split the file name and get the emotion par
        label = label.split('.')[0] # for taking 1st emotion on 0th index
        labels.append(label.lower())  #convert emotion to lower case
print('Dataset is loaded.')

len(paths) # length of the data set 

paths[:5]   #print 1st five entry for checking data set validation.

labels[:5]  #print emotion

# Create a DataFrame
df = pd.DataFrame() #data library
df['speech'] = paths #store speech in path
df['label'] = labels #store emotiom in lable
df.head()

df['label'].value_counts()  # count the frequency of each emotions

##Exploratory Data Analysis--------------------

df['label'] = pd.Categorical(df['label'])
# Now, you can plot the count of each category using sns.countplot
sns.countplot(data=df, x='label') #print count graph of each emotion

output_folder = 'spectrograms' # Define the folder name to save spectrograms
os.makedirs(output_folder, exist_ok=True)# Check if the folder exists, if not create it

def waveplot(data, sr, emotion):#define waveplot function to plot the waveform the emotion
    plt.figure(figsize=(10, 4)) # to fix the size of the figure
    plt.title(emotion, size=20) # to set the title of the emotion 
    librosa.display.waveshow(data, sr=sr) #  To plot the amplitude envelope of a waveform  # sr is the sampling rate 
    plt.show()
    plt.savefig(os.path.join(output_folder, f'waveplots_{emotion}.png'))  # Save in specified folder
    #plt.savefig(f'waveplot_{emotion}.png')
    plt.close()

def spectogram(data, sr, emotion):
    x = librosa.stft(data) # # to short-time fourier transform  (STFT)
     # stft represents a signal in the time-frequency domain by computing discrete fourier transform(DFT) over
    # short overlapping windows 
    xdb = librosa.amplitude_to_db(abs(x)) # this converts amplitude spectogram to dB-scaled spectrogram 
    plt.figure(figsize=(11, 4))# define the size of the figure
    plt.title(emotion, size=20) # Give the title of the emotion 
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')# display spectogram 
    plt.colorbar() # this we have done using pyplot interface 
    plt.show()
    plt.savefig(os.path.join(output_folder, f'spectogram_{emotion}.png'))  # Save in specified folder
    #plt.savefig(f'spectogram_{emotion}.png')
    plt.close()

# Visualize some waveforms and spectrograms
emotions = ['happy', 'fear', 'ps', 'angry','disgust','neutral','sad']
for emotion in emotions:

    emotion ='happy'    # assign emotion value
    path = df['speech'][df['label'] == emotion].iloc[0]
    data, sampling_rate = librosa.load(path)# load an audio file as a floating point time series 
    ## audio will be automatically resamples to given rate
    waveplot(data, sampling_rate, emotion)  #call the waveplot function
    spectogram(data, sampling_rate, emotion)# calling the spectogram function 
    Audio(path)

emotion = 'fear'
path = df['speech'][df['label']==emotion].iloc[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


emotion = 'ps'
path = df['speech'][df['label']==emotion].iloc[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'angry'
path = df['speech'][df['label']==emotion].iloc[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'disgust'
path = df['speech'][df['label']==emotion].iloc[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


emotion = 'neutral'
path = df['speech'][df['label']==emotion].iloc[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = df['speech'][df['label']==emotion].iloc[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)


##Feature Extraction------------------------

#Mel-frequency cepstral coefficients (MFCCs)
def extract_mfcc(filename): # feature extraction function
    y, sr = librosa.load(filename, duration=3, offset=0.5) #Load an audio file as a floating point time series.
    # set the default time duration of audio file as 3 sec
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    # librosa.feature.mfcc returns the mfcc sequence 
    #np.mean Compute the arithmetic mean along the specified axis 
    # n_mfcc means no of mfcc to return sr is the sampling rate
    return mfcc

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = np.array([x for x in X_mfcc])
X = np.expand_dims(X, -1)

# One-hot Encoding
from sklearn.preprocessing import OneHotEncoder #Encode categorical features as a one-hot numeric array.
enc = OneHotEncoder()       #OneHotEncoder from SciKit library only takes numerical categorical values, 
                            #hence any value of string type should be label encoded before one hot encoded.
y = enc.fit_transform(df[['label']])
#This method performs fit and transform on the input data at a single time and converts the data points. 
#If we use fit and transform separate when we need both then it will decrease the efficiency of the model so we use fit_transform() which will do both the work.
y = y.toarray()

# Save the OneHotEncoder
os.makedirs('models', exist_ok=True)
with open('models/one_hot_encoder.pkl', 'wb') as file:
    pickle.dump(enc, file)


##Create the LSTM Model----------------------

from keras.models import Sequential  # Keras is a high-level, user-friendly deep learning API written in Python.
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(123, return_sequences=False, input_shape=(40, 1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Used for multi-class classification tasks.
#(Adaptive Moment Estimation) learning rate optimization algorithm.
#Evaluates model performance during training
model.summary()


# Train the Model
history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)

# Save the trained model in the new Keras format
model.save('models/emotion_model.keras')


##Plot the Result-------------------

epochs = list(range(100))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Load the trained model
from keras.models import load_model
model = load_model('models/emotion_model.keras')

# Compile the model to ensure the metrics are configured
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the OneHotEncoder
with open('models/one_hot_encoder.pkl', 'rb') as file:
    enc = pickle.load(file)



## predict emotion--------

def predict_emotion(audio_path):    # Extract MFCC features
    
    mfcc_features = extract_mfcc(audio_path)
    # Ensure the extracted features have the correct shape
    
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=-1) # Add channel dimension

    # Predict the emotion
    predictions = model.predict(mfcc_features)

    # Get the index of the highest probability
    predicted_label_index = np.argmax(predictions, axis=1)
    
    # Decode the index to the corresponding emotion label
    predicted_emotion = enc.categories_[0][predicted_label_index][0]

    return predicted_emotion


# Trial one audio file testing
audio_file_path = 'test.wav'
emotion = predict_emotion(audio_file_path)
print(f'The predicted emotion is: {emotion}')
