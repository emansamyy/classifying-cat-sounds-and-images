try:
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import numpy as np
    import os
    import IPython.display as ipd
    import cv2
    import pandas as pd
    from keras.layers import Sequential
    from tensorflow import Dense
    from tensorflow import Flatten
    from tensorflow import Sequential
    from scipy.io.wavfile import write
    import tensorflow as tf
except ImportError:
    pass



filepath = "dataset/car_extcoll0101.wav"
data, sample_rate = librosa.load(filepath)
#plt.figure(figsize=(12, 5))
#librosa.display.waveshow(data, sr=sample_rate)




import keras.models



#%matplotlib inline
 

 
def create_save_spectrogram(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
 
    dir = os.listdir(input_path)
 
    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
     
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
     
        y, sr = librosa.load(input_file)
        ms = librosa.feature.melspectrogram(y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)
        fig.savefig(output_file)
        plt.close(fig)
         
    




from glob import glob
filenameglob=glob('C:/Users/User/Desktop/Latest-GP/model2-analyticslink/dataset/*')

def stretch(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def pitchShift(file,sr):
    data = librosa.effects.pitch_shift(file, sr=sr, n_steps=4)
    return data
 

def speedTuning(file,sr):
    speed_rate = np.random.uniform(0.7,1.3)
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()

    if len(wav_speed_tune) < 16000:
        pad_len = 16000 - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                           wav_speed_tune,
                           np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
            cut_len = len(wav_speed_tune) - 16000
            wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
    
    return wav_speed_tune

################### Augmentation ########################
 ############ time shifting
for file in  filenameglob:
    wav, sr = librosa.load(file, sr=None)
    

    start_ = int(np.random.uniform(-4800,4800))
    #print('time shift: ',start_)
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), wav[:start_]]
    



    



extracted_features=[]


from scipy.io.wavfile import write



  
    
create_save_spectrogram('originalData/Angry', 'Spectrograms/Angry')
create_save_spectrogram('originalData/Defense', 'Spectrograms/Defense')
create_save_spectrogram('originalData/Fighting', 'Spectrograms/Fighting')
create_save_spectrogram('originalData/Happy', 'Spectrograms/Happy')
create_save_spectrogram('originalData/HuntingMind', 'Spectrograms/HuntingMind')
create_save_spectrogram('originalData/Mating', 'Spectrograms/Mating')
create_save_spectrogram('originalData/MotherCall', 'Spectrograms/MotherCall')
create_save_spectrogram('originalData/Paining', 'Spectrograms/Paining')
create_save_spectrogram('originalData/Resting', 'Spectrograms/Resting')
create_save_spectrogram('originalData/Warning', 'Spectrograms/Warning')

from keras.preprocessing import image
 
def load_images_in_folder(path, label):
    images = []
    labels = []
 
    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
         
    return images, labels

x = []
y = []



spectrograms, emotions = load_images_in_folder('Spectrograms/Angry', 0)

     
x += spectrograms
y += emotions

    
spectrograms, emotions = load_images_in_folder('Spectrograms/Defense', 1)

     
x += spectrograms
y += emotions

spectrograms, emotions = load_images_in_folder('Spectrograms/Fighting', 2)

     
x += spectrograms
y += emotions

spectrograms, emotions = load_images_in_folder('Spectrograms/Happy', 3)

     
x += spectrograms
y += emotions

spectrograms, emotions = load_images_in_folder('Spectrograms/HuntingMind', 4)

     
x += spectrograms
y += emotions


spectrograms, emotions = load_images_in_folder('Spectrograms/Mating', 5)

     
x += spectrograms
y += emotions

spectrograms, emotions = load_images_in_folder('Spectrograms/MotherCall', 6)

     
x += spectrograms
y += emotions

spectrograms, emotions = load_images_in_folder('Spectrograms/Paining', 7)

     
x += spectrograms
y += emotions

spectrograms, emotions= load_images_in_folder('Spectrograms/Resting', 8)

     
x += spectrograms
y += emotions

spectrograms, emotions = load_images_in_folder('Spectrograms/Warning', 9)

     
x += spectrograms
y += emotions




from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
 
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)
 
x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255
 
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
 
FeatureExtractModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
 
x_train_norm = preprocess_input(np.array(x_train))
x_test_norm = preprocess_input(np.array(x_test))
 
train_features = FeatureExtractModel.predict(x_train_norm)
test_features = FeatureExtractModel.predict(x_test_norm)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
layers.Flatten(input_shape=train_features.shape[1:]),
layers.Dense(1024, activation='relu'),
layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_features, y_train_encoded, validation_data=(test_features, y_test_encoded), batch_size=64, epochs=20)

test_accuracy=model.evaluate(test_features,y_test_encoded,verbose=0)
print(test_accuracy[1])


plt.figure(0)
plt.plot(hist.history['accuracy'], label='training accuracy')
plt.plot(hist.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()



# Loss
plt.plot(hist.history['loss'], label='training loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()





