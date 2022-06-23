import librosa
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



BATCH_SIZE = 64
LEARNING_RATE = 10e-4
DROPOUT_RATE = 0.1



def get_song_features(song_path):
    song_features = {}
    features_means_vars = {}
    song, sr = librosa.load(song_path, sr=44100)
    chroma_stft = librosa.feature.chroma_stft(y=song)
    song_features.update({"chroma_stft": chroma_stft})

    rms = librosa.feature.rms(y=song)
    song_features.update({"rms": rms})

    spectral_centroid = librosa.feature.spectral_centroid(y=song, sr=sr)
    song_features.update({"spectral_centroid": spectral_centroid})

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=song, sr=sr)
    song_features.update({"spectral_bandwidth": spectral_bandwidth})

    rolloff = librosa.feature.spectral_rolloff(y=song, sr=sr)
    song_features.update({"rolloff": rolloff})

    zero_crossing_rate = librosa.zero_crossings(y=song)
    song_features.update({"zero_crossing_rate": zero_crossing_rate})
    
    harmony, perceptr = librosa.effects.hpss(y=song)
    song_features.update({"harmony": harmony})
    song_features.update({"perceptr": perceptr})

    tempo = librosa.beat.tempo(y=song, sr=sr, start_bpm=170)
    features_means_vars.update({"tempo": tempo[0]})

    mfccs = librosa.feature.mfcc(y=song, sr=sr)
    for m in range(len(mfccs)):
        name = "mfcc" + str((m+1))
        song_features.update({name: mfccs[m]})
    
    for txt, ftr in song_features.items():
        mean = txt + "_mean"
        var = txt + "_var"
        features_means_vars.update({mean: np.float32(np.mean(ftr))})
        features_means_vars.update({var: np.float32(np.var(ftr))})
    
    return features_means_vars





# Zdefiniujmy klasy (z góry ustalone)
klasy = ['frenchcore', 'mainstream', 'terror', 'uptempo']

# Szybkie definiowanie ścieżek jako zmiennych
base_path = os.path.dirname(os.path.abspath(__file__))
trener_p = base_path+"\\dataset\\train\\"
trener_gen_ps = []
test_p = base_path+"\\dataset\\test\\test\\"
for k in klasy:
    trener_gen_ps.append(trener_p + "genres\\" + k)

# Wczytanie pliku .csv
df = pd.read_csv(trener_p + "\\features_30_sec.csv")
# Pozbywamy się nazwy pliku
df = df.drop(labels=["filename", "length"],axis=1)

# Wysuwamy kolumnę label, żeby ją przenieść na koniec, a następnie mieszamy
df_labels = df.pop("label") 
df["label"] = df_labels
df = df.sample(frac=1)

# Przygotowanie danych
y = df["label"]
x = StandardScaler().fit_transform(np.array(df.iloc[:,:-1], dtype=np.float32))

# Podzielenie
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Model nr.1: sieć neuronowa
input_shape = X_train.shape[1:]
model_1 = Sequential(layers=[
    Dense(512, activation='relu', input_shape=input_shape),
    Dropout(DROPOUT_RATE),
    Dense(256, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(10, activation='softmax')
])
adam = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model_1.compile(optimizer=adam,
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
model_1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=BATCH_SIZE, callbacks=[early_stop])
print("Model 1 done!")

# Model nr.2: losowy las
model_2 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0, verbose=1)
model_2.fit(X_train, y_train)
print("Model 2 done!")

# Model nr.3: klasyfikator XGBoost
model_3 = XGBClassifier(n_estimators=1000, learning_rate=LEARNING_RATE, verbosity=1, n_jobs=4)
model_3.fit(X_train, y_train)
print("Model 3 done!")

# Model nr.4: k-najbliższych sąsiadów
model_4 = KNeighborsClassifier(n_neighbors=14)
model_4.fit(X_train, y_train)
print("Model 4 done!")

y_pred1 = model_1.predict(X_test, BATCH_SIZE)
y_pred2 = model_2.predict(X_test)
y_pred3 = model_3.predict(X_test)
y_pred4 = model_4.predict(X_test)

predictions = [np.argmax(y_pred1,axis=1), y_pred2, y_pred3, y_pred4]
for pred in predictions:
    print("Classification report: ")
    print(classification_report(y_test, pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_test, pred))



model_1_plik = "model_1.sv"
pickle.dump(model_1, open(model_1_plik,'wb'))
model_2_plik = "model_2.sv"
pickle.dump(model_2, open(model_2_plik,'wb'))
model_3_plik = "model_3.sv"
pickle.dump(model_3, open(model_3_plik,'wb'))
model_4_plik = "model_4.sv"
pickle.dump(model_4, open(model_4_plik,'wb'))