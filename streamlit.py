import pathlib
from pathlib import Path

import librosa
import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *

class Predict:
    def __init__(self):
        self.learn_inference_1 = load_learner(Path()/"model_1.sv")
        self.learn_inference_2 = load_learner(Path()/"model_2.sv")
        self.learn_inference_3 = load_learner(Path()/"model_3.sv")
        self.learn_inference_4 = load_learner(Path()/"model_4.sv")
        self.song, self.sr = self.get_song_from_upload()
        if self.song is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_song_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['mp3','wav', 'flac', 'ogg'])
        if uploaded_file is not None:
            return librosa.load(uploaded_file)
        return None

    def get_song_features(song, sr):
        song_features = {}
        features_means_vars = {}
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

    def display_output(self):
        st.audio(self.song, format='audio/ogg', caption='Uploaded Song')

    def get_prediction(self):
        if st.button('Classify'):
            features = self.get_song_features(self.song, self.sr)
            feature_vals = list(features.values())
            pred, pred_idx, probs = self.learn_inference_1.predict(feature_vals)
            st.write(f'**Prediction**: {pred}')
            st.write(f'**Probability**: {probs[pred_idx]*100:.02f}%')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':
    predictor = Predict()
