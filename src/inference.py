import librosa
import numpy as np
import pickle

class IRMASInference:
    def __init__(self, model_path='irmas_model.pkl'):
        """Load trained model artifacts"""
        with open(model_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        self.scaler = self.artifacts['scaler']
        self.baseline_model = self.artifacts['baseline_model']
        self.hierarchical_classifiers = self.artifacts['hierarchical_classifiers']
        self.hierarchy = self.artifacts['hierarchy_structure']
        self.instruments = self.hierarchy['instruments']
    
    def extract_features(self, audio_path):
        """Extract 51D feature vector from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            # MFCC Features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral Features
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Tempo Features
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=sr)
            
            # Combine features
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [np.mean(spec_centroid), np.std(spec_centroid)],
                [np.mean(spec_rolloff), np.std(spec_rolloff)],
                [np.mean(zcr), np.std(zcr)],
                np.mean(spec_contrast, axis=1),
                chroma_mean,
                [np.mean(tempogram), np.std(tempogram)]
            ])
            
            return features
        
        except Exception as e:
            raise Exception(f"Error extracting features: {e}")
    
    def predict_baseline(self, audio_path):
        """Predict using baseline Logistic Regression"""
        features = self.extract_features(audio_path)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.baseline_model.predict(features_scaled)[0]
        probabilities = self.baseline_model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = [(self.instruments[idx], probabilities[idx]) for idx in top_3_idx]
        
        return {
            'prediction': prediction,
            'confidence': float(probabilities[self.instruments.index(prediction)]),
            'top_3': top_3
        }
    
    def predict_hierarchical(self, audio_path):
        """Predict using hierarchical model"""
        features = self.extract_features(audio_path)
        features_scaled = self.scaler.transform([features])
        
        # Root classifier prediction
        root_clf = self.hierarchical_classifiers['root']
        root_proba = root_clf.predict_proba(features_scaled)[0]
        
        # Navigate through the tree
        # This is simplified - you'll need to implement full tree traversal
        # based on your actual hierarchy structure
        
        prediction = self.baseline_model.predict(features_scaled)[0]  # Fallback
        confidence = float(max(root_proba))
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'method': 'hierarchical'
        }
    
    def predict(self, audio_path, method='baseline'):
        """Main prediction function"""
        if method == 'baseline':
            return self.predict_baseline(audio_path)
        elif method == 'hierarchical':
            return self.predict_hierarchical(audio_path)
        else:
            raise ValueError("Method must be 'baseline' or 'hierarchical'")

# Test locally
if __name__ == "__main__":
    model = IRMASInference('irmas_model.pkl')
    result = model.predict('test_audio.wav', method='baseline')
    print(result)