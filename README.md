# IRMAS Audio Instrument Classification

Hierarchical classification of musical instruments using unsupervised and supervised learning techniques on the IRMAS dataset.

## Project Overview

This project implements a hierarchical binary classifier approach inspired by the paper "Using Unsupervised Learning to Inform a Binary Classifier Graph for Multiclassification" to classify musical instruments from audio recordings. Instead of treating all 11 instruments equally, we first discover natural groupings using K-Means clustering, then build a tree of binary classifiers based on these discovered patterns.

### Key Features

- **Unsupervised Learning**: K-Means clustering to discover natural instrument groupings
- **Hierarchical Classification**: Binary decision tree approach based on discovered patterns
- **Feature Engineering**: Comprehensive audio feature extraction (MFCC, Spectral, Chroma, Tempo)
- **Multiple Baselines**: Comparison against Logistic Regression, Random Forest, and Naive Bayes

## Dataset

**IRMAS (Instrument Recognition in Musical Audio Signals)**
- 11 instrument classes: Voice, Violin, Clarinet, Saxophone, Cello, Guitar, Trumpet, Piano, Flute, Organ, Electric Guitar
- ~6,500 training samples
- ~2,800 test samples
- Audio clips of 3 seconds each

**Download**: [IRMAS Dataset](https://www.upf.edu/web/mtg/irmas)

## Project Structure

```
IRMAS_Project/
├── notebooks/
│   ├── InstrumentRecog.ipynb
├── models/
│   ├── scaler.pkl
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   └── hierarchical_classifiers.pkl
├── results/
│   ├── baseline_results.csv
│   ├── confusion_pairs.csv
│   └── model_comparison.csv
├── visualizations/
│   ├── 01_audio_samples.png
│   ├── 02_pca_visualization.png
│   ├── 03_clustering_analysis.png
│   └── 05_baseline_comparison.png
├── data/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- Google Colab (recommended) or local Jupyter environment

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/irmas-classification.git
cd irmas-classification

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
librosa>=0.9.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

## Methodology

### Phase 1: Data Exploration
- Loaded 11 instrument classes from IRMAS dataset
- Visualized waveforms and mel spectrograms
- Analyzed audio characteristics and class distribution

### Phase 2: Feature Engineering
- Extracted 53 audio features per sample:
  - 13 MFCC coefficients (mean + std)
  - Spectral features (centroid, rolloff, contrast)
  - Zero-crossing rate
  - 12 Chroma features
- Normalized features using StandardScaler

### Phase 3: Unsupervised Learning
- Applied K-Means clustering with k=2 to k=11
- Used Silhouette Score to find optimal number of clusters
- Discovered natural instrument groupings:
  - Cluster analysis revealed instruments group by timbre and acoustic properties
  - Example: String instruments (violin, cello) clustered together
  - Percussive vs. sustained instruments separated naturally

### Phase 4: Baseline Models
Trained and evaluated three baseline models:
- **Logistic Regression (OVR)**: One-vs-Rest approach
- **Random Forest**: Ensemble of 100 decision trees
- **Naive Bayes**: Gaussian Naive Bayes classifier

### Phase 5: Hierarchical Model
- Built binary decision tree based on K-Means cluster groupings
- Root classifier: Splits instruments into two major groups
- Sub-classifiers: Further refine within each group
- Final prediction: Product of probabilities along the decision path

## Results

| Model | Train Accuracy | Test Accuracy | Training Time |
|-------|---------------|---------------|---------------|
| Logistic Regression | 0.580537 | 0.547353 | 2.005385 |
| Random Forest | 0.998881 | 0.620433 | 4.766489 |
| Naive Bayes | 0.412938 | 0.409396 | 0.011900s |
| **Hierarchical Model** |  | **0.6204** |  |

### Key Findings

- Hierarchical model achieved **X%** test accuracy, representing a **Y%** reduction in misclassification over baseline
- Most commonly confused pairs: [Instrument A ↔ Instrument B]
- K-Means clustering with k=X showed optimal silhouette score
- Natural instrument groupings aligned with acoustic properties (timbre, harmonic structure)

## Usage

### Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Upload IRMAS dataset to your Drive or download directly:
```python
# Set paths
train_path = "/content/drive/MyDrive/IRMAS_Project/IRMAS-TrainingData"
test_path = "/content/drive/MyDrive/IRMAS_Project/IRMAS-TestingData-Part1"
```

4. Run cells sequentially from Phase 0 to Phase 5

### Local Environment

1. Download IRMAS dataset
2. Update paths in notebook
3. Run Jupyter notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Reproducing Results

To reproduce our results:

1. Follow installation steps above
2. Download IRMAS dataset from official source
3. Run notebooks in order (01 → 05)
4. All random seeds are set to 42 for reproducibility
5. Feature extraction takes ~30-45 minutes for full dataset
6. Model training takes ~5-10 minutes

## Future Work

- [ ] Implement full recursive binary tree decomposition (like original paper)
- [ ] Test on other audio datasets (NSynth, UrbanSound8K)
- [ ] Deep learning approach: CNN on spectrograms
- [ ] Real-time instrument detection
- [ ] Multi-label classification (songs with multiple instruments)
- [ ] Data augmentation techniques (pitch shift, time stretch)

## References

1. **Original Paper**: [Spil, G. (2021). "Using Unsupervised Learning to Inform a Binary Classifier Graph for Multiclassification"](https://cs229.stanford.edu/proj2021spr/report2/81967616.pdf)

2. **IRMAS Dataset**: [Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. (2012). "A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals". ISMIR.](https://www.upf.edu/web/mtg/irmas)

3. **Librosa**: McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python". SciPy.

## Acknowledgments

- IRMAS dataset creators at Music Technology Group (MTG), Universitat Pompeu Fabra
- Original paper author Gabriel Spil for the hierarchical classification approach
- Stanford CS 229 course materials

## The Team
- [Aaron Thomas Mathew](https://github.com/aaronmat1905)
- [Preetham V J](https://github.com/PreethamVJ)

---
