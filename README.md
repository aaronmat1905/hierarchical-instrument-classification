# Hierarchical Instrument Classification

## Overview
This repository contains the implementation of a hierarchical graph-based framework for musical instrument classification.  
The project integrates unsupervised clustering with supervised learning to model inter-instrument relationships and multi-level decision boundaries.  
Developed as part of the **Machine Learning Course Project (2025)**.

## Base Paper:
https://cs229.stanford.edu/proj2021spr/report2/81967616.pdf

## Abstract
Traditional instrument classification approaches treat each class independently, often overlooking structural relationships between instruments with similar timbral characteristics.  
This project explores a **hierarchical classification architecture**, where feature-based clustering is used to form higher-level groups before training supervised classifiers within each group.  
The model aims to improve recognition accuracy and interpretability in multi-class instrument recognition tasks using the **IRMAS dataset**.

## Methodology
1. **Feature Extraction**  
   - Extraction of spectral, harmonic, and timbre-based features (MFCCs, Spectral Centroid, Chroma).  
2. **Unsupervised Clustering**  
   - K-Means applied to identify latent instrument families based on audio feature similarity.  
3. **Hierarchical Classification**  
   - A two-level classifier system:
     - Level 1: Cluster-level classification (instrument family prediction).  
     - Level 2: Fine-grained classification within clusters.  
4. **Evaluation**  
   - Accuracy, F1-Score, and Confusion Matrix analyses conducted on the IRMAS dataset.

## Dataset
- **IRMAS Dataset** (Instrument Recognition in Musical Audio Signals)
- Contains annotated audio samples across 11 instrument classes.
- Data preprocessed into feature vectors for clustering and classification.

## Project Structure
<Under Development>

## Tools and Libraries
- Python 3.10+
- NumPy, Pandas, Matplotlib
- Librosa (Audio Feature Extraction)
- Scikit-learn (Clustering and ML Models)

## Results
<Under Development>

## Team Members
- **Aaron Mathew**
- **Preetham VJ**
