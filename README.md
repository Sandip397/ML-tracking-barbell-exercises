# Tracking Barbell Exercises with Machine Learning

## Project Goal

The goal of this project is to develop a machine learning pipeline to track and analyze barbell exercises using sensor data from a MetaMotion device. The project aims to:

1.  **Count the number of repetitions** for different exercises (bench press, squat, row, overhead press, and deadlift).
2.  **Classify the type of exercise** being performed.

## Data

This project uses sensor data (accelerometer and gyroscope) from a MetaMotion device. The data is collected for five different exercises:

*   Bench Press
*   Squat
*   Row
*   Overhead Press (OHP)
*   Deadlift

The raw data is stored in the `data/raw` directory and is processed and transformed into a feature-engineered dataset in the `data/interim` directory.

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Processing:** The raw sensor data is cleaned, processed, and resampled to a consistent frequency.
2.  **Feature Engineering:** A variety of features are engineered from the sensor data, including:
    *   Time-domain features (mean, standard deviation)
    *   Frequency-domain features (dominant frequency, power spectral entropy)
    *   Principal Component Analysis (PCA) features
    *   Cluster-based features
3.  **Repetition Counting:** A peak detection algorithm is used to count the number of repetitions for each exercise set.
4.  **Exercise Classification:** Several machine learning models are trained and evaluated to classify the type of exercise being performed. The best-performing model is a Random Forest classifier.

## Installation

To run this project, you'll need to have Python 3.8 or higher installed. You can install the required dependencies using conda:

```bash
conda env create -f environment.yml
```

## Usage

To run the entire pipeline, you can execute the following scripts in order:

1.  `src/data/make_dataset.py`: Processes the raw data.
2.  `src/features/build_features.py`: Creates features from the processed data.
3.  `src/models/train_model.py`: Trains and evaluates the machine learning models.

## Results

The Random Forest classifier achieved the highest accuracy of **XX.X%** on the test set. The model was able to accurately classify the different barbell exercises. The repetition counting algorithm achieved a mean absolute error of **X.X** repetitions.

## Future Work

*   Implement the `predict_model.py` script to make predictions on new, unseen data.
*   Experiment with different machine learning models and feature engineering techniques to improve performance.
*   Develop a real-time application that can track and analyze barbell exercises in real-time.
