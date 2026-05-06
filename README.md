
# Natural Language Processing – Sentiment Analysis with MLP

This repository contains a sentiment analysis pipeline using a Multi-Layer Perceptron (MLP) neural network implemented in PyTorch. The code demonstrates preprocessing, feature engineering, model training, and evaluation for text sentiment classification.

## Project Structure

- `Sentiment_Analysis/mlp_sentiment_analysis.py` – Main script for sentiment analysis using MLP
- `Sentiment_Analysis/data.csv` – Input dataset (CSV with `Sentence` and `Sentiment` columns)

## Features

- Text preprocessing (lowercasing, punctuation handling, lemmatization with spaCy)
- Vocabulary creation and one-hot encoding of sentences
- Label encoding for sentiment classes
- Train/test split
- Multi-Layer Perceptron (MLP) model with PyTorch
- Early stopping during training
- Model evaluation and accuracy reporting

## Requirements

- Python 3.x
- pandas
- numpy
- torch
- scikit-learn
- spacy
- spaCy English model: `en_core_web_sm`

Install dependencies:

```bash
pip install pandas numpy torch scikit-learn spacy
python -m spacy download en_core_web_sm
```

## Usage

1. Place your dataset as `Sentiment_Analysis/data.csv` with columns `Sentence` and `Sentiment`.
2. Run the script:

```bash
python Sentiment_Analysis/mlp_sentiment_analysis.py
```

The script will preprocess the data, train the MLP model, and print the test accuracy.

## Example Output

```
Epoch: 1/10, Last Loss: 0, Current Loss: ...
...
Test Accuracy: 0.85
```

## Notes

- The model uses one-hot encoding for features and lemmatized tokens for improved generalization.
- Early stopping is implemented if the loss does not improve for 3 consecutive epochs.
- The trained model is saved as `mlp_model.pth` in the working directory.

---
Feel free to extend this project with other NLP models or preprocessing techniques!
