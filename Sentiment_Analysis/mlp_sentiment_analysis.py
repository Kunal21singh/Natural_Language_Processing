
# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import spacy
import numpy as np


# Function to preprocess text: lowercase, remove unwanted chars, space out punctuation
def preprocess_text(text):
  text = text.lower()
  text = re.sub(r"([.,!?])", r" ", text)
  text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
  return text


# Create a vocabulary list from all sentences in the dataframe
def create_vocab(df):
    vocab = []
    for i in range(len(df)):
        vocab.extend(df['Sentence'][i].split(' '))
    return list(set(vocab))

# Create a one-hot encoded vector for a sentence based on the vocabulary
def create_one_hot(vocab, text):
    split_sentence = text.split(' ')
    a = np.zeros(len(vocab))
    # Create a dictionary to map words to their indices for faster lookup
    vocab_to_index = {word: i for i, word in enumerate(vocab)}
    for word in split_sentence:
        if word in vocab_to_index:
            a[vocab_to_index[word]] = 1
    return a


# Define the Multi-Layer Perceptron model
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # First fully connected layer: input size = vocab size, output size = 100
        self.fc1 = nn.Linear(len(vocab), 100)
        # Second fully connected layer: input size = 100, output size = 3 (number of classes)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)  # Apply ReLU activation
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)  # Apply Softmax for multi-class output
        return x

# Training function
def train(device='cpu', EPOCHS=10):
    # Initialize model, loss function, and optimizer
    mlp = MultiLayerPerceptron().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # Move training data to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    last_loss = 0
    counter = 0


    # Training loop for the specified number of epochs
    for epoch in range(EPOCHS):
        training_loss = 0
        for i in range(len(X_train_tensor)):
            optimizer.zero_grad()  # Reset gradients
            x = X_train_tensor[i].unsqueeze(0)  # Add batch dimension
            y = y_train_tensor[i].unsqueeze(0)
            y_pred = mlp(x)  # Forward pass
            loss = criterion(y_pred, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            training_loss += loss.item()

        # Early stopping logic: stop if loss doesn't improve for 3 epochs
        if training_loss == last_loss:
            counter += 1
        else:
            if counter > 0:
                counter -= 1

        print(f'Epoch: {epoch+1}/{EPOCHS}, Last Loss: {last_loss}, Current Loss: {training_loss}')
        last_loss = training_loss

        if counter == 3:
            print('Early stopping')
            break
    # Save the trained model
    torch.save(mlp.state_dict(), 'mlp_model.pth')


# Testing function
def test(device='cpu'):
    # Load the trained model
    mlp = MultiLayerPerceptron().to(device)
    mlp.load_state_dict(torch.load('mlp_model.pth', map_location=device))

    with torch.no_grad():
        # Move test data to device
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test).to(device)
        y_pred = mlp(X_test_tensor)  # Get predictions
        y_pred = torch.argmax(y_pred, dim=1)  # Get predicted class
        accuracy = (y_pred == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy: {accuracy:.4f}')


# Main execution block
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data.csv')

    # Encode sentiment labels as integers
    encoder = LabelEncoder()
    df['Sentiment'] = encoder.fit_transform(df['Sentiment'])

    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Lemmatize each word in the sentence after preprocessing
    for i in range(len(df)):
        doc = nlp(preprocess_text(df['Sentence'][i]))
        lemmas = [token.lemma_ for token in doc]
        df.loc[i, 'Sentence'] = ' '.join(lemmas)

    # Build vocabulary from all sentences
    vocab = create_vocab(df)

    # Convert each sentence to a one-hot encoded vector
    df['Sentence_One_Hot'] = df['Sentence'].apply(lambda x: list(create_one_hot(vocab, x)))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['Sentence_One_Hot'], df['Sentiment'], test_size=0.3, random_state=40)

    # Convert lists to numpy arrays
    X_train = np.array(X_train.tolist())
    y_train = np.array(y_train.tolist())
    X_test = np.array(X_test.tolist())
    y_test = np.array(y_test.tolist())

    # Select device: GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train and test the model
    train(device, EPOCHS=10)
    test(device)