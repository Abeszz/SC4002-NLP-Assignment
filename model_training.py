import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from nltk.tokenize import word_tokenize
import data_preparation as data
import json


# Custom collate_fn to handle padding
def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [torch.tensor(x) for x in inputs]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)
    return padded_inputs, labels

class SentimentRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, num_layers=1):
        super(SentimentRNN, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # Embedding layer using pre-trained embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # freeze embeddings

        # Simple RNN layer, 1 layer, uni-directional
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)  # Uni-directional RNN with 1 layer

        # RNN layer
        # self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        # self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        # self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Double hidden size for bidirectional RNN

        # Regularization layers
        # self.dropout = nn.Dropout(0.3)  # dropout rate
        # self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # Add batch normalization
        # self.layer_norm = nn.LayerNorm(hidden_size * 2)  # Add layer normalization

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded) # for rnn and gru only two values are returned
        final_output = hidden[-1] # Use the last hidden state for uni-directional RNN
        # output, (hidden, _) = self.rnn(embedded) # for lstm
        # final_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # Concatenate forward and backward hidden states
        # final_output = self.batch_norm(final_output)  # Apply batch normalization
        # final_output = self.layer_norm(final_output)  # Apply layer normalization
        final_output = self.dropout(final_output)     # Apply dropout
        return self.fc(final_output)

# Dataset class to handle data batching
class TextDataset(Dataset):
    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        tokens = word_tokenize(sentence.lower())
        indices = [self.vocab.index(token) if token in self.vocab else self.vocab.index("<UNKNOWN>") for token in tokens]
        return torch.tensor(indices), torch.tensor(label)

def get_optimizer(params, model):
    optimizer_type = params["optimizer_type"]
    lr = params["learning_rate"]
    weight_decay = params.get("weight_decay", 0)  # Default to 0 if not specified

    if optimizer_type == "SGD":
        momentum = params.get("momentum", 0)  # Default to 0 if not specified
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def train_model(train_data, valid_data, vocab, embedding_matrix, params_file="hyperparams.txt",  optimizer_file="optimizer_params.txt"):

    # Read hyperparameters from file
    with open(params_file, "r") as f:
        params = json.load(f)

    # Load optimizer parameters
    with open(optimizer_file, "r") as f:
        optimizer_params = json.load(f)
    output_size = params['output_size']  # Binary classification (positive/negative)
    hidden_size = params['hidden_size']
    batch_size = params['batch_size']
    epochs = params['epochs']
    lr = params['learning_rate']
    patience = params['patience']
    model = SentimentRNN(embedding_matrix, hidden_size, output_size)


    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)  # Use weight decay

    # Get the optimizer dynamically based on the configuration in optimizer_file
    optimizer = get_optimizer(optimizer_params, model)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # StepLR for learning rate scheduling

    # Create DataLoaders for training and validation with custom collate_fn for padding
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(TextDataset(valid_data, vocab), batch_size=batch_size, collate_fn=collate_fn)

    best_val_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

        # Early stopping logic: check if validation accuracy improved
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            epochs_no_improve = 0  # Reset the patience counter
            # You can optionally save the best model here
            torch.save(model.state_dict(), 'best_model.pt')  # Save the best model's weights
        else:
            epochs_no_improve += 1

        scheduler.step()  # Step the learning rate scheduler

        # If validation accuracy hasn't improved for 'patience' epochs, stop training
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        # Call the test set evaluation after training is done
        print("Evaluating model on test set...")
        test_accuracy = evaluate_model_on_test(model, data.test_dataset, vocab)

    return model

def evaluate_model_on_test(model, test_data, vocab):
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(TextDataset(test_data, vocab), batch_size=32, collate_fn=collate_fn)

    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print the test accuracy
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy}%')
    return test_accuracy