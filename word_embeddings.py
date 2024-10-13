﻿import data_preparation as data
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import constants

vocabulary = None
vocab_size = None
embedding_matrix = None
oov_words = [] # Array to track OOV words
UNKNOWN_TOKEN = '<UNKNOWN>'  # Special token for OOV words

def load():
    global embedding_matrix
    print("Loading Embedding matrix..")
    embedding_matrix = np.load(constants.embedding_matrix_path)

def run():
    # Tokenize the dataset
    print("Tokenizing dataset..")
    tokenize_dataset(data.train_dataset)

    # Print the size of Vocabulary
    print(f"1(a) Number of Vocabulary words in training dataset = {vocab_size}")
    
    # 1(c) : Add <UNKNOWN> token and replace all OOV words with it
    print("Adding UNKNOWN token to vocabulary to handle OOV words..")
    add_unkown_token()

    # Initialize the glove model with our configuration
    print("Initializing GloVe model and generating Embedding matrix..")
    glove_file_path = constants.glove_model_path
    glove_model_dimension = constants.glove_model_dimension
    create_embedding_matrix(glove_file_path, glove_model_dimension)

    # Print the number of OOV words
    print(f"1(b) Number of Out Of Vocabulary words: {len(oov_words)}")

    print("Saving Embedding matrix..")
    # Save the embedding matrix for future use
    np.save(constants.embedding_matrix_path, embedding_matrix)
    
    

def tokenize_sentence(sentence):
    # Tokenize each sentence using NLTK's word_tokenize
    return word_tokenize(sentence.lower()) # Convert to lowercase for consistency

def tokenize_dataset(dataset):
    global vocabulary, vocab_size
    # Build a vocabulary from the dataset
    vocab_counter = Counter()

    for word in dataset:
        tokens = tokenize_sentence(word['text'])
        vocab_counter.update(tokens)

    # Update the vocabulary list
    vocabulary = list(vocab_counter.keys())
    vocab_size = len(vocabulary)

def add_unkown_token():
    global vocabulary, vocab_size, UNKNOWN_TOKEN

    # Check if 'UNK' is already in the vocabulary, if not, add it
    if UNKNOWN_TOKEN not in vocabulary:
        vocabulary.append(UNKNOWN_TOKEN)
        vocab_size += 1

def create_embedding_matrix(glove_file_path, glove_model_dimension):
    global vocabulary, embedding_matrix, oov_words
    # Load the GloVe embeddings into a dictionary
    glove_embeddings = load_glove_embeddings(glove_file_path)

    # Create the embedding matrix
    embedding_matrix = np.zeros((vocab_size, glove_model_dimension))  # Initialize the matrix with zeros

    # Initialize the UNKNOWN token with a random vector
    unknown_vector = np.random.normal(scale=0.6, size=(glove_model_dimension,))
    
    # Fill the embedding matrix
    for idx, word in enumerate(vocabulary):
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]  # Use the pretrained GloVe vector
        elif word == UNKNOWN_TOKEN:
            embedding_matrix[idx] = unknown_vector  # Do not add UNKNOWN token to oov
        else:
            oov_words.append(word)  # Track OOV words
            embedding_matrix[idx] = unknown_vector

def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index