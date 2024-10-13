import data_preparation as data
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

vocabulary = None
vocab_size = None
embedding_matrix = None
oov_words = [] # Array to track OOV words

def run():
    # Tokenize the dataset
    tokenize_dataset(data.train_dataset)
    
    # Print the size of Vocabulary
    print(f"1(a) Vocabulary size of training dataset = {vocab_size}")
    
    # Initialize the glove model with our configuration
    glove_file_path = './GlovePreTrainedModels/glove.6B/glove.6B.300d.txt'
    embedding_dim = 300  # Specify the number of dimensions of the chosen model
    init_glove_model(glove_file_path, embedding_dim)

    # Print the number of OOV words
    print(f"1(b) Number of Out Of Vocabulary words: {len(oov_words)}")
    
    
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

def init_glove_model(glove_file_path, embedding_dim):
    global vocabulary, embedding_matrix, oov_words
    # Load the GloVe embeddings into a dictionary
    glove_embeddings = load_glove_embeddings(glove_file_path)
    
    # Create the embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))  # Initialize the matrix with zeros
    # Fill the embedding matrix
    for idx, word in enumerate(vocabulary):
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]  # Use the pretrained GloVe vector
        else:
            oov_words.append(word)  # Track OOV words
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))  # Random vector for OOV words
    
def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index