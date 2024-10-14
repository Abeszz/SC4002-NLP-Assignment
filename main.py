from partone import data_preparation as data
from partone import word_embeddings


def main():
    
    # Data Preparation
    print("Loading data...")
    data.load_datasets()
    print("Data Loaded !")
    
    # Prepare the Words Embeddings
    print("Setting up word embeddings...")
    # word_embeddings.run() # Run the word embedding to generate embedding model and save it.
    word_embeddings.load()  # Use load to load the saved model.
    print("Word Embedding Completed !")

if __name__ == '__main__':
    main()