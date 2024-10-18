import data_preparation as data
import word_embeddings
import model_training

def main():
    
    # Data Preparation
    print("Loading data...")
    data.load_datasets()
    print("Data Loaded !")
    
    # Prepare the Words Embeddings
    print("Setting up word embeddings...")
    word_embeddings.run() # Run the word embedding to generate embedding model and save it.
    # word_embeddings.load()  # Use load to load the saved model.
    print("Word Embedding Completed !")

    params_file = "hyperparams.txt"
    optimizer_file = "optimizer_params.txt"
    
    # Train the model 
    print("Training RNN model...")
    model = model_training.train_model(data.train_dataset, data.validation_dataset, word_embeddings.vocabulary, word_embeddings.embedding_matrix, params_file=params_file, optimizer_file=optimizer_file)
    print("Model Training Completed!")
    
if __name__ == '__main__':
    main()