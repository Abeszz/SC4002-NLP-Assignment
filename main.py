import data_preparation as data
import word_embeddings

def main():
    # Data Preparation
    data.load_datasets()
    
    # Prepare the Words Embeddings
    word_embeddings.run()
    
if __name__ == '__main__':
    main()