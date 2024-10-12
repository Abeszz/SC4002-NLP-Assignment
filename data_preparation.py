from datasets import load_dataset

def load_datasets():
    dataset = load_dataset('rotten_tomatoes')
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']