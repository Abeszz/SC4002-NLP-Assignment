from datasets import load_dataset

train_dataset = None
validation_dataset = None
test_dataset = None

def load_datasets():
    global train_dataset, validation_dataset, test_dataset
    dataset = load_dataset('rotten_tomatoes')
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']