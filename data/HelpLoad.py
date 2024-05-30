import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_label_names():
    meta_data = unpickle('data/batches.meta')
    return [label.decode('utf-8') for label in meta_data[b'label_names']]
