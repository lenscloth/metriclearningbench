import pickle

with open('data/inception_v1.pkl', 'rb') as f:
    x = pickle.load(f, encoding='bytes')
    print("hello")