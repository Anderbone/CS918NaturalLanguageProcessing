import os
import pickle

if os.path.isfile('SGD.pkl'):
    with open('SGD1.pkl', 'rb') as f:
        f1 = pickle.load(f)
        print('weel done')