import json
import os
import numpy as np  
import pandas as pd 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import random
import warnings


train_frac = 0.9

corpus_dir = os.path.join(os.path.dirname(os.getcwd()), 'corpus')
with open(os.path.join(corpus_dir, 'fulltext.json'), 'r') as f:
    all_files = json.load(f)

file_list = list(all_files.keys())
random.shuffle(file_list)

train_ind = int((len(file_list) + 1) * train_frac)
train_files = file_list[:train_ind]
test_files = file_list[train_ind:]

x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts()