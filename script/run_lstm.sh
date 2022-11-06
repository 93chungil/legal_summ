#! /bin/bash

echo "Glove LSTM"
python lstm.py "glove" 100

echo "Law2Vec LSTM"
python lstm.py "law2vec" 100
