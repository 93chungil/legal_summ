import os
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import random
import warnings


corpus_dir = os.path.join(os.path.dirname(os.getcwd()), 'corpus')

trimmed_df = pd.read_csv(os.path.join(corpus_dir, 'fulltext.csv'))

maxlen_text = 10000
maxlen_summ = 700

test_size = 0.8

train_x, test_val_x, train_y, test_val_y = train_test_split(trimmed_df['cleaned_text'], trimmed_df['cleaned_summary'], test_size=test_size, random_state=0)
val_x, test_x, val_y, test_y = train_test_split(test_val_x, test_val_y, test_size=0.5, random_state=123)
del trimmed_df


t_tokenizer = Tokenizer()
t_tokenizer.fit_on_texts(list(train_x))

thresh = 4
count = 0
total_count = 0
frequency = 0
total_frequency = 0

for key, value in t_tokenizer.word_counts.items():
    total_count += 1
    total_frequency += value
    if value < thresh:
        count += 1
        frequency += value

print('% of rare words in vocabulary: ', (count/total_count)*100.0)
print('Total Coverage of rare words: ', (frequency/total_frequency)*100.0)
t_max_features = total_count - count
print('Text Vocab: ', t_max_features)


s_tokenizer = Tokenizer()
s_tokenizer.fit_on_texts(list(train_y))

thresh = 6
count = 0
total_count = 0
frequency = 0
total_frequency = 0

for key, value in s_tokenizer.word_counts.items():
    total_count += 1
    total_frequency += value
    if value < thresh:
        count += 1
        frequency += value
        
print('% of rare words in vocabulary: ', (count/total_count)*100.0)
print('Total Coverage of rare words: ', (frequency/total_frequency)*100.0)
s_max_features = total_count-count
print('Summary Vocab: ', s_max_features)

t_tokenizer = Tokenizer(num_words=t_max_features)
t_tokenizer.fit_on_texts(list(train_x))
train_x = t_tokenizer.texts_to_sequences(train_x)
val_x = t_tokenizer.texts_to_sequences(val_x)

train_x = pad_sequences(train_x, maxlen=maxlen_text, padding='post')
val_x = pad_sequences(val_x, maxlen=maxlen_text, padding='post')

s_tokenizer = Tokenizer(num_words=s_max_features)
s_tokenizer.fit_on_texts(list(train_y))
train_y = s_tokenizer.texts_to_sequences(train_y)
val_y = s_tokenizer.texts_to_sequences(val_y)

train_y = pad_sequences(train_y, maxlen=maxlen_summ, padding='post')
val_y = pad_sequences(val_y, maxlen=maxlen_summ, padding='post')

print("Training Sequence", train_x.shape)
print('Target Values Shape', train_y.shape)
print('Test Sequence', val_x.shape)
print('Target Test Shape', val_y.shape)

embedding_index = {}
embed_dim = 300
with open('../embeddings/glove.42B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

t_embed = np.zeros((t_max_features, embed_dim))
for word, i in t_tokenizer.word_index.items():
    vec = embedding_index.get(word)
    if i < t_max_features and vec is not None:
        t_embed[i] = vec

s_embed = np.zeros((s_max_features, embed_dim))
for word, i in s_tokenizer.word_index.items():
    vec = embedding_index.get(word)
    if i < s_max_features and vec is not None:
        s_embed[i] = vec


latent_dim = 128
filters = 64
kernel_size = 5
pool_size = 4
lstm_output_size = 200

encoder_inputs = Input(shape=(maxlen_text,), dtype='float')
encoder_embedding = Embedding(t_max_features, embed_dim, input_length=maxlen_text, 
                                weights=[t_embed], trainable=False)(encoder_inputs)
encoder_cnn = Conv1D(filters,
                    kernel_size,
                    padding='valid',
                    activation='relu',
                    strides=1)(encoder_embedding)
encoder_maxpool = MaxPooling1D(pool_size=pool_size)(encoder_cnn)
encoder_lstm = LSTM(lstm_output_size, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_maxpool)

decoder_inputs = Input(shape=(None,), dtype='int32')
decoder_embedding = Embedding(s_max_features, embed_dim, 
                                weights=[s_embed], trainable=False)(decoder_inputs)
decoder_lstm = LSTM(lstm_output_size, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

outputs = TimeDistributed(Dense(s_max_features, activation='softmax'))(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
model.fit([train_x, train_y[:, :-1]], 
            train_y.reshape(train_y.shape[0], train_y.shape[1], 1)[:, 1:], 
            epochs=10, 
            callbacks=[early_stop], 
            batch_size=128, 
            verbose=2, 
            validation_data=([val_x, val_y[:, :-1]], 
            val_y.reshape(val_y.shape[0], val_y.shape[1], 1)[:, 1:]))
# model = Sequential()
# model.add(Embedding(t_max_features, embed_dim, inputh_length=maxlen_text, weights=[t_embed], trainable=False))
# model.add(Conv1D)
# model.add(MaxPooling1D(pool_size=pool_size))
# model.add(LSTM(lstm_output_size))
# model.add(Dense(1))
# # Encoder
# enc_input = Input(shape=(maxlen_text, ))
# enc_embed = Embedding(t_max_features, embed_dim, input_length=maxlen_text, weights=[t_embed], trainable=False)(enc_input)
# # h_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# # h_out, _, _ = h_lstm(enc_embed)
# enc_lstm = Bidirectional(LSTM(latent_dim, return_state=True))
# enc_output, enc_fh, enc_fc, enc_bh, enc_bc = enc_lstm(enc_embed)
# enc_h = Concatenate(axis=-1, name='enc_h')([enc_fh, enc_bh])
# enc_c = Concatenate(axis=-1, name='enc_c')([enc_fc, enc_bc])
# #Decoder
# dec_input = Input(shape=(None, ))
# dec_embed = Embedding(s_max_features, embed_dim, weights=[s_embed], trainable=False)(dec_input)
# dec_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
# dec_outputs, _, _ = dec_lstm(dec_embed, initial_state=[enc_h, enc_c])

# dec_dense = TimeDistributed(Dense(s_max_features, activation='softmax'))
# dec_output = dec_dense(dec_outputs)

# model = Model([enc_input, dec_input], dec_output)
# model.summary()

# plot_model(
#     model,
#     to_file='./seq2seq_encoder_decoder.png',
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=96)