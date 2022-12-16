import os
import sys
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import csv
from rouge import Rouge

def embedding_map(embedding_type):
    if embedding_type == 'glove':
        return 'glove.6B.300d.txt', 300
    elif embedding_type == 'law2vec':
        return 'Law2Vec.200d.txt', 200

embedding_type = 'law2vec'
embedding_file, embed_dim = embedding_map(embedding_type)
epochs = 10


print("Embedding type: " + embedding_type)

corpus_dir = os.path.join(os.path.dirname(os.getcwd()), 'corpus')

trimmed_df = pd.read_csv(os.path.join(corpus_dir, 'fulltext.csv'))

maxlen_text = 10000
maxlen_summ = 700

test_size = 0.2

train_x, test_val_x, train_y, test_val_y = train_test_split(trimmed_df['cleaned_text'], trimmed_df['cleaned_summary'], test_size=test_size, random_state=0)
val_x, test_x, val_y, test_y = train_test_split(test_val_x, test_val_y, test_size=0.5, random_state=123)
del trimmed_df

og_test_x= test_x
og_test_y = test_y

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
test_x = t_tokenizer.texts_to_sequences(test_x)

train_x = pad_sequences(train_x, maxlen=maxlen_text, padding='post')
val_x = pad_sequences(val_x, maxlen=maxlen_text, padding='post')
test_x = pad_sequences(test_x, maxlen=maxlen_text, padding='post')

s_tokenizer = Tokenizer(num_words=s_max_features)
s_tokenizer.fit_on_texts(list(train_y))
train_y = s_tokenizer.texts_to_sequences(train_y)
val_y = s_tokenizer.texts_to_sequences(val_y)
test_y = s_tokenizer.texts_to_sequences(test_y)

train_y = pad_sequences(train_y, maxlen=maxlen_summ, padding='post')
val_y = pad_sequences(val_y, maxlen=maxlen_summ, padding='post')
test_y = pad_sequences(test_y, maxlen=maxlen_summ, padding='post')

print("Training Sequence", train_x.shape)
print('Target Values Shape', train_y.shape)
print('Test Sequence', val_x.shape)
print('Target Test Shape', val_y.shape)

embedding_index = {}
with open(f'../embeddings/{embedding_file}', 'r', encoding='utf-8') as f:
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


filters = 64
kernel_size = 5
pool_size = 4
lstm_output_size = 200

encoder_inputs = Input(shape=(maxlen_text,))
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

decoder_inputs = Input(shape=(None,))
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

model.save(f'./{embedding_type}_model')

model = k

enc_model = Model(inputs=encoder_inputs, outputs=[state_h, state_c])

dec_init_state_h = Input(shape=(lstm_output_size, ))
dec_init_state_c = Input(shape=(lstm_output_size, ))

dec_out, dec_h, dec_c = decoder_lstm(decoder_embedding, initial_state=[dec_init_state_h, dec_init_state_c])

dec_final = TimeDistributed(Dense(s_max_features, activation='softmax'))(dec_out)
dec_model = Model([decoder_inputs]+[dec_init_state_h, dec_init_state_c], [dec_final]+[dec_h, dec_c])

def generate_summary(input_seq):
    h, c = enc_model.predict(input_seq)
    
    next_token = np.zeros((1, 1))
    next_token[0, 0] = s_tokenizer.word_index['sostok']
    output_seq = ''
    
    stop = False
    count = 0
    
    while not stop:
        if count > 100:
            break
        decoder_out, state_h, state_c = dec_model.predict([next_token]+[h, c])
        token_idx = np.argmax(decoder_out[0, -1, :])
        
        if token_idx == s_tokenizer.word_index['eostok']:
            stop = True
        elif token_idx > 0 and token_idx != s_tokenizer.word_index['sostok']:
            token = s_tokenizer.index_word[token_idx]
            output_seq = output_seq + ' ' + token
        
        next_token = np.zeros((1, 1))
        next_token[0, 0] = token_idx
        h, c = state_h, state_c
        count += 1
        
    return output_seq

hyps = []
with open(f'./{embedding_type}_result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Article', 'Original Summary', 'Model Output'])
    for i in range(len(test_x)):
        our_summ = generate_summary(test_x[i].reshape(1, maxlen_text))
        hyps.append(our_summ)
        writer.writerow([og_test_x.iloc[i], og_test_y.iloc[i], our_summ])

rouge = Rouge()
rouge.get_scores(hyps, og_test_y, avg=True, ignore_empty=True)