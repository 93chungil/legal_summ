import os
from collections import defaultdict
import nltk
import pandas as pd
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

corpus_dir = os.path.join(os.path.dirname(os.getcwd()), 'corpus')
full_text_dir = os.path.join(corpus_dir, 'fulltext')
full_text_files = os.listdir(full_text_dir)


contractions = {"ain't": "am not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

def clean_text(text, remove_stopwords=True):
    text = text.lower()
    text = text.split()
    tmp = []
    for word in text:
        if word in contractions:
            tmp.append(contractions[word])
        else:
            tmp.append(word)
    text = ' '.join(tmp)
    
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words('english'))
        text = [w for w in text if w not in stops]
        text = ' '.join(text)
        
    return text

def parse_fulltext(ft):
    '''
        Parses fulltext file to dictionary with values as lists.
        catchphrase: no period, sentence: period when available.
    '''
    ftf = open(os.path.join(full_text_dir, ft), 'r')
    lines = ftf.readlines()
    ftf.close()

    xmldict = defaultdict(list)
    xmlstack = []
    curstuff = []
    for _, l in enumerate(lines, 1):
        if l.startswith('<'):
            if l[1] == '/':
                pop = xmlstack.pop()
                xmldict[pop].append(''.join(curstuff))
                curstuff = []
                continue
            else:
                close = l.index('>')
                name = l[1:close].split(' ')[0]
                xmlstack.append(name)
                curstuff = []
                l = l[close + 1:]
        l = l.replace('\n', '')
        try:
            end = l.index(f'</{xmlstack[-1]}>')
            curstuff.append(l[:end])
            pop = xmlstack.pop()
            xmldict[pop].append(''.join(curstuff))
            curstuff = []
        except:
            curstuff.append(l)

    return xmldict

all_files = {}
for ft in full_text_files:
    print(ft)
    all_files[ft] = parse_fulltext(ft)
    all_files[ft]['text'] = ' '.join(all_files[ft]['sentence'])
    all_files[ft]['summary'] = '. '.join(all_files[ft]['catchphrase'])
    all_files[ft]['cleaned_text'] = clean_text(all_files[ft]['text'])
    all_files[ft]['cleaned_summary'] = clean_text(all_files[ft]['summary'], remove_stopwords=False)

all_files_df = pd.DataFrame(all_files).transpose()
all_files_df['cleaned_summary'] = all_files_df['cleaned_summary'].apply(lambda x: '<sostok>' + ' ' + x + ' ' + '<eostok>')

all_files_df['text_len'] = all_files_df['cleaned_text'].str.split(' ').str.len()
all_files_df['summary_len'] = all_files_df['cleaned_summary'].str.split(' ').str.len()

import matplotlib.pyplot as plt
all_files_df.hist(bins=100)
print(all_files_df['text_len'].quantile(.95))

trimmed_df = all_files_df[all_files_df['text_len']<10000]
trimmed_df.hist(bins=100)
trimmed_df.to_csv(os.path.join(corpus_dir, 'fulltext.csv'))
