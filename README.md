# Legal Document Summarization Project for COMS 6998 NLG

1. Data, embeddings download and preprocessing

You can curl these embeddings to /embeddings folder.

Glove Embeddings: https://nlp.stanford.edu/data/glove.6B.zip

Law2Vec Embeddings: https://ia903102.us.archive.org/23/items/Law2Vec/Law2Vec.200d.txt 

You can curl this dataset and unzip to /corpus folder. Only fulltext is used. 

To pre-process the data, you can run python data_clean.py inside /scripts folder that will save pre-processed data in /corpus/fulltext.csv

2. To train the baseline Glove embeddings, you can go to /script folder and run python lstm.py "glove" 10

3. For experiment, you can change embedding_type to "law2vec" and  run python lstm.py "law2vec" 10

4. Evaluating the results are inside lstm.py