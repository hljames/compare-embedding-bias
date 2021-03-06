# compare-embedding-bias

Compare bias in word embeddings (over time, using different algorithms, using different corpora, before/after debiasing) using Word Embedding Association Tests (WEATs). Results are stored as JSON -- examples of graphing these results can be found in this colab notebook: 
https://colab.research.google.com/drive/1WNdOOmEenxtDhG-PRJ3K79HXBzZ-Nt-Q

The WEAT statistic was developed by Caliskan et al. https://purehost.bath.ac.uk/ws/portalfiles/portal/168480066/CaliskanEtAl_authors_full.pdf

![Compare biases in Google News vs Reddit](images/google_news_reddit.png)

![Compare biases over time](images/inst_weap_science_art.png)


# Requirements:
- Python 3 
- Gensim 
- Numpy 
- cPickle
- json
- scipy

Install with:

  $ pip install -r requirements.txt
  
## Quick Start

### Replicate Caliskan Results

1. Download the the word embedding used in the original research paper: Word2Vec Google News pretrained embeddings https://code.google.com/archive/p/word2vec/

2. Place the embedding inside a directory (EX: `embeddings`)

```
python run_tests.py embeddings/GoogleNews-vectors-negative300.bin
```

## Embeddings

### Examples of word embeddings to examine:

Word2Vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

GloVe (Common Crawl 840B): http://nlp.stanford.edu/data/glove.840B.300d.zip

GloVe (Twitter 2B): http://nlp.stanford.edu/data/glove.twitter.27B.zip

### Example of word embeddings to examine bias over time:

HistWords: https://nlp.stanford.edu/projects/histwords/

*Note:* the original histwords word embeddings are not compatible with gensim. Run sgns-to-txt to convert to a format compatible with gensim.

Example:

```
python sgns-to-txt.py embeddings/sgns-fiction
```

# Example Commands
```
python weat.py configs/histwords.json
python weat.py configs/compare_embeddings.json 
```
