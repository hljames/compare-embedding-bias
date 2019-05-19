import numpy as np
from gensim.models import KeyedVectors
import gensim
import random
import read_config
import sys
import glob
import os
import json
from gensim.models import Word2Vec
from scipy import stats
import sys
import math

def word_assoc(w,A,B,embedding):
    """
    Calculates difference in mean cosine similarity between a word and two sets
    of words.
    """
    return embedding.n_similarity([w],A) - embedding.n_similarity([w],B)

def diff_assoc(X,Y,A,B,embedding):
    """
    Caclulates the WEAT test statics for four sets of words in an embeddings
    """
    word_assoc_X = np.array(list(map(lambda x : word_assoc(x,A,B,embedding), X)))
    word_assoc_Y = np.array(list(map(lambda y : word_assoc(y,A,B,embedding), Y)))
    mean_diff = np.mean(word_assoc_X) - np.mean(word_assoc_Y)
    std = np.std(np.concatenate((word_assoc_X, word_assoc_Y), axis=0))
    return mean_diff / std

def get_bias_scores_mean_err(word_pairs,embedding):
    """
    Caculate the mean WEAT statistic and standard error using a permutation test
    on the sets of words (defaults to 100 samples)
    """
    word_pairs['X'] = list(filter(lambda x: np.count_nonzero(embedding[x]) > 0, word_pairs['X']))
    word_pairs['Y'] = list(filter(lambda x: np.count_nonzero(embedding[x]) > 0, word_pairs['Y']))
    word_pairs['A'] = list(filter(lambda x: np.count_nonzero(embedding[x]) > 0, word_pairs['A']))
    word_pairs['B'] = list(filter(lambda x: np.count_nonzero(embedding[x]) > 0, word_pairs['B']))
    subset_size_target = max((min(len(word_pairs['X']),len(word_pairs['Y'])))//2,1)
    subset_size_attr = max((min(len(word_pairs['A']),len(word_pairs['B'])))//2,1)
    bias_scores = []
    for i in range(100):
        sX = np.random.choice(word_pairs['X'],subset_size_target,replace=False)
        sY = np.random.choice(word_pairs['Y'],subset_size_target,replace=False)
        sA = np.random.choice(word_pairs['A'],subset_size_attr,replace=False)
        sB = np.random.choice(word_pairs['B'],subset_size_attr,replace=False)
        bias_scores.append(diff_assoc(sX,sY,sA,sB,embedding))
    return np.mean(bias_scores), stats.sem(bias_scores)


def run_test(config, embedding):
    word_pairs = {}
    min_len = sys.maxsize
    # Only include words that are present in the word embedding
    for word_list_name, word_list in config.items():
        if word_list_name in ['X', 'Y', 'A', 'B']:
            word_list_filtered = list(filter(lambda x: x in embedding and np.count_nonzero(embedding[x]) > 0, word_list))
            word_pairs[word_list_name] = word_list_filtered
            if len(word_list_filtered) < 2:
                print('ERROR: Words from list {} not found in embedding\n {}'.\
                format(word_list_name, word_list))
                print('All word groups must contain at least two words')
                return None, None
            else:
                print('Number of words from {} in word embeddings: {}'.\
                format(word_list_name,len(word_list_filtered)))

    return get_bias_scores_mean_err(word_pairs,embedding)

def load_embedding(embed_path):
    if embed_path.endswith('wv'):
        return KeyedVectors.load(embed_path)
    elif embed_path.endswith('txt'):
        return KeyedVectors.load_word2vec_format(embed_path, binary=False)
    elif embed_path.endswith('bin'):
        return KeyedVectors.load_word2vec_format(embed_path, binary=True)
    # NOTE reddit embedding is saved as model (no ext) + syn1neg + syn0
    else:
        return Word2Vec.load(embed_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python weat.py config.json results_file=config_results.json')
        sys.exit(1)

    fname = sys.argv[1]
    if len(sys.argv) > 2:
        results_file = sys.argv[2]
    else:
        results_file = 'results_' + fname
    results = {}
    config = read_config.read_json_config(fname)
    for e_name, e in config['embeddings'].items():
        results[e_name] = {}
        if not isinstance(e,dict):
            print('loading embedding {}...'.format(e_name))
            try:
                embedding = load_embedding(e)
            except:
                print('could not load embedding {}'.format(e_name))
                continue;
            for name_of_test, test_config in config['tests'].items():
                mean, err = run_test(test_config, embedding)
                print('mean: {} err: {}'.format(mean, err))
                if mean is not None:
                    results[e_name][name_of_test] = (round(mean, 4), round(err,4))
        else:
            print('loading time series embeddings...')
            for time, embed_path in e.items():
                results[e_name][time] = {}
                # try:
                embedding = load_embedding(embed_path)
                # except:
                    # print('could not load embedding {}'.format(e_name))
                    # continue;
                for name_of_test, test_config in config['tests'].items():
                    print(name_of_test)
                    mean, err = run_test(test_config, embedding)
                    print('mean: {} err: {}'.format(mean, err))
                    if mean is not None:
                        results[e_name][time][name_of_test] = (round(mean, 4), round(err,4))
        with open(results_file, 'wb') as outfile:
            json.dump(results, outfile)
