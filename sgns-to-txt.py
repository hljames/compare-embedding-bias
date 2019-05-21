'''
Convert pretrained histwords embeddings to be compatible with gensim
'''

import sys
import glob, os
import numpy as np
import pickle

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python sgns-to-txt.py sgn-directory decade=all')
        sys.exit(1)
    sgn_dir = sys.argv[1]
    if sgn_dir[-1] == '/':
        sgn_dir = sgn_dir[:-1]
     # Create target Directory if doesn't exist
    outputdir = './' + sgn_dir + '-txts'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        print('Directory {} created'.format(outputdir))
    else:
        print('Directory {} already exists'.format(outputdir))
    if len(sys.argv) > 2:
        decade = sys.argv[2]
        vectors = np.load(sgn_dir + '/'+ decade + "-w.npy", mmap_mode="c")
        f = open(sgn_dir  + '/' + decade + "-vocab.pkl", "rb")
        vocab = pickle.load(f)
        word_indicies = {w:i for i,w in enumerate(vocab)}
        # embeddings = Embedding.load('../' + sgn_dir + '/' + decade)
        vocab_size = len(vocab)
        print('vocab_size: {}'.format(vocab_size))
        vector_dim = len(vectors[0])
        print('vector_dim: {}'.format(vector_dim))

        with open(outputdir + '/' + decade + '.txt', 'w') as fp:
            fp.write(str(vocab_size) + ' ' + str(vector_dim) + '\n')
            for word in vocab:
                fp.write((word + ' ' + ' '.join(map(str, (vectors[word_indicies[word], :]))) + '\n').encode('utf-8'))
    else:
        print("Changing directory to {}".format('./' + sgn_dir))
        os.chdir('./' + sgn_dir)
        print("Current directory is {}".format(os.getcwd()))
        for file in glob.glob("*.npy"):
            # get the year of the file
            d = file[:4]
            print("Loading embedding for {}".format(d))
            vectors = np.load(d + "-w.npy", mmap_mode="c")
            f = open(d + "-vocab.pkl", "rb")
            vocab = pickle.load(f)
            vocab_size = len(vocab)
            vector_dim = len(vectors[0])
            word_indicies = {w:i for i,w in enumerate(vocab)}
            output_txt = '../' + outputdir.split('/')[-1] + '/' + d + '.txt'
            print("Writing {}".format(output_txt))
            with open(output_txt, 'w') as fp:
                fp.write(str(vocab_size) + ' ' + str(vector_dim) + '\n')
                for word in vocab:
                    fp.write((word + ' ' + ' '.join(map(str, (vectors[word_indicies[word], :]))) + '\n'))
