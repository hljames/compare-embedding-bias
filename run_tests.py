"""
Validate results with Caliskan Paper

USAGE: python run_tests.py path_to_google_news_corpus
"""
import weat
import read_config
import sys
import json

def replicate_caliskan(embed_path):
    print('loading caliskan embedding...')
    embedding = weat.load_embedding(embed_path)
    print('embedding loaded')
    with open('configs/caliskan.json') as config_file:
        config = json.load(config_file)
    with open('results/caliskan.json') as res_file:
        exp_results = json.load(res_file)
    for name_of_test, test_config in config['tests'].items():
        res = weat.diff_assoc(test_config['X'],test_config['Y'],test_config['A'],test_config['B'],embedding)
        print(name_of_test + ':')
        print('Result: {} Original Finding: {}\n'.format(res, exp_results[name_of_test][0]))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python run_tests.py path_to_google_news_corpus')
        sys.exit()
    if len(sys.argv) > 1:
        em_path = sys.argv[1]
        replicate_caliskan(em_path)
    print('tests complete')
