'''
Read config file for WEAT test
'''
import sys
import json
import os


def read_json_config(file_name):
    '''
    Read a set of experiment configuration parameters from a JSON file,
    and return a dictionary with those parameters.

    The JSON must two values:

    1. embeddings: list of EITHER paths to all embeddings to compare OR nested
    JSON containing embedding name then JSON mapping years to embedding paths
    (used for time series data)

    2. tests: JSON of tests including test name, followed by a JSON representing
    the particular test configuration. Each test must have X, Y, A, and B as keys.

    (3.) compare_tests: (OPTIONAL) If the experiment is a time series with multiple
    tests and multiple embeddings, indicates whether to compare embeddings
    (one graph per test) or to compare tests (one graph per emebedding).
    Defaults to false, or the latter.

    :param file_name: Name of the file containing the configuration
    :return: a dictionary with key the name of the experiment and value a dictionary representing
    '''
    with open(file_name) as json_file:
        data = json.load(json_file)

        if 'embeddings' not in data:
            print('Config must contain embedding_paths')
            sys.exit()
        elif 'tests' not in data:
            print('Config must contain tests')
            sys.exit()
        for test_name, experiment_config in data['tests'].items():
            for k in ['X','Y','A','B']:
                if k not in experiment_config:
                    print('required key ' + k + ' not found in config')
                    sys.exit()

    return data
