import functools

from preprocess.preprocess import *
from preprocess.complexStyleFeatures import *

if __name__ == '__main__':
    preprocess_pipeline(data_path="../C50/C50train/", number_of_authors=50,
                        repesention_handler=functools.partial(combine_features, 20),
                        normalize=True, data_filter=num_sentences_based_chucking)
