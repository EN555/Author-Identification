import functools
from preprocess.preprocess import *
from preprocess.complexStyleFeatures import *
from preprocess.simpleStyleFeatures import *

if __name__ == '__main__':
    preprocess_pipeline(data_path="../C50/C50train/", number_of_authors=2,
                        repesention_handler=functools.partial(combine_features,
                        [complex_style_features_extraction, simple_style_features_extraction,
                         functools.partial(pos_n_grams, n=4), functools.partial(characters_n_grams, n=2, min_df=0.1)]),
                        normalize=True, data_filter=chunking,
                        save_path="../../Data/clean/twoauthors/")
