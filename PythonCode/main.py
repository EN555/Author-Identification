from preprocess.preprocess import *
from preprocess.complexStyleFeatures import *


if __name__ == '__main__':
    preprocess_pipeline("../C50/C50train/", 2, complex_style_features_extraction, False, data_filter=chunking)
