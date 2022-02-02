from preprocess.preprocess import *
from preprocess.simpleStyleFeatures import *
from preprocess.contentFeatures import *


if __name__ == '__main__':
    preprocess_pipeline("../C50/C50train/", 2, aggregative_word2vec, False, data_filter=chunking, word2vec_path='../../GoogleNews-vectors-negative300.bin.gz',  embedding_size=300, aggregative_function=np.sum)