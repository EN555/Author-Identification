from preprocess import *



if __name__ == '__main__':
    preprocess_pipeline("../Data/C50train/", "bag of words", 0.3,
                        ("x_train_2.csv", "x_test_2.csv", "y_train_2.csv", "y_test_2.csv"))