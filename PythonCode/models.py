
class Model:
    def __init__(self):
        pass

    def train(self,x_train_y_train):
        raise NotImplementedError

    def predict(self,x_val, y_val):
        raise NotImplementedError
