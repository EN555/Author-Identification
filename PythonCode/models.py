import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from abc import ABC
import seaborn as sns
from contextlib import redirect_stdout
from sklearn.model_selection import KFold
from Constants import *
from pathlib import Path


class Model(ABC):
    def train(self, x_train, y_train,**kwargs):
        """
        don't forget to remove the labels
        :param x_train: df | numpy | tensor with the features
        :param y_train: the labels ground truth
        :return: the trained model
        """
        raise NotImplementedError

    def predict(self, x_test) -> np.ndarray:
        """
        predict x_test to it's labels
        :param x_test: df | numpy | tensor with the features
        :return: numpy array with the
        """
        raise NotImplementedError

    def pipeline(self, x_train, y_train, x_test, y_test, model_name: str, do_cross_validation=True,
                 cross_validator=KFold(n_splits=5), **kwargs):
        sns.set_theme()
        dir_name = fr"{model_name}-{datetime.datetime.now().strftime('%m-%d--%H-%M-%S')}"
        dir_name = os.path.join(REPORT_DIRECTORY_NAME,dir_name)
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name,"model_configuration.json"),"w") as file:
            json.dump(kwargs,file,indent=4, sort_keys=True)
        with open(os.path.join(dir_name, "model_output.txt"), 'w') as f:
            with redirect_stdout(f):
                self.train(x_train, y_train,**kwargs)
                prediction = self.predict(x_test)

        class_report = classification_report(y_test, prediction)
        print(class_report)
        with open(os.path.join(dir_name, "classification_report.txt"), "w") as file:
            file.write(class_report)
        mat = confusion_matrix(y_test, prediction)
        plt.figure(figsize=(10, 7))
        sns.heatmap(mat, annot=True)
        plt.savefig(os.path.join(dir_name, "confusion_matrix.png"))
        plt.draw()
        loss_values = self.get_loss()
        if loss_values is not None:
            plt.plot(loss_values)
            plt.title("Loss Over Epochs")
            plt.draw()
            plt.savefig(os.path.join(dir_name, "loss.png"))
        if do_cross_validation:
            with open(os.path.join(dir_name, f"classification_report_cross_validation.txt"), "w") as file:
                for i,(train, test) in enumerate(list(cross_validator.split(x_train))):

                    curr_x_train, curr_x_test, curr_y_train, curr_y_test = x_train[train], x_train[test], y_train[
                        train], y_train[test]
                    self.train(curr_x_train, curr_y_train)
                    prediction = self.predict(curr_x_test)
                    file.write(f"{'-'*25}{i+1}{'-'*25}\n{classification_report(curr_y_test, prediction)}\n")

    def get_loss(self) -> np.ndarray:
        """
        can be implemented to if you want the pipeline to export it
        :return: the loss array over the train iteration
        """
        pass

