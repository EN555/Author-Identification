import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from PythonCode.Constants import *
import pandas as pd
import tensorflow as tf


def get_results(y_test, prediction):
    sns.set_theme()
    ax = plt.subplot()
    mat = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(10, 7))
    sns.heatmap(mat, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.draw()
    # save as fig the classification report
    plt.figure(figsize=(10, 7))
    clf_report = classification_report(y_test, prediction, target_names=["AaronPressman", "AlanCrosby"],
                                       output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :-2].T, annot=True)
    plt.savefig("classification report.png")
    plt.draw()


def train_model(model, X_train, y_train, model_name: str):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_PART)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"./{model_name}-checkpoints",
                                                                   save_weights_only=False,
                                                                   monitor='val_accuracy', mode='max',
                                                                   save_best_only=True)
    history = model.fit(x=X_train, y=y_train, epochs=30000, shuffle=True,
                        batch_size=200, validation_data=(X_val, y_val), callbacks=[callback, model_checkpoint_callback])
    with open(f"{model_name}-history", "w") as file:
        pickle.dump(history, file)
    model.save(model_name)
    return model
