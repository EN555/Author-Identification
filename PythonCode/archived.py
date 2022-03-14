import json
import os
import pandas as pd
from tqdm import tqdm
import swifter
import re
import csv
import nltk
import numpy as np
from typing import Optional
from PythonCode.Constants import *


def get_data() -> pd.DataFrame:
    # columns = ["Text", "title", "author", "publish"]
    data_path = "./books1/epubtxt"
    txt_files = pd.Series(next(os.walk(data_path))[-1])
    clean_txt_files = txt_files.swifter.apply(
        lambda s: re.sub(r"(\d-)|(-\d)", "", s).replace('.txt', '').replace(".epub", "")).values
    rows = []
    with open("url_list.json", "r") as url_file:
        for line in tqdm(url_file.readlines()):
            dict_row = json.loads(line)
            series_row = pd.Series()
            try:
                book_name = ''
                if "txt" in dict_row and dict_row["txt"] != '':
                    book_name = dict_row['txt']
                elif "epub" in dict_row and dict_row["epub"] != '':
                    book_name = dict_row['epub']
                else:
                    continue
                book_name = book_name.replace(".txt", "").replace(".epub", "").split('/')[-1]
                if book_name not in clean_txt_files:
                    continue
                filenames = txt_files[clean_txt_files == book_name]
                series_row["title"] = dict_row["title"]
                series_row["author"] = dict_row["author"]
                series_row["publish"] = dict_row["publish"]  # TODO: change to date
                series_row["genres"] = " ".join(dict_row["genres"]).replace("\n", "").replace(" ", "").replace(
                    "Category:", "").split("»")

                for filename in filenames:
                    curr_row = series_row.copy()
                    with open(os.path.join(data_path, filename), encoding="utf8") as file:
                        curr_row['Text'] = file.read()
                        rows.append(curr_row)
            except Exception as e:
                print(dict_row)
                print(e)
    df = pd.concat(rows, axis=1).transpose()
    df.to_csv("data.csv", quoting=csv.QUOTE_ALL)
    return df


def replace_name() -> pd.DataFrame:
    """
    replace author's names to numebrs
    and drop 3 not important columns
    """
    data = pd.read_csv('data.csv')
    data.drop(["publish", "genres", "title", "Unnamed: 0"], axis=1, inplace=True)
    values = [i for i in range(0, data["author"].value_counts().sum())]
    key = list(dict.fromkeys(list(data["author"])))  # remove duplicate author's name
    dic_name_id = dict(zip(key, values))
    data["author"] = data["author"].swifter.apply(lambda s: dic_name_id[s])
    return data


# average of sentence
def average_sentence_size(data) -> pd.DataFrame:
    data["sentence_len"] = data["Text"].swifter \
        .apply(lambda text: pd.Series(nltk.sent_tokenize(text)).map(lambda sent: len(nltk.word_tokenize(sent))).mean())
    return data


# number of time uses comma
def average_comma_uses(data) -> pd.DataFrame:
    data["sentence_len"] = data["Text"].swifter \
        .apply(lambda text: pd.Series(nltk.sent_tokenize(text)).map(
        lambda sent: pd.Series(sent).value_counts()[","] / len(nltk.word_tokenize(sent))).mean())
    return data


def pad_array(arr: np.ndarray, pad_size: int):
    if arr.size == pad_size:
        return arr
    elif arr.size > pad_size:
        return arr[:pad_size, ]
    return np.concatenate([arr, np.zeros(pad_size - arr.size)], dtype=np.float32)


def pad_matrix(arr: np.ndarray, max_length: int) -> Optional[np.ndarray]:
    if arr.size == 0:
        return np.zeros((max_length, MAX_SENTENCE_LENGTH), dtype=np.float32)
    if arr.shape[0] == max_length:
        return arr
    if arr.shape[0] > max_length:
        return arr[:max_length, :]
    return np.concatenate([arr, np.zeros((max_length - arr.shape[0], arr.shape[1]))], axis=0, dtype=np.float32)