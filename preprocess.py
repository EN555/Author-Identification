import json
import os
import pandas as pd
from tqdm import tqdm
import swifter
import re
import csv


def get_data_to_df() -> pd.DataFrame:
    """
    will save the whole data in csv format with
    ["id","text","author"] columns
    :return: result df
    """
    df = pd.DataFrame()
    data_path = "./books1/epubtxt"
    for i,filename in enumerate(next(os.walk(data_path))[2]):
        if i > 100:
            break
        with open(os.path.join(data_path,filename),"r",encoding="utf8") as file:
            text = ""
            author = ""
            title = ""
            for line in file.readlines():
                if line == "":
                    continue
                print(line.replace("\n"," "))
                if "copyright" in line.lower():
                    pass
                    # print(line)
                    # line = line.lower()
                    # line = line.split(" ")
                    # author = " ".join(line[line.index("by"):])
                elif author != "":
                    text += line
            print(author,text,title)
    # df.to_csv("data.csv")
    return df


def get_data() -> pd.DataFrame:
    # columns = ["Text", "title", "author", "publish"]
    data_path = "./books1/epubtxt"
    txt_files = pd.Series(next(os.walk(data_path))[-1])
    clean_txt_files = txt_files.swifter.apply(lambda s: re.sub(r"(\d-)|(-\d)","",s).replace('.txt','').replace(".epub","")).values
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
                book_name = book_name.replace(".txt","").replace(".epub","").split('/')[-1]
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
                    with open(os.path.join(data_path,filename),encoding="utf8") as file:
                        curr_row['Text'] = file.read()
                        rows.append(curr_row)
            except Exception as e:
                print(dict_row)
                print(e)
    df = pd.concat(rows, axis=1).transpose()
    df.to_csv("data.csv",quoting=csv.QUOTE_ALL)
    return df
