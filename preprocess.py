import json
import os
import numpy as np
import pandas as pd
import requests
import validators


# def get_data_to_df() -> pd.DataFrame:
#     """
#     will save the whole data in csv format with
#     ["id","text","author"] columns
#     :return: result df
#     """
#     df = pd.DataFrame()
#     data_path = "./books1/epubtxt"
#     for i,filename in enumerate(next(os.walk(data_path))[2]):
#         if i > 100:
#             break
#         with open(os.path.join(data_path,filename),"r",encoding="utf8") as file:
#             text = ""
#             author = ""
#             title = ""
#             for line in file.readlines():
#                 if line == "":
#                     continue
#                 print(line.replace("\n"," "))
#                 if "copyright" in line.lower():
#                     pass
#                     # print(line)
#                     # line = line.lower()
#                     # line = line.split(" ")
#                     # author = " ".join(line[line.index("by"):])
#                 elif author != "":
#                     text += line
#             print(author,text,title)
#     # df.to_csv("data.csv")
#     return df


def get_data() -> pd.DataFrame:
    # columns = ["Text", "title", "author", "publish"]
    df = pd.DataFrame()
    with open("url_list.json", "r") as url_file:
        for line in url_file.readlines():
            dict_row = json.loads(line)
            series_row = pd.Series()
            try:
                if "txt" in dict_row and validators.url(dict_row["txt"]):
                    response = requests.get(dict_row["txt"],cookies={"S2":"S17KVRALQvzdiS3Q%2FEdhbdqQZkMGUnf657JqvIatmaeHqshUrnCEjsMSAtYoZ6njbhxMDTwf1bHnxkymMBifDlXEWLuUXxC38gB%2F%2F5%2BVpTYciNdZ4Z%2BMXel6VvHm86BnQ12tHvYIOR9ZRgBG1bdJIm%2Bqm82Xefi9RE%2FNMMebzK6OKCcB"})
                    if "text/download;" not in response.headers['content-type']:
                        continue
                    series_row["Text"] = response.text
                # elif "epub" in dict_row and validators.url(dict_row["epub"]):
                #     response = requests.get(dict_row["epub"])
                #     series_row["Text"] = response.text
                else:
                    continue
                series_row["title"] = dict_row["title"]
                series_row["author"] = dict_row["author"]
                series_row["publish"] = dict_row["publish"]  # TODO: change to date
                series_row["genres"] = " ".join(dict_row["genres"]).replace("\n", "").replace(" ", "").replace(
                    "Category:", "").split("»")
                print(series_row)
                print(dict_row["txt"])
            except Exception as e:
                print(dict_row)
                print(e)

    return df
