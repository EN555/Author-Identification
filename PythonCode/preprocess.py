import os
import pandas as pd
import swifter


def load_data(path: str) -> pd.DataFrame:
    rows_list = []
    _, authors, _ = next(os.walk(path))
    for author_name in authors:
        curr_row = {"author_name":author_name}
        author_path = os.path.join(path, author_name)
        _, _, books_files = next(os.walk(author_path))
        for book_name in books_files:
            curr_row["book_name"] = book_name
            with open(os.path.join(author_path,book_name),"r") as book:
                curr_row["book_text"] = book.read()
            rows_list.append(curr_row.copy())
    return pd.DataFrame(rows_list)
