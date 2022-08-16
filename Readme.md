# Author Identification

## Installation

```cmd
pip install -r requiremnets.txt
```

```
pip install -r requiremnets-dev.txt
make fmt
make lint
```
## Project Goal

The problem of Author identification is about the identification of  author of a tested document from a group of potential authors.
Our research focuses on try distinguish between different authors when they are about similar topics. 
![image](https://user-images.githubusercontent.com/61500507/184867334-a7e60ffd-a1c8-423b-9bd2-2450aa84c121.png)


## Data
we use the C50 data set1 which compose of 2,500 texts by 50 different authors (50 for each) for train, and the same for test. The texts are not particularly long - the average length is around 500 words. 
![image](https://user-images.githubusercontent.com/61500507/184867169-e786e565-33e5-4e11-b664-bac23c32ed63.png)

-  [https://archive.ics.uci.edu/ml/datasets/Reuter_50_50#](small-dataset)
- [https://drive.google.com/file/d/1UnTLPc0pnxDZUso-ruCu_egOnHHkJ0sh/view?usp=sharing](big-dataset)
- [https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip](smalldata) and put `Data\C50`



`pre-commit install`
