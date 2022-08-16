# Author Identification

## Table of content

- [Installation](#installation)
- [Project Goal](#project-goal)
- [Data](#data)
- [Our Approach](#our-approach)

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

## Data
we use the C50 data set1 which compose of 2,500 texts by 50 different authors (50 for each) for train, and the same for test. The texts are not particularly long - the average length is around 500 words. 
![image](https://user-images.githubusercontent.com/61500507/184867169-e786e565-33e5-4e11-b664-bac23c32ed63.png)

-  [https://archive.ics.uci.edu/ml/datasets/Reuter_50_50#](small-dataset)
- [https://drive.google.com/file/d/1UnTLPc0pnxDZUso-ruCu_egOnHHkJ0sh/view?usp=sharing](big-dataset)
- [https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip](smalldata) and put `Data\C50`



`pre-commit install`


# Our Approach 
We first encoding each sentence using Glove50, then we use average pooling over all the sentence,

<a href="url"><img src="https://user-images.githubusercontent.com/61500507/184868989-47c10e3a-7c86-4b88-b774-c759c2e8ae98.png" height="300" width="300" ></a>

Then we got vector with all the probability for each of the authors.  
We check the max probability in compare to threshold, if the result lower than the threshold we extract the 10 authors with the high probability.

<a href="url"><img src="https://user-images.githubusercontent.com/61500507/184871278-8e365b09-4b56-4658-b9bc-d562112f3333.png" height="50" width="400" ></a>

Then we check the correct author using pure style model, we extract complex and simple features for each of the document. The accuracy that we got 83.2%.

