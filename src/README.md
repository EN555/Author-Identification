# Python Utility Codes

Any code that can be reused will put here under the right filename 
and will be used in the jupyter notebook with simple import
for **example**: 

```
import sys
sys.path.append("../")
from PythonCode.preprocess.complexStyleFeatures import *
```

## Preprocess:
Generic pipeline example use:

    from PythonCode.models import Model
    class CustomDescionTree(Model):
        def train(self, x_train, y_train,**kwargs):
            clf = DecisionTreeClassifier(**kwargs)
            clf.fit(x_train,y_train)
            return clf
        def predict(self,x_test):
            return clf.predict(x_test)
    CustomDescionTree().pipeline(x_train, y_train, x_test, y_test,"simple_descion_tree",do_cross_validation=True,random_state=93)




## Models:


## Features:
### Complex Style Features Sources
גיט האב עם הרבה מימושים של פיצ'רים של סגנון. הרבה מהמימושים לא יעילים/לא אלגנטיים, אבל לקחתי ממנו הרבה רעיונות
https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis/blob/master/Code/main.py

רשימת מקורות קריאה על כל הפיצ'רים המורכבים:
honore measure R: https://link.springer.com/content/pdf/bbm%3A978-0-230-51180-4%2F1.pdf (page 6 at the buttom)
Yules characteristic : https://link.springer.com/content/pdf/bbm%3A978-0-230-51180-4%2F1.pdf (page 17 at the buttom)
simpsons measure: https://geographyfieldwork.com/Simpson'sDiversityIndex.htm (biology explenation, its the same with words)
brunets measure: https://link.springer.com/content/pdf/bbm%3A978-0-230-51180-4%2F1.pdf (page 4)
zipf's law: https://he.wikipedia.org/wiki/%D7%97%D7%95%D7%A7_%D7%96%D7%99%D7%A3
flesch reading ease and lesch-kincaid grade level: https://readable.com/readability/flesch-reading-ease-flesch-kincaid-grade-level/
gunning fog index: https://en.wikipedia.org/wiki/Gunning_fog_index


```
import functools
from preprocess.common import *
from preprocess.complexStyleFeatures import *
from preprocess.simpleStyleFeatures import *

preprocess_pipeline(data_path="../C50/C50train/", number_of_authors=2,
                    repesention_handler=functools.partial(combine_features,
                    [complex_style_features_extraction, simple_style_features_extraction,
                     functools.partial(pos_n_grams, n=4), functools.partial(characters_n_grams, n=2, min_df=0.1)]),
                    normalize=True, data_filter=chunking,
                    save_path="../../Data/clean/twoauthors/")
```