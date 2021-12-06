# Python Utility Codes

Any code that can be reused will put here under the right filename 
and will be used in the jupyter notebook with simple import
for example: <br><br>`from PythonCode.preprocess import load_data`

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