# Final Prudact

This part of the project comes to show how our model can be used in real life with unkown authors


# FrontEnd

written with react.js framework implemented this screens:

## Inference
TODO
## Models
TODO
## Retrain
TODO

## Development
install node.js before...

### Install Requirments

```
npm i
```

### Run
```
npm start
```


# Backend

written with FastApi library and mongodb in python.

## Endpoints
- `GET inferences`
- `GET models`
- `PUT model(model_id)`
- `POST retrain(dataset)`
- `POST infer(text)`

## Development
install python 3.8 before...

### Install Requirments
```
pip install -r requirments.txt
pip install -r requirments-dev.txt
```
### SetUp Envirement Variables

```
export APP_PORT=80
export APP_HOST=localhost
export MONGO_USER=
export MONGO_PASS=
MONGO_DB_NAME="db_name"
```

### Run Server
```
python product/backend/main.py
```

