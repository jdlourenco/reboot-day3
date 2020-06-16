# Iterate with MlFlow

Now that we have already built a simple, we want to make it better! The ultimate goal is having a model that makes more accurate predictions on the test set, hence getting a RMSE as low as possible.

**So what can we do?**

There are many different things that make models better:
- build and try to use different or more features
- test with different estimators (linear, non linear, etc..)
- Test influce of distance definition


## Installation
```bash
make install
```

## Parameters to set
 - Inpsect how parameters are passed to `Trainer()` class inside `TaciFareMoel/trainer.py`
 - We will play with these parameters to run different experiments

## Experiments
Inside `main.py` we define 3 experiments, for each experiments we generate a list of parameters, i.e:
```python
def distance_experiment(default_params=DEFAULT_PARAMS):
    new_params = copy(default_params)
    new_params["experiment_name"] = "distance"
    new_params["estimator"] = "RandomForest"
    l_params = []
    for distance_type in ["haversine", "euclidian", "manhattan"]:
        params = copy(new_params)
        params["distance_type"] = distance_type
        l_params.append(params)
    return l_params
``` 

## How to use
- Chose experiments amongst `distance_experiment`, `model_experiment` and `feat_eng_experiment`
- generate list of parameters 
- run workflow for each set of parameters


