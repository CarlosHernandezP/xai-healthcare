import numpy as np
from functools import partial
from typing import List

def get_model_name(model):
    """
    Returns the name of the model given as input.

    Parameters:
    -----------
    model: object
        The model for which the name is to be obtained.

    Returns:
    --------
    model_name: str
        The name of the model as a string.

    Raises:
    -------
    TypeError:
        If the model parameter is not an object or if it is not supported.
    """
    
    if 'deep' in str(type(model)).lower():
        model_name = 'DeepHit'
    elif 'xgbse' in str(type(model)).lower():
        model_name = 'XGBSE'

    else:
        if 'cox' in str(type(model)).lower():
            model_name = 'CoxPH'
        elif 'forest' in str(type(model)).lower():
            model_name = 'RSF'
        else:
            raise TypeError("The model is not supported.")
    return model_name


def get_output_times(model,
                     labels : List=None) -> np.ndarray:
    """
    Returns the output times for the given survival model.

    Parameters:
    -----------
    model :
        The trained survival model for which the output times need to be obtained.

    Returns:
    --------
    output_times : numpy.ndarray
        The output times of the given survival model.
    """
    
    if 'pycox' in str(type(model)).lower():
        if 'deephit' in str(type(model)).lower():
            output_times = model.duration_index
        else:
            train_times = [x for x in labels[0]]
            output_times = np.unique(train_times)
    elif 'xgbse' in str(type(model)).lower():
        output_times = model.time_bins
    else:
        output_times = model.event_times_
    return output_times

def get_predict_fn(model, args):
    """
    Returns the predict function for the given survival model.
    and the type of function
    """

    if args.model in ['rsf', 'cox']:
        predict_fn = partial(
                    model.predict_cumulative_hazard_function, return_array=True
                )
        type_fn = "cumulative"

    elif args.model == 'deepsurv':
        def create_chf(fun):
            def inner(X):
                Y = fun(X)
                return Y.T
            return inner

        predict_fn =  create_chf(model.predict_cumulative_hazards) 
        type_fn = "cumulative"
    elif args.model == 'deephit':
        predict_fn = model.predict_surv
        type_fn = "survival"
    else:
        raise ValueError("Model not supported")

    return predict_fn, type_fn
