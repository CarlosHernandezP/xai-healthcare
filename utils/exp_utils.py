import numpy as np
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
    if 'deep' in str(type(model)).lower():
        train_times = [x for x in y_train[0]]
        output_times = np.unique(train_times)
       #if dfs:
       #    output_times = np.array(deephit_times_dfs)
       #else:
       #    output_times = np.array(deephit_times)
    elif 'xgbse' in str(type(model)).lower():
        output_times = model.time_bins
    else:
        output_times = model.event_times_
    return output_times


deephit_times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
