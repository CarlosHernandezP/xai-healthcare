import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from xgbse import XGBSEKaplanNeighbors
from xgbse._kaplan_neighbors import DEFAULT_PARAMS
from xgbse.metrics import concordance_index

# Our very own survLime!
from survlimepy import SurvLimeExplainer
#from survlime import survlime_explainer
from functools import partial

np.random.seed(123456)

import sys

# Now you can import modules from the utils directory
from script_utils import obtain_data_cval
from exp_utils import get_model_name, get_output_times

def main(args):
    import ipdb;ipdb.set_trace() 
    if args.model in ['deepsurv', 'deephit']:
        deep_learning = True
    else:
        deep_learning = False
    
    inner_data, inner_labels, outer_data, outer_labels = obtain_data_cval(dl=deep_learning)
    total_c = []
    total_b = []
    
    train_data = outer_data[0]
    test_data  = outer_data[1]

    train_labels = outer_labels[0]
    test_labels  = outer_labels[1]
    
    # Load model
    
    # get model name and output times
    model_name = get_model_name(model)
    model_output_times = get_output_times(, train_labels)

    explainer = SurvLimeExplainer(
            training_features=X_train_model,
            training_events=[tp[0] for tp in y_train],
            training_times=[tp[1] for tp in y_train],
            model_output_times=model_output_times,
            random_state=repetition,
    )

    computation_exp = compute_weights(explainer, X_test_t[:371],
                                      model_pipe, num_neighbors = num_neighbors
                                      , column_names = columns,
                                      predict_chf = predict_chf,
                                      type_fn = type_fn,
                                      )

def compute_weights(
    explainer: SurvLimeExplainer,
    x_test:  pd.DataFrame,
    model: CoxPHSurvivalAnalysis,
    num_neighbors: int = 1000,
    column_names: list = None,
    predict_chf = None,
    type_fn = 'cumulative',

) -> pd.DataFrame:
    compt_weights = []
    num_pat = num_neighbors
    for test_point in tqdm(x_test):
        b = explainer.explain_instance(
            test_point, predict_chf, verbose=False, num_samples=num_pat,
            type_fn = type_fn
        )

        compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=column_names)



if __name__ == "__main__":
    main(None)
