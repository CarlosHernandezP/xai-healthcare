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


# Now you can import modules from the utils directory
import utils.train_script as train_script
from utils.script_utils import obtain_data_cval
from utils.exp_utils import get_model_name, get_output_times, get_predict_fn

def main(args):
    if args.model in ['deepsurv', 'deephit']:
        deep_learning = True
    else:
        deep_learning = False
    
    _, _, data, labels = obtain_data_cval(dl=deep_learning)
    
    train_data = data[0]
    test_data  = data[1]
    train_labels = labels[0]
    test_labels  = labels[1]
    
    # Load model
    if deep_learning:
        cindex, brier, log, model = train_script.use_dl(data, labels, args)
    else:
        cindex, brier, model = train_script.use_ml_models(data, labels, args)
    # Print cindex and brier in one line
    print(f"{cindex:.3f} & {brier:.3f}")

    if deep_learning:
        cindex, brier, log, model = train_script.use_dl(data, labels, args)
    else:
        cindex, brier, model = train_script.use_ml_models(data, labels, args)

    # get model name and output times
    model_name = get_model_name(model)
    model_output_times = get_output_times(model, train_labels)
    predict_fn, type_fn = get_predict_fn(model, args)

    explainer = SurvLimeExplainer(
            training_features=train_data,
            training_events=[tp[0] for tp in train_labels],
            training_times=[tp[1] for tp in train_labels],
            model_output_times=model_output_times,
    )
    print('compt_weights_{}.csv'.format(get_model_name(model)))
    computation_exp = compute_weights(explainer, test_data,
                                      model, num_neighbors = 1000,
                                      column_names = test_data.columns,
                                      predict_chf = predict_fn,
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
    
    for test_point in tqdm(x_test.values):
        b = explainer.explain_instance(
            test_point, predict_chf, verbose=False, num_samples=num_pat,
            type_fn = type_fn
        )

        compt_weights.append(b)
    
    weights_df  = pd.DataFrame(compt_weights, columns=column_names)
    weights_df.to_csv(
        'compt_weights_{}.csv'.format(get_model_name(model)))
    return weights_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Configuration for cross validation multimodal survival"
    )

    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--reg", type=float, default=0.001, help="Regularization")
    parser.add_argument(
        "--num_durations", type=int, default=30, help="Number of durations"
    )

    # DeepSurv argument
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of layers in the network"
    )
    parser.add_argument("--num_nodes", type=int, default=16)
    parser.add_argument("--batch_norm", type=bool, default=True)
    parser.add_argument("--output_bias", type=bool, default=True)

    # XGboost argument add aft_loss_distribution aft_loss_distribution_scale
    parser.add_argument("--aft_loss_distribution", type=str, default="normal")
    parser.add_argument("--aft_loss_distribution_scale", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--num_boost_round", type=int, default=100)
    parser.add_argument(
        "--gamma", type=float, default=0.001
    )  # <- monomum loss reduction required to make further partition
    # parser.add_argument('lambda', type=float, default=1)

    # Machine learning Arguments
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--max_features", type=str, default="auto")

    parser.add_argument(
        "--model",
        type=str,
        default="rsf",
        help="Model to be used for training it can be xgb, cox, rsf, deepsurv or multi",
    )

    args = parser.parse_args()
    
    main(args)
