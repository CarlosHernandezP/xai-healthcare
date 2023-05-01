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

from sklearn_pandas import DataFrameMapper

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
    column_names = train_data.columns
    train_labels = labels[0]
    test_labels  = labels[1]
    
    # Load model
    if deep_learning:
        cindex, brier, log, model = train_script.use_dl(data, labels, args)
        # Convert the list into two arrays
        train_events = [x for x in train_labels[1]]
        train_times  = [x for x in train_labels[0]]
        # Convert data to the right format
        leave = [(col, None) for col in train_data.columns]
        x_mapper = DataFrameMapper(leave) 
        train_data = x_mapper.fit_transform(train_data).astype('float32')
        test_data  = x_mapper.transform(test_data).astype('float32')
        
    else:
        cindex, brier, model = train_script.use_ml_models(data, labels, args)
        ### Number 0 is train, 1 is test and 2 is val
        # Convert labels to the right format
        train_events = [tp[0] for tp in train_labels]
        train_times  = [tp[1] for tp in train_labels]    


    print(f"{cindex:.3f} & {brier:.3f}")

    # get model name and output times
    model_output_times = get_output_times(model, train_labels)
    predict_fn, type_fn = get_predict_fn(model, args)

    explainer = SurvLimeExplainer(
            training_features=train_data,
            training_events=train_events,
            training_times= train_times,
            model_output_times=model_output_times,
    )

    print('compt_weights_{}.csv'.format(get_model_name(model)))

    # Compute the mean values of each column
    mean_test = test_data.mean()

    # Repeat the mean values 100 times to create a new DataFrame
    mean_test_df = pd.DataFrame(np.repeat(mean_test.values.reshape(1, -1), 100, axis=0), columns=mean_test.index)
    computation_exp = compute_weights(explainer, mean_test_df,
                                      model, num_neighbors = 1000,
                                      column_names = column_names,
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
    
    # if x_test is a dataframe convert it to numpy array
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values

    for test_point in tqdm(x_test):
        b = explainer.explain_instance(
            test_point, predict_chf, verbose=False, num_samples=num_pat,
            type_fn = type_fn, max_hazard_value_allowed=99
        )

        compt_weights.append(b)
    
    weights_df  = pd.DataFrame(compt_weights, columns=column_names)

    # Make a component-wise multiplication of weights_df and x_test
    weights_df = weights_df.mul(x_test)

    weights_df.to_csv(
        'compt_weights_{}_mean_repeat.csv'.format(get_model_name(model)), index=False
    )

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
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=11)
    parser.add_argument("--min_samples_split", type=int, default=25)
    parser.add_argument("--max_features", type=str, default="auto")

    parser.add_argument(
        "--model",
        type=str,
        default="rsf",
        help="Model to be used for training it can be xgb, cox, rsf, deepsurv or multi",
    )

    args = parser.parse_args()
    
    main(args)
