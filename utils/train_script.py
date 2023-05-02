import argparse
import pickle
from typing import List, Tuple, Union, Any

import numpy as np
import pandas as pd

import pycox
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, DeepHitSingle, LogisticHazard

from sklearn_pandas import DataFrameMapper

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

import torch
from torch import optim
import torchtuples as tt

import wandb

import xgboost as xgb
from xgbse import XGBSEKaplanNeighbors, XGBSEStackedWeibull
from xgbse.converters import convert_data_to_xgb_format, convert_to_structured
from xgbse.metrics import approx_brier_score, concordance_index

from sksurv.util import Surv



def use_ml_models(data: Tuple, labels: Tuple, args: argparse.Namespace) -> Tuple[float, float]:
    """Use ML models to predict survival.
    
    Args:
        data: A tuple containing the data to use for training, validation, and testing.
        labels: A tuple containing the survival information corresponding to the data.
        args: An argparse.Namespace object containing the command-line arguments.
        model_name: A string indicating which model to use.
        
    Returns:
        A tuple containing the test set score and the integrated Brier score.
    """
    
    # instantiate a RandomSurvivalForest model with n_iter, n_estimators, max_features, max_depth taken from args
    if args.model=='rsf': 
        if args:
            model = RandomSurvivalForest(n_estimators=args.n_estimators,
                                        max_features=args.max_features,
                                        min_samples_split=args.min_samples_split,
                                        max_depth=args.max_depth,
                                        n_jobs=-1, random_state=123)
        else:
            model = RandomSurvivalForest()
    elif args.model == 'gbt':
       # model = GradientBoostingSurvivalAnalysis(learning_rate=args.lr,
       #                     n_estimators=args.n_estimators, 
       #                     min_samples_split=args.min_samples_split,
       #                     max_depth=args.max_depth, dropout_rate=args.dropout_prob, 
       #                     random_state=42)
        model = GradientBoostingSurvivalAnalysis(random_state=123)
    elif args.model=='cox':
   ## Repeat with CoxPH
        model = CoxPHSurvivalAnalysis(alpha=0.001)#,
                                    #n_iter=args.n_iter,
                                    #tol=args.tol)
    
    # Fit the model on the training data.
    model.fit(data[0].copy(), labels[0])
    
    
    c_index, ib_score = compute_brier_n_c_index(model, data[1], labels[1])

    return c_index, ib_score, model

def use_dl(data, labels, args):
    """Use DeepSurv to predict survival"""

   # num_nodes = [32, 32, 32, 16]
     # Transform data into numpy arrays
    leave = [(col, None) for col in data[0].columns]
    x_mapper = DataFrameMapper(leave) 
    
    labels_list = [(label, value) for label, value in zip(labels[0][0], labels[0][1])]
    data_train, data_val, labels_train, labels_val = train_test_split(data[0],
                                        labels_list, test_size=0.2, random_state=42)
    # Convert the list into two arrays
    labels_array = np.array(labels_train)
    labels_arr1 = labels_array[:, 0]
    labels_arr2 = labels_array[:, 1]

    # Convert the arrays back into a tuple
    labels[0] = (labels_arr1, labels_arr2)

    labels_array = np.array(labels_val)
    labels_arr1 = labels_array[:, 0]
    labels_arr2 = labels_array[:, 1]
    
    ## Add val label
    labels.append((labels_arr1, labels_arr2))
    

    ### Number 0 is train, 1 is test and 2 is val
    data[0] = x_mapper.fit_transform(data[0]).astype('float32')
    data[1] = x_mapper.transform(data[1]).astype('float32')
  # data.append(x_mapper.fit_transform(data_val).astype('float32'))

    if args.model == 'deepsurv':
        out_features = 1
    elif args.model =='deephit':
        num_durations = args.num_durations
        labtrans = DeepHitSingle.label_transform(num_durations)
        labels[0] = labtrans.fit_transform(labels[0][0], labels[0][1])
        labels[2] = labtrans.transform(labels[2][0], labels[2][1])
        out_features = labtrans.out_features
    else:
        out_features = 1
        
    num_nodes = [args.num_nodes for i in range(args.num_layers)]

    net = tt.practical.MLPVanilla(data[0].shape[1], num_nodes, out_features, batch_norm=args.batch_norm,
                                  dropout=args.dropout_prob, output_bias=args.output_bias)
    if args.model == 'deepsurv':
        model = CoxPH(net, tt.optim.Adam(weight_decay=args.reg))#args.reg))
    elif args.model == 'deephit':
        model = DeepHitSingle(net, tt.optim.Adam(weight_decay=args.reg), duration_index=labtrans.cuts)

    callbacks = [tt.callbacks.EarlyStopping(patience=15)]
    model.optimizer.set_lr(args.lr)

    # Train!
    log = model.fit(input=data[0], target=labels[0], batch_size=args.bs,
                    epochs=args.epochs, val_data = (data[2], labels[2]),val_batch_size=args.bs,
                    callbacks=callbacks, verbose=args.verbose)

    pandas_log =log.to_pandas()
   
    ## This chunk of code plots the losses for the validation 
    if True:
        try:
            losses = [[train, val] for (train, val) in zip(pandas_log['train_loss'], pandas_log.index.to_list())]
            table = wandb.Table(data= losses, columns = ['train_loss', 'epoch'])
            wandb.log({"train_loss" : wandb.plot.line(table, "epoch", "train_loss",
                       stroke = None,  title="Train loss")})

            losses = [[loss, val] for (loss, val) in zip(pandas_log['val_loss'], pandas_log.index.to_list())]
            table = wandb.Table(data= losses, columns = ['val_loss', 'epoch'])
            wandb.log({"val_loss" : wandb.plot.line(table, "epoch", "val_loss",
                       stroke = None,  title="Validation loss")})
        except:
            print('Could not log loss')


    if args.model == 'deepsurv':
        model.compute_baseline_hazards()
    # Compute metrics
    c_index, ib_score = compute_brier_n_c_index(model, data[1], labels[1])

 #   if args.ajcc_subset:
 #       save_path = f'saved_models/{args.model}_{args.event_type}_miss_ajcc.pkl'
 #   else:
 #       save_path = f'saved_models/{args.model}_{args.event_type}_miss_{args.missing}.pkl'
   #if data[2].shape[0] >800:
   #    model.save_net(save_path)
    return c_index, ib_score, log, model


def use_xgboost(data: List[pd.DataFrame], labels: List[pd.DataFrame], args: argparse.Namespace) -> Tuple[float, float]:
    """
    Use XGBoost models to predict survival.

    Parameters:
        data (List[pd.DataFrame]): A list containing the train and test data.
        labels (List[pd.DataFrame]): A list containing the train and test labels.
        args (argparse.Namespace): An object containing command-line arguments.

    Returns:
        Tuple[float, float]: A tuple containing the concordance index and the integrated Brier score.
    """
    
    
    # Convert labels to structured arrays
    y_train = convert_to_structured(labels[0]['duration'], labels[0]['event'])
    y_test = convert_to_structured(labels[2]['duration'], labels[2]['event'])

    
    # Define XGBoost AFT model parameters
    PARAMS_XGB_AFT = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': 'normal',
        'aft_loss_distribution_scale': 1.0,
        'tree_method': 'hist', 
        'learning_rate': args.lr, 
        'max_depth': args.max_depth,
        'alpha': args.alpha,
        'booster':'dart',
        'subsample':0.5,
        'min_child_weight': args.min_child_weight,
        'colsample_bynode':0.5
    }
    
    # Instantiate and fit the XGBSE model
    #bst = XGBSEKaplanNeighbors(PARAMS_XGB_AFT, n_neighbors=100)
    model = XGBSEStackedWeibull(PARAMS_XGB_AFT)
    model = XGBSEStackedWeibull()
    model = model.fit(data[0], y_train, num_boost_round=args.num_boost_round)

    # Define the path to save the model
    if args.ajcc_subset:
        save_path = f'saved_models/{args.model}_{args.event_type}_miss_ajcc.pkl'
    else:
        save_path = f'saved_models/{args.model}_{args.event_type}_miss_{args.missing}.pkl'

    # Save the model
    #with open(save_path, 'wb') as f:
    #    pickle.dump(model, f)

    # Compute metrics
    c_index, ib_score = compute_brier_n_c_index(model, data[2], y_test)

    return c_index, ib_score


def compute_brier_n_c_index(model : Any, data: Union[np.ndarray, pd.DataFrame], labels : Union[np.ndarray, pd.DataFrame]) -> Tuple[float, float]:
    """
    Compute the integrated Brier score and the concordance index.

    Parameters:
    model (Any): The model to use.
    preds (Union[np.ndarray, pd.DataFrame]): The predictions.

    Returns:
    c_index (float): The concordance index.
    ib_score (float): The integrated Brier score.
    """
    if 'pycox' in str(type(model)).lower():
        preds = model.predict_surv_df(data)
        # Create evaluation object
        ev = EvalSurv(preds, labels[0], labels[1], censor_surv='km')
        # TODO check if creating 100 time points is the best way to do this
        time_grid = np.linspace(labels[0].min(), labels[0].max(), 100)
       
        # Compute the concordance index
        c_index = ev.concordance_td('antolini')
        # Compute the integrated Brier score
        ib_score = ev.integrated_brier_score(time_grid)
    elif 'xgbse' in str(type(model)).lower():
        preds = model.predict(data)
        # Compute the concordance index
        c_index = concordance_index(labels, preds)
        # Compute the integrated Brier score
        ib_score = approx_brier_score(labels, preds, aggregate='mean')
    else:
        ## Sksurv models fall here
        c_index = model.score(data.copy(), labels)
        # Calculate the survival function for the test set.
        surv = model.predict_survival_function(data.copy(), return_array=True)
        times = [x[1] for x in labels]
        event_times = model.event_times_

        # Create a new array of event times that only includes the time points within the follow-up time of your test data
        max_followup_time = max(times)
        event_times_within_followup = event_times[event_times < max_followup_time]
        # Retain only the indexes of surv_test that correspond to the indexes of event_times_within_followup
        ix = np.isin(event_times, event_times_within_followup)
        surv_event_times = [sf[ix] for sf in surv]
        ib_score = integrated_brier_score(labels, labels, surv_event_times, event_times_within_followup)

    return c_index, ib_score


#### CHUNK OF CODE TO SAVE A MODEL
    # Define the path for saving the model.
   # if args.ajcc_subset:
   #     save_path = f'saved_models/{model_name}_{args.event_type}_miss_ajcc.pkl'
   # else:
   #     save_path = f'saved_models/{model_name}_{args.event_type}_miss_{args.missing}.pkl'

    # Save the model as a pickle.
   # with open(save_path, 'wb') as f:
   #     pickle.dump(model, f)
