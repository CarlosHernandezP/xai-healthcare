import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable

import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

import wandb

import torchtuples as tt
from torch.utils.data import DataLoader
from torchtuples.callbacks import Callback

import torchvision
from torchvision import transforms
from sksurv.util import Surv

from pycox.evaluation import EvalSurv

from preprocessing import pre_process_seer

def obtain_data(dl : bool=False):
    """
    Returns pre-processed datasets
    """
    
    df_train = pd.read_csv('xai_train_data.csv')
    df_test  = pd.read_csv('xai_test_data.csv')
    
    df_train_pre, scaler = pre_process_seer(df_train.copy(), scale_data = True)
    df_test_pre, _ = pre_process_seer(df_test.copy(), scale_data = True)
    
    # Order columns
    column_order = df_train_pre.columns.tolist()
    df_test_pre = df_test_pre.reindex(columns=column_order)
    
    # Scale test
    # fit the scaler to the data and transform it
    scaled_data = scaler.transform(df_test_pre)

    # convert the numpy array back to a DataFrame
    df_test_pre = pd.DataFrame(scaled_data, columns=df_test_pre.columns)

    y_train, y_test = obtain_target(df_train, df_test, deep_learning=dl)
    
    return df_train_pre, df_test_pre, y_train, y_test


def obtain_data_cval(dl : bool=False):
    """
    Returns pre-processed datasets
    """
    
    df_train = pd.read_csv('xai_train_data.csv')
    df_test_outer  = pd.read_csv('xai_test_data.csv')
    
    inner_folds = cross_val_folds(df_train)
    
    inner_processed = []
    label_inner     = []
    for train_inner, val_inner in inner_folds:
        train_inner_pre, scaler = pre_process_seer(train_inner.copy(), scale_data = True)
        val_inner_pre, _ = pre_process_seer(val_inner.copy(), scale_data = False)

        # Order columns
        column_order = train_inner_pre.columns.tolist()
        val_inner_pre = val_inner_pre.reindex(columns=column_order)

        # Scale test
        # fit the scaler to the data and transform it
        scaled_data = scaler.transform(val_inner_pre)
        # convert the numpy array back to a DataFrame
        val_inner_pre = pd.DataFrame(scaled_data, columns=val_inner_pre.columns)

        y_train, y_val = obtain_target(train_inner, val_inner, deep_learning=dl)
        
        inner_processed.append([train_inner_pre, val_inner_pre])
        label_inner.append([y_train, y_val])
    
    
    ## Now do the outer loop
    df_train_pre, scaler = pre_process_seer(df_train.copy(), scale_data = True)
    df_test_pre, _ = pre_process_seer(df_test_outer.copy(), scale_data = False)
    
    # Order columns
    column_order = df_train_pre.columns.tolist()
    df_test_pre = df_test_pre.reindex(columns=column_order)  
    # Scale test
    # fit the scaler to the data and transform it
    scaled_data = scaler.transform(df_test_pre)
    # convert the numpy array back to a DataFrame
    df_test_pre = pd.DataFrame(scaled_data, columns=df_test_pre.columns)

    y_train, y_test = obtain_target(df_train, df_test_outer, deep_learning=dl) 
    
    data_outer  = [df_train_pre, df_test_pre]
    label_outer = [y_train, y_test]
    return inner_processed, label_inner, data_outer, label_outer


def obtain_target(df_train, df_test, deep_learning : bool=False):
    df_train.loc[: , 'event'] = [False if x=='Alive' else True for x in df_train['Vital status recode (study cutoff used)']]
    df_test.loc[:, 'event'] = [False if x=='Alive' else True for x in df_test['Vital status recode (study cutoff used)']]
    
    if deep_learning:
        get_label =  lambda df: (df["Survival months"].values, df["event"].values)
        y_train = get_label(df_train)
        y_test = get_label(df_test)
    else:
        y_train = Surv.from_dataframe('event','Survival months', df_train)
        y_test = Surv.from_dataframe('event','Survival months', df_test)
    
    return y_train, y_test
    

def cross_val_folds(data, n_splits=3, shuffle=True):
    # Create a KFold object with the desired number of splits
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=123)

    # Initialize an empty list to hold the folds
    folds = []

    # Loop over the folds generated by the KFold object
    for train_index, val_index in kf.split(data):
        # Split the data into training and validation sets
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]

        # Add the training and validation sets to the folds list
        folds.append([train_data, val_data])

    return folds

class Concordance(tt.cb.MonitorMetrics):
    """
    Concordance index callback for PyTorch-TensorFlow
    """

    def __init__(
        self,
        x,
        durations,
        events,
        per_epoch=5,
        verbose=True,
        wandb_metric: str = "none",
        use_wandb: bool = False,
    ):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose
        self.wandb_metric = wandb_metric
        self.use_wandb = use_wandb

    def on_epoch_end(self):
        super().on_epoch_end()
        # log in wandb self.model_val_metrics.scores['loss']['score'][-1]
        if self.use_wandb:
            wandb.log(
                {
                    "Validation loss": self.model.val_metrics.scores["loss"]["score"][
                        -1
                    ],
                    "Train loss": self.model.train_metrics.scores["loss"]["score"][-1],
                }
            )
        if self.epoch % self.per_epoch == 0:
            self.model.net.eval()
            surv = self.model.interpolate(20).predict_surv_df(self.x)
            self.model.net.train()
            ev = EvalSurv(surv, self.durations, self.events)

            concordance = ev.concordance_td()
            self.append_score("concordance", concordance)

            if self.verbose:
                print(f"C-index val", concordance)
                if self.use_wandb:
                    wandb.log({"C-index val": concordance}, commit=False)

    def get_last_score(self):
        return self.scores["concordance"]["score"][-1]

class LRScheduler(Callback):
    """
    Callback to change the learning rate of the optimizer using the loss
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self):
        if "ReduceLROnPlateau" in str(type(self.scheduler)):
            self.scheduler.step(self.model.train_metrics.scores["loss"]["score"][-1])
        else:
            self.scheduler.step()


# def on_batch_end(self):
#    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
#        self.scheduler.step(self.model.batch_loss)




def obtain_label_funct(args) -> Callable:
    """
    Obtains the function to be used to obtain the labels

    Inputs:
    -----------------------------
    args : argparse.Namespace : Arguments of the script

    Outputs:
    -----------------------------
    label_funct : function : Function to be used to obtain the labels
    """
    if args.model in ['cox', 'rsf', 'xgb', 'gbt']:
        import ipdb;ipdb.set_trace()
        label_funct = lambda df: df[["event", "duration"]]
    elif args.model in ['deephit', 'multimodal']:
        label_funct = lambda df: (df["duration"].values, df["event"].values)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    return label_funct


