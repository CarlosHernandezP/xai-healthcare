import numpy as np
import re
from preprocessing import pre_process_seer
from script_utils import obtain_data, obtain_data_cval

import train_script
import pandas as pd

import wandb
import argparse

def main(args):
    
    if args.model in ['deepsurv', 'deephit']:
        deep_learning = True
    else:
        deep_learning = False
    
    inner_data, inner_labels, outer_data, outer_labels = obtain_data_cval(dl=deep_learning)
    total_c = []
    total_b = []
    
    # Initialize wandb with the catai entity
    wandb.init(entity="catai", settings=args)
    for data, labels in zip(inner_data, inner_labels):
        if deep_learning:
            cindex, brier, log = train_script.use_dl(data, labels, args)
        else:
            cindex, brier = train_script.use_ml_models(data, labels, args)
        total_c.append(cindex)
        total_b.append(brier)
            
    # Calculate the mean and standard deviation of total_c and total_b
    mean_c = np.mean(total_c)
    std_c = np.std(total_c)
    mean_b = np.mean(total_b)
    std_b = np.std(total_b)

    # Print the results
    print("C-index: mean = {:.3f}, std = {:.3f}".format(mean_c, std_c))
    print("Brier score: mean = {:.3f}, std = {:.3f}".format(mean_b, std_b))

    # Log the results to wandb
    wandb.log({
        "mean_c": mean_c,
        "std_c": std_c,
        "mean_b": mean_b,
        "std_b": std_b
    })
    
if __name__ =='__main__':
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
