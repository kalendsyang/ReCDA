from collections import Counter
import os
import random
import numpy as np
import pandas as pd
import torch
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def fix_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_data(root_path, dataset_name, unknown_classes=[0, 3, 5]):
    """Load and preprocess dataset based on the given dataset name."""
    data_path = os.path.join(root_path, "data", dataset_name)

    if dataset_name == "unsw":
        train_df = pd.read_csv(os.path.join(data_path, 'treated_train.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'treated_test.csv'))

        # Prepare training and testing features and labels
        x_train = train_df.drop(['classes'], axis=1).values.astype(np.float64)
        y_train = train_df['classes'].values.astype(np.float64)

        x_test = test_df.drop(['classes'], axis=1).values.astype(np.float64)
        y_test = test_df['classes'].values.astype(np.float64)

        # Remap classes, merge 0,1,8,9
        merge_mapping = {6: 6, 0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 7, 8: 1, 9: 1}
        y_train = np.array(list(map(merge_mapping.get, y_train)))
        y_test = np.array(list(map(merge_mapping.get, y_test)))

        # Identify known and unknown classes
        all_classes = np.unique(y_train)
        known_classes = np.setdiff1d(all_classes, unknown_classes)

        x_known = x_train[np.isin(y_train, known_classes)]
        y_known = y_train[np.isin(y_train, known_classes)]

        x_unknown = x_train[np.isin(y_train, unknown_classes)]
        y_unknown = y_train[np.isin(y_train, unknown_classes)]

        # Convert labels to binary
        binary_mapping = {6: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 7: 1, 8: 1, 9: 1}
        y_train_binary = np.array(list(map(binary_mapping.get, y_train)))
        y_known_binary = np.array(list(map(binary_mapping.get, y_known)))
        y_unknown_binary = np.array(list(map(binary_mapping.get, y_unknown)))
        y_test_binary = np.array(list(map(binary_mapping.get, y_test)))

        print("y_known_binary", Counter(y_known_binary))
    
    return x_train, y_train_binary, x_test, y_test_binary, x_known, y_known_binary, x_unknown, y_unknown_binary

def split_data(data, target, sample_rate, seed):
    """Split data into sampled and validation sets."""
    sampled_data, x, sampled_target, y = train_test_split(
        data,
        target,
        test_size=1 - sample_rate,
        stratify=target,
        random_state=seed
    )
    return sampled_data, sampled_target

def get_data_embeddings(model, loader):
    """Obtain embeddings from the model for the given data loader."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for anchor, _, _ in tqdm(loader):
            anchor = anchor.to(next(model.parameters()).device)
            embeddings.append(model.get_embeddings(anchor))

    return torch.cat(embeddings)

def get_logger(args):
    """Set up the logger for tracking progress and errors."""
    log_filename = os.path.join(args.root_path, "log", f"{args.dataset}_{args.task}_{args.e_depth}_{args.h_depth}.txt")

    # Ensure the log file exists
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(level_dict[1])

    fh = logging.FileHandler(log_filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

