import os
import copy
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, 
                             precision_score, recall_score)

from src.utils import (get_data_embeddings, fix_seed, get_data, 
                   split_data, get_logger)
from src.loss import SimCLR
from src.model import ReCDA, MLP
from src.data import DriftDataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_dataset(args):
    """Load dataset."""

    train_data, train_target, test_data, test_target, historical_data, historical_target, recent_data, _ = get_data(args.root_path, args.dataset, args.unknown_classes)

    if args.sam_rate != 1:
        historical_data, historical_target = split_data(historical_data, historical_target, args.sam_rate, args.seed)

    re_dataset = DriftDataset(train_data, train_target, recent_data)
    ct_dataset = DriftDataset(historical_data, historical_target, recent_data)
    te_dataset = DriftDataset(test_data, test_target, recent_data)

    return re_dataset, ct_dataset, te_dataset

def evaluate_model(model, classifier, te_dataset):
    """Evaluate the model on the test dataset."""

    test_loader = DataLoader(te_dataset, batch_size=128, shuffle=False)
    model.eval()
    classifier.eval()

    with torch.no_grad():
        test_emb = get_data_embeddings(model, test_loader)
        test_target = te_dataset.target
        predictions = classifier(test_emb).max(1).indices.cpu()

        # Calculate evaluation metrics
        acc = accuracy_score(test_target, predictions)
        recall = recall_score(test_target, predictions, average='macro')
        precision = precision_score(test_target, predictions, average='macro')
        f1 = f1_score(test_target, predictions, average='macro')
        report = classification_report(te_dataset.target, predictions)
        cm = confusion_matrix(te_dataset.target, predictions)

    return [acc, recall, precision, f1, report, cm]

def train_epoch(model, ce_loss, train_loader, optimizer, scheduler, device, epoch):
    """Train the model for one epoch."""

    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for anchor, positive, _ in batch:
        anchor, positive = anchor.to(device), positive.to(device)
        optimizer.zero_grad()
        
        anchor_emb, positive_emb = model(anchor, positive)
        loss = ce_loss(anchor_emb, positive_emb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += anchor.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    return epoch_loss / len(train_loader.dataset)

def re_stage(args, logger, train_ds, test_ds):
    """Representation enhancement stage."""

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = ReCDA(train_ds.shape[1], args.emb_dim, args.per_rate, args.e_depth, args.h_depth).to(args.device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    ra_loss = SimCLR().to(args.device)

    logger.info('Start Representation Enhancement Stage!')
    for epoch in range(args.epochs):
        epoch_loss = train_epoch(model, ra_loss, train_loader, optimizer, scheduler, args.device, epoch)
        logger.info(f'Epoch: [{epoch}/{args.epochs}], epoch_loss={epoch_loss:.6f}')
    
    logger.info('Finish Representation Enhancement Stage!')

    model_path = os.path.join(args.root_path, "model", "encoder.pth")
    torch.save(model.state_dict(), model_path)

def ct_stage(args, logger, train_ds, te_dataset):
    """Constrained tuning stage."""

    pre_model_path = os.path.join(args.root_path, "model", "encoder.pth")
    pre_model = ReCDA(train_ds.shape[1], args.emb_dim, args.per_rate, args.e_depth, args.h_depth).to(args.device)
    pre_model.load_state_dict(torch.load(pre_model_path))

    classifier = MLP(args.emb_dim, 2, args.c_depth).to(args.device)
    model = copy.deepcopy(pre_model)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_ct, shuffle=True)

    cr_loss = nn.MSELoss(reduction='sum')
    ce_loss = nn.CrossEntropyLoss()

    optimizer = Adam(classifier.parameters(), lr=1e-3) # , weight_decay=0.01
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_ct, eta_min=0)

    if not args.freeze:
        model_optimizer = Adam(model.parameters(), lr=1e-3) # , weight_decay=0.01
        # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=args.epochs_ct, eta_min=0)
    else:
        for param in model.parameters():
            param.requires_grad = False

    logger.info('Start Constrained Tuning Stage')
    for epoch in range(args.epochs_ct):
        classifier.train()
        if not args.freeze:
            model.train()

        train_loss, train_acc = 0, 0
        batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for anchor, positive, target in batch:
            target = target.type(torch.LongTensor).to(args.device)
            anchor, positive = anchor.to(args.device), positive.to(args.device)
            
            anchor_emb = model.get_embeddings(anchor)
            pre_anchor_emb = pre_model.get_embeddings(anchor)
            loss1 = cr_loss(pre_anchor_emb, anchor_emb)
            outputs = classifier(anchor_emb)
            loss2 = ce_loss(outputs, target)
            loss = (1 - args.lamda) * loss2 + args.lamda * loss1

            train_loss += loss.item() * target.size(0)
            train_acc += (outputs.max(1)[1] == target).sum().item()

            optimizer.zero_grad()
            if not args.freeze:
                model_optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            # scheduler.step()
            if not args.freeze:
                model_optimizer.step()
                # model_scheduler.step()

        test_score = evaluate_model(model, classifier, te_dataset)
        logger.info(f'Epoch: [{epoch}/{args.epochs_ct}], loss={train_loss/train_ds.shape[0]:.6f}, train_acc={train_acc/train_ds.shape[0]:.6f}, test_acc={test_score[0]:.6f}')

    logger.info('Finish Constrained Tuning Stage')
    logger.info(f'Result at Epoch: [{epoch}], acc={test_score[0]:.6f}, recall={test_score[1]:.6f}, precision={test_score[2]:.6f}, f1={test_score[3]:.6f}')

    return test_score

def argument_parser():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='ReCDA')
    parser.add_argument('--task', type=str, default='re', choices=['re', 'ct'], help='Task name, choose from re, ct')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--dataset', type=str, default='unsw', help='Dataset name')
    parser.add_argument('--unknown_classes', type=list, default=[3, 5], help='Unknown attack types without label')
    parser.add_argument('--per_rate', type=float, default=0.6, help='Perturbation rate')
    parser.add_argument('--sam_rate', type=float, default=0.75, help='Sample selection rate')
    parser.add_argument('--emb_dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--e_depth', type=int, default=4, help='Encoder depth')
    parser.add_argument('--h_depth', type=int, default=2, help='Head depth')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--root_path', type=str, default="./", help='Root path')

    # parameters for constrasted tuning
    parser.add_argument('--lamda', type=float, default=0, help='Balance loss weight')
    parser.add_argument('--freeze', type=bool, default=False, help='Freeze encoder or not')
    parser.add_argument('--c_depth', type=int, default=1, help='Classifier depth')
    parser.add_argument('--epochs_ct', type=int, default=20, help='Number of epochs for constrasted tuning')
    parser.add_argument('--batch_size_ct', type=int, default=128, help='Batch size for constrasted tuning')
    
    return parser.parse_args()



if __name__=="__main__":
    args = argument_parser()
    args.device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')
    fix_seed(args.seed)
    logger = get_logger(args)
    re_dataset, ct_dataset, te_dataset = load_dataset(args)
    logger.info('======START======')
    logger.info("Agrs:" + str(args))
    if args.task == 're':
        re_stage(args, logger, re_dataset, te_dataset)
    elif args.task == 'ct':
        ct_stage(args, logger, ct_dataset, te_dataset)
    logger.info('======END=======')