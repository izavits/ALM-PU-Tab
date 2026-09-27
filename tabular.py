"""ALM-PU on tabular csv datasets (e.g. dataset/scaled_*.csv) with the CardMLP model.

Each csv is headerless: numeric feature columns followed by a binary label column.

  phase 1: labeled CE + "all unlabeled are negative" CE with an augmented-Lagrangian
           penalty; record per-epoch predictions on U, turn their trends into pseudo-labels
           with Jenks natural breaks.
  phase 2: fresh model trained on labeled CE + CE against labels that move from the
           observed unlabeled label towards the pseudo-label over epochs (loss_ft without
           the FixMatch consistency term, since there are no augmentations for tabular data).

"""
import argparse
import glob
import logging
import os
import random
from types import SimpleNamespace

import jenkspy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler

from model.ema import ModelEMA
from model.mlp import CardMLP
from utils.misc import three_sigma, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

METRICS = ['acc', 'auc', 'f1', 'prec', 'recall']


def parse_args():
    parser = argparse.ArgumentParser(description='ALM-PU on tabular csv datasets')
    parser.add_argument('--data', nargs='+', default=['./dataset'],
                        help='csv file(s) and/or directories (every *.csv inside is used)')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help="dataset names skipped when expanding directories (with or without the "
                             "'scaled_' prefix)")
    parser.add_argument('--seeds', nargs='+', default=[0], type=int)
    parser.add_argument('--out', default='results/tabular_results.csv', type=str,
                        help='per-(dataset, seed) results are written here')
    parser.add_argument('--positive-class', default=1, type=int,
                        help='raw label (last csv column) treated as the positive class')
    parser.add_argument('--test-frac', default=0.2, type=float, help='stratified test split')
    parser.add_argument('--label-frac', default=0.5, type=float,
                        help='fraction of training positives that are labeled (the rest go to U)')
    add_method_args(parser)
    parser.add_argument('-v', '--verbose', action='store_true', help='log every epoch')
    return parser.parse_args()


def add_method_args(parser):
    """ALM-PU / CardMLP hyperparameters, shared with pu_benchmark.py."""
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--eval-step', default=100, type=int, help='iterations per epoch')
    parser.add_argument('--warming-epochs', default=20, type=int, help='phase 1 epochs')
    parser.add_argument('--ft-epochs', default=30, type=int, help='phase 2 epochs')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wdecay', default=5e-4, type=float)
    parser.add_argument('--use-ema', action=argparse.BooleanOptionalAction, default=True,
                        help='record phase-1 predictions with the EMA model (as the original train.py does)')
    parser.add_argument('--ema-decay', default=0.99, type=float)
    parser.add_argument('--rho', default=0.1, type=float, help='label smoothing on labeled CE in phase 1')
    parser.add_argument('--alpha', default=0.00007, type=float,
                        help='ALM constraint target inside the penalty')
    parser.add_argument('--false_alarm_cutoff', default=0.05, type=float)
    parser.add_argument('--penalty_mult', default=1.05, type=float)
    parser.add_argument('--three-sigma', action=argparse.BooleanOptionalAction, default=False,
                        help="the original train.py's re-split with utils.misc.three_sigma when the Jenks break "
                             "is > 0; its cutoff is tuned for the image datasets and the original tabular script "
                             "(safe.py) disabled it")
    parser.add_argument('--device', default=None, type=str, help='default: cuda if available, else cpu')


def resolve_csvs(paths, exclude):
    excluded = set(exclude) | {'scaled_' + e for e in exclude}
    csvs = []
    for p in paths:
        if os.path.isdir(p):
            csvs.extend(f for f in sorted(glob.glob(os.path.join(p, '*.csv')))
                        if os.path.splitext(os.path.basename(f))[0] not in excluded)
        else:
            csvs.append(p)
    if not csvs:
        raise SystemExit(f"no csv files found in {paths}")
    return csvs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tabular(args, path):
    data = pd.read_csv(path, header=None)
    x = data.iloc[:, :-1].values.astype(np.float32)
    raw_y = data.iloc[:, -1].values
    if len(np.unique(raw_y)) != 2 or args.positive_class not in raw_y:
        raise ValueError(f"{path}: expected a binary label column containing {args.positive_class}, "
                         f"got values {np.unique(raw_y)[:10]}")
    y = np.where(raw_y == args.positive_class, 0, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_frac, stratify=y, random_state=args.seed)
    positive_idx = np.where(y_train == 0)[0]
    labeled_idx = np.random.choice(positive_idx, max(1, int(round(args.label_frac * len(positive_idx)))),
                                   False)
    unlabeled_idx = np.setdiff1d(np.arange(len(y_train)), labeled_idx)
    return build_sets(x_train[labeled_idx], x_train[unlabeled_idx], y_train[unlabeled_idx], x_test, y_test)


def build_sets(x_l, x_u, y_u, x_test, y_test):
    """Datasets from already split arrays; y_u / y_test use the repo convention (positive = 0)."""
    x_l, x_u, x_test = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in
                        (x_l, x_u, x_test))
    y_u, y_test = np.asarray(y_u), np.asarray(y_test)
    labeled_set = data_utils.TensorDataset(x_l, torch.zeros(len(x_l), dtype=torch.long))
    # (x, index into U for pseudo-label lookup, true label for diagnostics only)
    unlabeled_set = data_utils.TensorDataset(x_u, torch.arange(len(x_u)), torch.from_numpy(y_u).long())
    test_set = data_utils.TensorDataset(x_test, torch.from_numpy(y_test).long())

    info = {'n': len(x_l) + len(x_u) + len(x_test), 'features': x_l.shape[1], 'P': len(labeled_set),
            'U': len(unlabeled_set), 'U_pos': int((y_u == 0).sum()), 'test': len(test_set),
            'test_pos': int((y_test == 0).sum())}
    logger.info(' | '.join(f"{k}: {v}" for k, v in info.items()))
    return labeled_set, unlabeled_set, test_set, info


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def make_model_and_optim(args, in_features, total_steps):
    model = CardMLP(in_features).to(args.device)
    no_decay = ['bias']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
    return model, optimizer, scheduler


@torch.no_grad()
def predict_positive(model, dataset, device):
    """P(positive) for every sample, in dataset order."""
    model.eval()
    x = dataset.tensors[0].to(device)
    prob = torch.softmax(model(x), dim=1)[:, 0].cpu().numpy()
    model.train()
    return prob


def binary_metrics(y_true, pred, score_pos):
    # positive class is 0; report P/R/F1 for it and AUC on P(positive)
    return {
        'acc': accuracy_score(y_true, pred),
        'auc': roc_auc_score(y_true == 0, score_pos),
        'f1': f1_score(y_true, pred, pos_label=0, zero_division=0),
        'prec': precision_score(y_true, pred, pos_label=0, zero_division=0),
        'recall': recall_score(y_true, pred, pos_label=0, zero_division=0),
    }


def fmt(m):
    return ' '.join(f"{k} {v:.4f}" for k, v in m.items())


def train_phase1(args, labeled_set, unlabeled_set, in_features):
    model, optimizer, scheduler = make_model_and_optim(args, in_features,
                                                       args.warming_epochs * args.eval_step)
    ema_model = ModelEMA(SimpleNamespace(device=args.device), model, args.ema_decay) if args.use_ema else None

    labeled_iter = cycle(DataLoader(labeled_set, batch_size=args.batch_size, sampler=RandomSampler(
        labeled_set, replacement=True, num_samples=args.batch_size * args.eval_step)))
    unlabeled_iter = cycle(DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                      drop_last=len(unlabeled_set) >= args.batch_size))

    lam, beta = 0.0, 1.0
    preds_sequence = []
    model.train()
    for epoch in range(args.warming_epochs):
        ln_total = 0.0
        for _ in range(args.eval_step):
            inputs_x, targets_x = next(labeled_iter)
            inputs_u, _, _ = next(unlabeled_iter)
            batch_size = inputs_x.shape[0]
            logits = model(torch.cat((inputs_x, inputs_u)).to(args.device))
            logits_x, logits_u = logits[:batch_size], logits[batch_size:]
            targets_x = targets_x.to(args.device)
            targets_u = torch.ones(len(inputs_u), dtype=torch.long, device=args.device)

            Lx = F.cross_entropy(logits_x, targets_x, label_smoothing=args.rho)
            Ln = F.cross_entropy(logits_u, targets_u)
            # augmented-Lagrangian (PHR) penalty on the "unlabeled are negative" loss
            Ln_tempo = Ln - args.alpha
            if beta * Ln_tempo.item() + lam >= 0:
                penalty = Ln_tempo * lam + (beta / 2) * Ln_tempo * Ln_tempo
            else:
                penalty = torch.tensor(-(lam * lam) / (2 * beta), device=args.device)
            loss = Lx + Ln + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                ema_model.update(model)
            ln_total += Ln.item()

        # multiplier / penalty updates once per epoch, as in the original train.py
        mean_ln = ln_total / args.eval_step
        constraint = mean_ln - args.false_alarm_cutoff
        if constraint * beta + lam >= 0:
            lam += constraint
        else:
            lam += -lam / beta
        if mean_ln - args.alpha >= 0:
            beta *= args.penalty_mult

        record_model = ema_model.ema if ema_model is not None else model
        preds_sequence.append(predict_positive(record_model, unlabeled_set, args.device))
        logger.debug(f"[phase 1] epoch {epoch + 1}/{args.warming_epochs} loss {loss.item():.4f} "
                     f"Lx {Lx.item():.4f} Ln {mean_ln:.4f} lam {lam:.4f} beta {beta:.4f}")

    # trend of each unlabeled sample's P(positive) across epochs
    preds_sequence = np.stack(preds_sequence, axis=1)
    diff_1 = np.diff(preds_sequence, axis=1)
    trends = np.log(1 + diff_1 + 0.5 * diff_1 ** 2).mean(axis=1)

    intervals = jenkspy.jenks_breaks(trends, n_classes=2)
    break_point = intervals[1]
    if args.three_sigma and break_point > 0:
        trends_std = three_sigma(trends)
        if len(trends_std) > 2:
            intervals = jenkspy.jenks_breaks(trends_std, n_classes=2)
            break_point = intervals[1]
    logger.debug(f"The interval is {intervals}; Break Point is {break_point}")
    pseudo_targets = np.where(trends > break_point, 0, 1)

    true_u = unlabeled_set.tensors[2].numpy()
    stats = {'est_prior': (pseudo_targets == 0).mean(), 'true_prior': (true_u == 0).mean()}
    pl_metrics = binary_metrics(true_u, pseudo_targets, trends)
    logger.info(
        f"Estimated positive fraction in U: {stats['est_prior']:.4f} (true {stats['true_prior']:.4f})")
    logger.info(f"Pseudo-labels vs. truth: {fmt(pl_metrics)}")
    stats.update({f'pl_{k}': pl_metrics[k] for k in ('f1', 'prec', 'recall')})
    return pseudo_targets, stats


def train(args, labeled_set, unlabeled_set, test_set, pseudo_targets, in_features):
    model, optimizer, scheduler = make_model_and_optim(args, in_features, args.ft_epochs * args.eval_step)
    pseudo_targets = torch.from_numpy(pseudo_targets).long()

    labeled_iter = cycle(DataLoader(labeled_set, batch_size=args.batch_size, sampler=RandomSampler(
        labeled_set, replacement=True, num_samples=args.batch_size * args.eval_step)))
    unlabeled_iter = cycle(DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                      drop_last=len(unlabeled_set) >= args.batch_size))
    y_test = test_set.tensors[1].numpy()

    best = None
    model.train()
    for epoch in range(args.ft_epochs):
        lamda = (epoch / args.ft_epochs) ** 0.8
        for _ in range(args.eval_step):
            inputs_x, targets_x = next(labeled_iter)
            inputs_u, idx_u, _ = next(unlabeled_iter)
            batch_size = inputs_x.shape[0]
            logits = model(torch.cat((inputs_x, inputs_u)).to(args.device))
            logits_x, logits_u = logits[:batch_size], logits[batch_size:]

            Lx = F.cross_entropy(logits_x, targets_x.to(args.device))
            label_u = F.one_hot(torch.ones(len(inputs_u), dtype=torch.long), 2).float()
            label_p = F.one_hot(pseudo_targets[idx_u], 2).float()
            label = (lamda * label_p + (1 - lamda) * label_u).to(args.device)
            Lu = F.cross_entropy(logits_u, label)
            loss = Lx + Lu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        score_pos = predict_positive(model, test_set, args.device)
        metrics = binary_metrics(y_test, np.where(score_pos >= 0.5, 0, 1), score_pos)
        if best is None or metrics['acc'] > best['acc']:
            best = metrics
        logger.debug(
            f"[phase 2] epoch {epoch + 1}/{args.ft_epochs} loss {loss.item():.4f} | test {fmt(metrics)}")

    logger.info(f"Final test: {fmt(metrics)}")
    logger.debug(f"Best-acc epoch test (selected on test, optimistic): {fmt(best)}")
    return metrics, score_pos


def run(args, path, seed):
    args = argparse.Namespace(**{**vars(args), 'seed': seed})
    set_seed(seed)
    name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"===== {name} (seed {seed}) =====")
    labeled_set, unlabeled_set, test_set, info = get_tabular(args, path)
    pseudo_targets, pl_stats = train_phase1(args, labeled_set, unlabeled_set, info['features'])
    test_metrics, _ = train(args, labeled_set, unlabeled_set, test_set, pseudo_targets, info['features'])
    return {'dataset': name, 'seed': seed, **info, **pl_stats, **test_metrics}


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    args.device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info({k: v for k, v in vars(args).items() if k != 'data'})

    csvs = resolve_csvs(args.data, args.exclude)
    rows = []
    for path in csvs:
        for seed in args.seeds:
            try:
                rows.append(run(args, path, seed))
            except ValueError as e:
                logger.error(f"skipping: {e}")
            # write after every run so partial results survive an interrupted sweep
            if rows:
                os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    if not rows:
        return

    results = pd.DataFrame(rows)
    cols = ['est_prior', 'true_prior', 'pl_f1'] + METRICS
    summary = results.groupby('dataset', sort=False)[cols].agg(
        ['mean', 'std'] if len(args.seeds) > 1 else ['mean'])
    summary.columns = [f"{c}_{s}" if len(args.seeds) > 1 else c for c, s in summary.columns]
    print(f"\nTest metrics are for the final epoch; F1/precision/recall are for the positive class. "
          f"{len(args.seeds)} seed(s). Per-run results: {args.out}\n")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    if len(csvs) > 1:
        print('\nMean over datasets:\n' + results.groupby('dataset')[METRICS].mean().mean().to_string(
            float_format=lambda v: f"{v:.3f}"))


if __name__ == '__main__':
    main()
