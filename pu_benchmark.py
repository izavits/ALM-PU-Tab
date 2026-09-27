"""Run ALM-PU (tabular.py, CardMLP) in the pu_learning benchmark setting (see pu_setting.py):
70/30 stratified split with random_state = run index, SCAR/SAR labeling that hides `pu_ratio` of
the training positives, and validate_sarem's metrics on P(positive).

Results are written one per setting:
    {out}/{EXPERIMENT}_{MECHANISM}_{PU_RATIO}_default_ALMPU/results.csv   (run,dataset,pu_ratio,labeling_mechanism,roc_auc,ap,f1)
    {out}/{EXPERIMENT}_{MECHANISM}_{PU_RATIO}_default_ALMPU/runs_detail.csv (+ other metrics, pseudo-label diagnostics)
and, when --baseline-dir exists, compared per setting against another method's results.
"""
import argparse
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch

import tabular
from pu_setting import prepare_pu_data_plain, validate_scores

logger = logging.getLogger(__name__)

# pu_learning's config.ini experiment names for files whose name is not simply upper-cased
EXPERIMENT_NAMES = {
    'scaled_piechart2': 'SCALED_PIECHART',
    'scaled_pizzacutter1': 'SCALED_PIZZACUTTER',
    'scaled_poker-8-9_vs_5': 'SCALED_POKER',
    'scaled_segment0': 'SCALED_SEGMENT',
    'scaled_winequality-red-4': 'SCALED_WINEQUALITY',
}
SCORES = ['roc_auc', 'ap', 'f1']


def parse_args():
    parser = argparse.ArgumentParser(description='ALM-PU in the pu_learning (main_LBE.py) setting')
    parser.add_argument('--data', nargs='+', default=['./dataset'],
                        help='csv file(s) and/or directories (every *.csv inside is used)')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help="dataset names skipped when expanding directories")
    parser.add_argument('--pu-ratios', nargs='+', default=[0.25, 0.5, 0.75], type=float,
                        help='fraction of training positives hidden in U (PU_RATIO in pu_learning)')
    parser.add_argument('--mechanisms', nargs='+', default=['SCAR', 'SAR'], choices=['SCAR', 'SAR'])
    parser.add_argument('--runs', default=10, type=int, help='runs per setting; run i uses random_state=i')
    parser.add_argument('--out', default='output_ALMPU', type=str)
    parser.add_argument('--overwrite', action='store_true', help='rerun settings that already have results')
    parser.add_argument('--workers', default=max(1, (os.cpu_count() or 2) - 2), type=int,
                        help='parallel processes, one setting (dataset, mechanism, ratio) each')
    parser.add_argument('--summary-only', action='store_true', help='only summarize existing results')
    parser.add_argument('--baseline-dir', default='../pu_learning/output_LBE', type=str)
    parser.add_argument('--baseline-name', default='LBE', type=str,
                        help="baseline folders are {EXPERIMENT}_{MECH}_{RATIO}_default_{name}")
    tabular.add_method_args(parser)
    return parser.parse_args()


def experiment_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return EXPERIMENT_NAMES.get(stem, stem.upper())


def setting_dir(root, experiment, mech, ratio, method):
    return os.path.join(root, f'{experiment}_{mech}_{ratio}_default_{method}')


def run_setting(args, path, mech, ratio):
    """All runs of one (dataset, mechanism, ratio) setting; returns per-run rows."""
    experiment = experiment_name(path)
    rows = []
    for i in range(args.runs):
        x_train, y_, x_test, y_test, _, y_train = prepare_pu_data_plain(path, ratio, mech, random_state=i)
        labeled = y_.ravel() == 1
        # tabular.py uses positive = 0; pu_learning uses positive = 1
        to_repo = lambda y: np.where(y.ravel() == 1, 0, 1)
        tabular.set_seed(i)
        labeled_set, unlabeled_set, test_set, info = tabular.build_sets(
            x_train[labeled], x_train[~labeled], to_repo(y_train[~labeled]), x_test, to_repo(y_test))
        pseudo_targets, pl_stats = tabular.train_phase1(args, labeled_set, unlabeled_set, info['features'])
        _, score_pos = tabular.train(args, labeled_set, unlabeled_set, test_set, pseudo_targets, info['features'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            metrics = validate_scores(y_test.astype(int), score_pos)
        rows.append({'run': i, 'dataset': experiment, 'pu_ratio': ratio, 'labeling_mechanism': mech,
                     **metrics, **info, **pl_stats})

    out_dir = setting_dir(args.out, experiment, mech, ratio, 'ALMPU')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df[['run', 'dataset', 'pu_ratio', 'labeling_mechanism'] + SCORES].to_csv(
        os.path.join(out_dir, 'results.csv'), index=False)
    df.to_csv(os.path.join(out_dir, 'runs_detail.csv'), index=False)
    return experiment, mech, ratio, df


def init_worker():
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings('ignore')


def load_method(root, experiments, mechanisms, ratios, method):
    """Per-run roc_auc/ap/f1 from pu_learning-style results.csv files."""
    frames = []
    for experiment in experiments:
        for mech in mechanisms:
            for ratio in ratios:
                path = os.path.join(setting_dir(root, experiment, mech, ratio, method), 'results.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)[['run'] + SCORES]
                    frames.append(df.assign(dataset=experiment, labeling_mechanism=mech, pu_ratio=ratio))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(args, experiments):
    ours = load_method(args.out, experiments, args.mechanisms, args.pu_ratios, 'ALMPU')
    if ours.empty:
        print(f'No results found in {args.out}')
        return
    keys = ['dataset', 'labeling_mechanism', 'pu_ratio']
    base = load_method(args.baseline_dir, experiments, args.mechanisms, args.pu_ratios, args.baseline_name) \
        if os.path.isdir(args.baseline_dir) else pd.DataFrame()
    if base.empty:
        print(f'\nALM-PU mean ROC-AUC (no baseline found at {args.baseline_dir})\n')
        print(ours.pivot_table(index='dataset', columns=['labeling_mechanism', 'pu_ratio'], values='roc_auc',
                               sort=False).to_string(float_format=lambda v: f'{v:.3f}'))
        return

    b = args.baseline_name
    # pair by run index so both methods are averaged over exactly the same splits
    paired = ours.merge(base, on=keys + ['run'], suffixes=('_ALMPU', f'_{b}'))
    comp = paired.groupby(keys, sort=False).agg(
        runs=('run', 'size'),
        **{f'{m}_{s}': (f'{m}_{s}', 'mean') for m in SCORES for s in ('ALMPU', b)},
        **{f'std_{m}_{s}': (f'{m}_{s}', 'std') for m in SCORES for s in ('ALMPU', b)}).reset_index()
    for m in SCORES:
        comp[f'{m}_diff'] = comp[f'{m}_ALMPU'] - comp[f'{m}_{b}']
    comp.to_csv(os.path.join(args.out, f'comparison_{b}.csv'), index=False)

    runs = sorted(comp.runs.unique())
    print(f'\nALM-PU vs {b}: mean over {"/".join(map(str, runs))} paired runs per dataset; '
          f'each cell is ALM-PU / {b}')
    for mech in args.mechanisms:
        for ratio in args.pu_ratios:
            c = comp[(comp.labeling_mechanism == mech) & (comp.pu_ratio == ratio)]
            if c.empty:
                continue
            table = pd.DataFrame({m: [f'{x:.3f} / {y:.3f}' for x, y in zip(c[f'{m}_ALMPU'], c[f'{m}_{b}'])]
                                  for m in SCORES}, index=c.dataset)
            table.loc['MEAN'] = [f"{c[f'{m}_ALMPU'].mean():.3f} / {c[f'{m}_{b}'].mean():.3f}" for m in SCORES]
            table.loc['ALM-PU wins'] = [f"{(c[f'{m}_diff'] > 0).sum()} / {len(c)}" for m in SCORES]
            print(f'\n--- {mech}, pu_ratio {ratio} ---')
            print(table.to_string())
    print(f'\nFull table: {os.path.join(args.out, f"comparison_{b}.csv")}')


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    tabular.logger.setLevel(logging.WARNING)
    args.device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    csvs = tabular.resolve_csvs(args.data, args.exclude)
    experiments = [experiment_name(p) for p in csvs]

    if not args.summary_only:
        tasks = [(p, mech, ratio) for p in csvs for mech in args.mechanisms for ratio in args.pu_ratios]
        if not args.overwrite:
            tasks = [t for t in tasks if not os.path.exists(os.path.join(
                setting_dir(args.out, experiment_name(t[0]), t[1], t[2], 'ALMPU'), 'results.csv'))]
        # biggest datasets first for better load balancing
        tasks.sort(key=lambda t: -os.path.getsize(t[0]))
        logger.info(f'{len(tasks)} settings x {args.runs} runs, {args.workers} workers, device {args.device}')
        start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'),
                                 initializer=init_worker) as pool:
            futures = [pool.submit(run_setting, args, *t) for t in tasks]
            for done, fut in enumerate(as_completed(futures), 1):
                experiment, mech, ratio, df = fut.result()
                logger.info(f'[{done}/{len(tasks)}] {experiment} {mech} {ratio}: ' +
                            ' '.join(f'{m} {df[m].mean():.3f}' for m in SCORES) +
                            f' ({(time.time() - start) / 60:.1f} min)')

    summarize(args, experiments)


if __name__ == '__main__':
    main()
