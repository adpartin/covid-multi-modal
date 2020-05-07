""" 
This code trains ML models using multiple sets of features:
    [dsc], [dsc, ecfp2], ...
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pformat
from glob import glob

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.externals import joblib

import lightgbm as lgb

# File path
filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
# from gen_prd_te_table import update_prd_table
from utils.utils import verify_path, load_data, dump_dict, get_print_func
from datasplit.split_getter import get_unq_split_ids, get_data_by_id
from ml.scale import scale_fea
from ml.evals import calc_preds, dump_preds, calc_scores
from ml.data import extract_subset_fea


def parse_args(args):
    parser = argparse.ArgumentParser(description='Large cross-validation runs.')
    parser.add_argument('-td', '--trg_dir', required=True, default=None, type=str,
                        help='Target dir that contains the feature types. (default: None).')
    parser.add_argument('--fea', default=['dsc'], type=str, nargs='+',
                        help='Feature types (default: [dsc]).')
    parser.add_argument('--gout', default=None, type=str, help='Gloabl outdir (default: None).')
    parser.add_argument('-ns', '--n_splits', default=5, type=int, help='Use a subset of splits (default: 10).')
    parser.add_argument('-t', '--target_name', default='reg', choices=['reg', 'cls', 'binner'],
                        type=str, help='Name of target variable (default: reg).')
    parser.add_argument('-sc', '--scaler', default=None, type=str, choices=['stnd', 'minmax', 'rbst'],
                        help='Feature normalization method (stnd, minmax, rbst) (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args
        

def trn_lgbm_model(model, xtr, ytr, fit_kwargs, eval_set=None):
    """ Train and save LigthGBM model. """
    fit_kwargs = fit_kwargs
    # fit_kwargs = self.fit_kwargs.copy()
    fit_kwargs['eval_set'] = eval_set
    fit_kwargs['early_stopping_rounds'] = 10
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    runtime = (time() - t0)/60
    fit_kwargs.pop('eval_set', None)
    # joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
    return model, runtime


def trn_sklearn_model(model, xtr, ytr, fit_kwargs, eval_set=None):
    """ Train and save sklearn model. """
    fit_kwargs = fit_kwargs
    # fit_kwargs = self.fit_kwargs.copy()
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    runtime = (time() - t0)/60
    # joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
    return model, runtime


def scores_to_df(scores_all):
    """ Dict to df. """
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['run'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['run'], columns=['metric'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


def get_model_kwargs(args):
    """ Get ML model init and fit args. """
    if args['framework'] == 'lightgbm':
        model_init_kwargs = { 'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
                              'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
                              'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
        model_fit_kwargs = {'verbose': False}

    elif args['framework'] == 'sklearn':
        model_init_kwargs = { 'n_estimators': args['rf_trees'], 'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
        model_fit_kwargs = {}

    elif args['framework'] == 'keras':
        model_init_kwargs = { 'input_dim': data.shape[1], 'dr_rate': args['dr_rate'],
                              'opt_name': args['opt'], 'lr': args['lr'], 'batchnorm': args['batchnorm']}
        model_fit_kwargs = { 'batch_size': args['batch_size'], 'epochs': args['epochs'], 'verbose': 1 }        
    
    return model_init_kwargs, model_fit_kwargs


def trn_single_split( split_id, all_split_files, xdata, ydata, meta,
        ml_model_def, ml_init_kwargs, ml_fit_kwargs, framework, mltype, outdir ):
    """ ... """
    split_pattern = f'1fold_s{split_id}'
    single_split_files = [p for p in all_split_files if split_pattern in p]
    if len(single_split_files) < 2:
        print_fn(f'the split {split_id} contains only one file.')
        return None

    for id_file in single_split_files:
        if 'tr_id' in id_file:
            tr_id = load_data( id_file ).values.reshape(-1,)
        elif 'vl_id' in id_file:
            vl_id = load_data( id_file ).values.reshape(-1,)
        elif 'te_id' in id_file:
            te_id = load_data( id_file ).values.reshape(-1,)

    rout = outdir
    os.makedirs(rout, exist_ok=True)

    # Extract T/V/E sets
    xtr, ytr, mtr = get_data_by_id(tr_id, xdata, ydata, meta)
    xvl, yvl, mvl = get_data_by_id(vl_id, xdata, ydata, meta)
    xte, yte, mte = get_data_by_id(te_id, xdata, ydata, meta)

    model = ml_model_def( **ml_init_kwargs )

    # Train
    eval_set = (xvl, yvl)
    if framework=='lightgbm':
        model, runtime = trn_lgbm_model(model=model, fit_kwargs=ml_fit_kwargs, xtr=xtr, ytr=ytr, eval_set=eval_set)
    elif framework=='sklearn':
        model, runtime = trn_sklearn_model(model=model, xtr=xtr, ytr=ytr, eval_set=None)
    elif framework=='keras':
        model, runtime = trn_keras_model(model=model, xtr=xtr, ytr=ytr, eval_set=eval_set)
    elif framework=='pytorch':
        raise ValueError(f'framework {framework} is not yet supported.')
    else:
        raise ValueError(f'framework {framework} is not yet supported.')

    # Calc preds and scores
    def pred_and_dump(model, X, Y, M, mltype, sufx):
        y_pred, y_true = calc_preds(model, x=X, y=Y, mltype=mltype)
        scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=M, outpath=rout/f'preds_{sufx}.csv')
        return scores

    tr_scores = pred_and_dump(model, xtr, ytr, mtr, mltype, sufx='tr')
    vl_scores = pred_and_dump(model, xvl, yvl, mvl, mltype, sufx='vl')
    te_scores = pred_and_dump(model, xte, yte, mte, mltype, sufx='te')

    # Add metadata
    tr_scores['run'] = split_id
    vl_scores['run'] = split_id
    te_scores['run'] = split_id

    # Append scores (dicts)
    res = {'tr_scores': tr_scores, 'vl_scores': vl_scores,
           'te_scores': te_scores, 'runtime': runtime}
    del xtr, ytr, mtr, xvl, yvl, mvl, xte, yte, mte, eval_set, model
    return res


def run(args):
    t0 = time()

    trg_dir = Path(args['trg_dir']).resolve()
    trg_name = trg_dir.name
    splitdir = list(trg_dir.glob('ml.*.splits'))[0]

    fea_prfx_dct = {'dsc':'dsc', 'ecfp2':'ecfp2', 'ecfp4':'ecfp4', 'ecfp6':'ecfp6'}
    file_format = 'parquet'

    fea_list = args['fea']
    data_dct = {}
    for fea_name in fea_list:
        fname = f'ml.{trg_name}.{fea_name}.{file_format}'
        fpath = trg_dir/fname
        df = load_data( fpath ) 
        data_dct[fea_name] = df

    # outdir
    if args['gout'] is not None:
        gout = Path( args['gout'] ).resolve()
    else:
        gout = filepath.parent/'trn'
        gout = gout/trg_name
    args['gout'] = str(gout)
    os.makedirs(gout, exist_ok=True)

    lg = Logger( gout/'lc.log' )
    print_fn = get_print_func( lg.logger )
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=gout/'trn.args.txt')

    # --------------------
    # Multi-model 
    # --------------------
    # Compute possible combos
    fea_list = data_dct.keys()
    fea_combos = []
    from itertools import combinations
    for r in range(1, len(fea_list)+1):
    # for r in range(2, len(fea_list)+1):
        cc = sorted(combinations(fea_list, r))
        fea_combos.extend( cc )

    for fea_comb in fea_combos:
        fea_name = '_'.join([i for i in fea_comb])
        print_fn(f'\nProcessing {fea_name}')

        dfs = []
        for fea in fea_comb:
            dfs.append( data_dct[fea] )
        if len(dfs) > 1:
            data = pd.concat( dfs, axis=1 )
            data = data.iloc[:, ~data.columns.duplicated() ]
        else:
            data = dfs[0]

        # Get features (x), target (y), and meta
        xdata = extract_subset_fea(data, fea_list=fea_comb, fea_sep='.')
        meta = data.drop( columns=xdata.columns )
        ydata = meta[[ args['target_name'] ]]
        del data

        # Scaling
        xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])  # scale features

        all_split_files = glob(str( splitdir/'1fold_*_id.csv' ))
        unq_split_ids = get_unq_split_ids( all_split_files )
        run_times = []

        # Append scores (dicts)
        tr_scores_all = []
        vl_scores_all = []
        te_scores_all = []

        # Iterate over splits
        n_splits = np.min([ len(unq_split_ids), args['n_splits'] ])
        for i, split_id in enumerate(unq_split_ids[:n_splits]):
            print(f'Train split {split_id}')

            # LGBM regressor model definition
            framework = 'lightgbm'
            ml_model_def = lgb.LGBMRegressor
            mltype = 'reg'
            ml_init_kwargs = { 'n_estimators': 100, 'max_depth': -1,
                               'learning_rate': 0.1, 'num_leaves': 31,
                               'n_jobs': 8, 'random_state': None }
            ml_fit_kwargs = {'verbose': False, 'early_stopping_rounds': 10}
            keras_callbacks_def = None
            keras_clr_kwargs = None

            # Train single splits
            outdir = gout/fea_name/f'run_{split_id}'
            res = trn_single_split( split_id, all_split_files, xdata, ydata, meta,
                    ml_model_def, ml_init_kwargs, ml_fit_kwargs, framework, mltype, outdir )

            run_times.append( res['runtime'] )
            tr_scores_all.append( res['tr_scores'] )
            vl_scores_all.append( res['vl_scores'] )
            te_scores_all.append( res['te_scores'] )

        # Scores to df
        tr_scores_df = scores_to_df( tr_scores_all )
        vl_scores_df = scores_to_df( vl_scores_all )
        te_scores_df = scores_to_df( te_scores_all )

        outdir = gout/fea_name
        tr_scores_df.to_csv(outdir/'tr_scores.csv', index=False)
        vl_scores_df.to_csv(outdir/'vl_scores.csv', index=False)
        te_scores_df.to_csv(outdir/'te_scores.csv', index=False)

    del tr_scores_df, vl_scores_df, te_scores_df
    # ---------------------------------------------------------------
    if (time()-t0)//3600 > 0:
        print_fn('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )

    print_fn('Done.')
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])


