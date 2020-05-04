""" 
This code trains ML model using multple splits of train/val/test sets.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.externals import joblib

# File path
filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
# import ml_models
# from gen_prd_te_table import update_prd_table
from utils.utils import verify_path, load_data, dump_dict, get_print_func
from datasplit.split_getter import get_unq_split_ids, get_data_by_id
from ml import ml_models
from ml.scale import scale_fea
from ml.evals import calc_preds, dump_preds, calc_scores #, scores_to_df
from ml.data import extract_subset_fea

# Default settings
# OUT_DIR = filepath / 'out'    
# FILE_PATH = filepath / 'data/top_21.res_reg.cf_rnaseq.dd_dragon7.labled.r0.parquet'
# FILE_PATH = filepath / 'data/top_21.res_bin.cf_rnaseq.dd_dragon7.labled.r0.parquet'
# SPLITS_DIR = filepath / 'splits_old'

        
def parse_args(args):
    parser = argparse.ArgumentParser(description='Large cross-validation runs.')

    # Input data
    parser.add_argument('-dp', '--datapath', required=True, default=None, type=str,
                        help='Full path to data (default: None).')

    # Path to splits
    parser.add_argument('-sp', '--splitpath', required=True, default=None, type=str,
                        help='Full path to data splits (default: None).')
    parser.add_argument('-ns', '--n_splits', default=10, type=int, help='Use a subset of splits (default: 10).')

    # Global outdir
    parser.add_argument('--gout', default=None, type=str, help='Gloabl outdir (default: out).')

    # Select target to predict
    parser.add_argument('-t', '--target_name', default='reg', choices=['reg', 'cls', 'binner'],
                        type=str, help='Name of target variable (default: reg).')

    # Select feature types
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['mod'], choices=['mod'],
                        help='Drug features (default: mod).')

    # Drop a specific range of target values 
    parser.add_argument('--min_gap', default=None, type=float, help='Min gap of AUC value (default: None).')
    parser.add_argument('--max_gap', default=None, type=float, help='Max gap of AUC value (default: None).')

    # List of samples to drop
    parser.add_argument('--drop_smiles', default=None, type=str, help='A list of drugs to drop (default: None).')

    # Data split methods
    # TODO: is this still required??
    parser.add_argument('-cvf', '--cv_folds', default=1, type=str, help='Number cross-val folds (default: 1).')
    parser.add_argument('-cvf_arr', '--cv_folds_arr', nargs='+', type=int, default=None,
                        help='The specific folds in the cross-val run (default: None).')
    
    # ML models
    parser.add_argument('-frm', '--framework', default='lightgbm', type=str, choices=['keras', 'lightgbm', 'sklearn'], help='ML framework (default: lightgbm).')
    parser.add_argument('-ml', '--model_name', default='lgb_reg', type=str,
                        choices=['lgb_reg', 'rf_reg', 'nn_reg', 'lgb_cls'], help='ML model (default: lgb_cls).')
    parser.add_argument('--save_model', default=None, help='Save ML model (default: None).')

    # LightGBM params
    parser.add_argument('--gbm_leaves', default=31, type=int, help='Maximum tree leaves for base learners (default: 31).')
    parser.add_argument('--gbm_lr', default=0.1, type=float, help='Boosting learning rate (default: 0.1).')
    parser.add_argument('--gbm_max_depth', default=-1, type=int, help='Maximum tree depth for base learners (default: -1).')
    parser.add_argument('--gbm_trees', default=100, type=int, help='Number of trees (default: 100).')
    parser.add_argument('--gbm_cls_weight', default=None, type=str, choices=[None, 'balanced'],
            help='Whether to weight the classes (default: None).')
    
    # Random Forest params
    parser.add_argument('--rf_trees', default=100, type=int, help='Number of trees (default: 100).')   
    
    # NN hyper_params
    parser.add_argument('-ep', '--epochs', default=200, type=int, help='Number of epochs (default: 200).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    parser.add_argument('-sc', '--scaler', default='stnd', type=str, choices=['stnd', 'minmax', 'rbst'],
                        help='Feature normalization method (stnd, minmax, rbst) (default: stnd).')
    parser.add_argument('--batchnorm', action='store_true', help='Whether to use batch normalization (default: False).')
    # parser.add_argument('--residual', action='store_true', help='Whether to use residual conncetion (default: False).')
    # parser.add_argument('--initializer', default='he', type=str, choices=['he', 'glorot'], help='Keras initializer name (default: he).')

    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer name (default: sgd).')
    parser.add_argument('--lr', default='0.0001', type=float, help='Learning rate of adaptive optimizers (default: 0.001).')

    parser.add_argument('--clr_mode', default=None, type=str, choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1).')
    parser.add_argument('--clr_base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--clr_max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--clr_gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    # Other
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 8.')
    parser.add_argument('--seed', default=0, type=int, help='Default: 0.')

    # Parse args
    args, other_args = parser.parse_known_args(args)
    return args
        

def trn_lgbm_model(model, xtr, ytr, fit_kwargs, eval_set=None):
    """ Train and save LigthGBM model. """
    # Fit params
    fit_kwargs = fit_kwargs
    # fit_kwargs = self.fit_kwargs.copy()
    fit_kwargs['eval_set'] = eval_set
    fit_kwargs['early_stopping_rounds'] = 10

    # Train and save model
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    runtime = (time() - t0)/60

    # Remove key (we'll dump this dict so we don't need to print all the eval set)
    fit_kwargs.pop('eval_set', None)

    # joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
    return model, runtime


def trn_sklearn_model(model, xtr, ytr, fit_kwargs, eval_set=None):
    """ Train and save sklearn model. """
    # Fit params
    fit_kwargs = fit_kwargs
    # fit_kwargs = self.fit_kwargs.copy()

    # Train and save model
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    runtime = (time() - t0)/60
    # joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
    return model, runtime


def create_trn_outdir(fold, tr_sz):
    trn_outdir = outdir / ('cv'+str(fold) + '_sz'+str(tr_sz))
    os.makedirs(trn_outdir, exist_ok=True)
    return trn_outdir
    

def drop_samples(x_df, y_df, m_df, drop_items, drop_by):
    """
    Args:
        drop_by : col in df ('CELL', 'DRUG', 'CTYPE')
    """
    id_drop = m_df[drop_by].isin( drop_items )
    x_df = x_df[~id_drop].reset_index(drop=True)
    y_df = y_df[~id_drop].reset_index(drop=True)
    m_df = m_df[~id_drop].reset_index(drop=True)
    return x_df, y_df, m_df


def scores_to_df(scores_all):
    """ Dict to df. """
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['run'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['run'], columns=['metric'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


# def bin_rsp(y, resp_thres=0.5):
#     """ Binarize drug response values. """
#     y = pd.Series( [0 if v>resp_thres else 1 for v in y.values] )
#     return y


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


def run(args):
    t0 = time()
    args['datapath'] = str(Path(args['datapath']).absolute())
    datapath = args['datapath']
    
    # ML type ('reg' or 'cls')
    if 'reg' in args['model_name']:
        mltype = 'reg'
    elif 'cls' in args['model_name']:
        mltype = 'cls'
    else:
        raise ValueError("model_name must contain 'reg' or 'cls'.")
        
    # -----------------------------------------------
    #       Create outdir
    # -----------------------------------------------
    # Global outdir
    # gout = Path(args['gout'])
    # gout = filepath/'./' if args['gout'] is None else Path(args['gout'])
    if args['gout'] is not None:
        gout = Path( args['gout'] )
    else:
        # dir_sufx = '.trn'
        # gout = '.'.join( str( Path(args['datapath']).absolute() ).split('.')[:-1] )
        # gout = Path(gout + dir_sufx)
        # This is unique to every project!!
        split_str_on_sep = datapath.split('/data/')
        dir1 = split_str_on_sep[0] + '/trn'
        dir2 = Path( split_str_on_sep[1] ).with_suffix('')
        gout = Path(dir1, dir2)
    os.makedirs(gout, exist_ok=True)
    # args['gout'] = str(gout)

    # -----------------------------------------------
    #       Create logger
    # -----------------------------------------------
    lg = Logger(gout/'trn.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=gout/'trn.args.txt') # dump args.
    
    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    data = load_data( args['datapath'] )
    print_fn('data.shape {}'.format(data.shape))
    # print_fn('Total mod: {}'.format( len([c for c in data.columns if 'mod.' in c]) ))
    
    # Get features (x), target (y), and meta
    fea_list = args['drug_fea']
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep='.')
    meta = data.drop( columns=xdata.columns )
    ydata = meta[[ args['target_name'] ]]
    del data

    # -----------------------------------------------
    #       Data splits
    # -----------------------------------------------
    all_split_files = glob(str(Path(args['splitpath'])/'1fold_*_id.csv'))
    unq_split_ids = get_unq_split_ids(all_split_files)
    run_times = []

    # Append scores (dicts)
    tr_scores_all = []
    vl_scores_all = []
    te_scores_all = []

    # Sample size at each run
    smp_sz = []
    file_smp_sz = open(gout/'sample_sz', 'w')
    file_smp_sz.write('run\ttr_sz\tvl_sz\tte_sz\n')

    # Iterate over splits
    n_splits = np.min([ len(unq_split_ids), args['n_splits'] ])
    for i, split_id in enumerate(unq_split_ids[:n_splits]):
        print(f'Train split {split_id}')

        # Get indices for the split
        single_split_files = [p for p in all_split_files if f'1fold_{split_id}' in p]
        if len(single_split_files) < 2:
            print_fn(f'The split {s} contains only one file.')
            continue
        for id_file in single_split_files:
            if 'tr_id' in id_file:
                tr_id = load_data( id_file )
            elif 'vl_id' in id_file:
                vl_id = load_data( id_file )
            elif 'te_id' in id_file:
                te_id = load_data( id_file )

        # Create run outdir
        rout = gout/f'run_{split_id}'
        os.makedirs(rout, exist_ok=True)

        # Scaling
        # xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])  # scale features

        # Get training and val data
        # Extract Train set T, Validation set V, and Test set E
        tr_id = tr_id.iloc[:,0].values.astype(int).tolist()
        vl_id = vl_id.iloc[:,0].values.astype(int).tolist()
        te_id = te_id.iloc[:,0].values.astype(int).tolist()
        xtr, ytr, mtr = get_data_by_id(tr_id, xdata, ydata, meta) # samples from xtr are sequentially sampled for TRAIN
        xvl, yvl, mvl = get_data_by_id(vl_id, xdata, ydata, meta) # fixed set of VAL samples for the current CV split
        xte, yte, mte = get_data_by_id(te_id, xdata, ydata, meta) # fixed set of TEST samples for the current CV split

        # Dump samples
        drop_sample_list = gout/'drop_smiles.csv'
        # if args['drop_sample_list'] is not None:
        if drop_sample_list.exists():
            drop_items = load_data( drop_sample_list )['smiles'].values
            # drop_items = load_data( args['drop_sample_list'] )
            xtr, ytr, mtr = drop_samples(x_df=xtr, y_df=ytr, m_df=mtr, drop_items=drop_items, drop_by='smiles')
            xvl, yvl, mvl = drop_samples(x_df=xvl, y_df=yvl, m_df=mvl, drop_items=drop_items, drop_by='smiles')
            xte, yte, mte = drop_samples(x_df=xte, y_df=yte, m_df=mte, drop_items=drop_items, drop_by='smiles')

        line = 's{}\t{}\t{}\t{}\n'.format(split_id, xtr.shape[0], xvl.shape[0], xte.shape[0])
        file_smp_sz.write(line)

        # Define ML model
        if 'lgb' in args['model_name']:
            args['framework'] = 'lightgbm'
        elif args['model_name'] == 'rf_reg':
            args['framework'] = 'sklearn'
        elif 'nn_' in args['model_name']:
            args['framework'] = 'keras'

        model_init_kwargs, model_fit_kwargs = get_model_kwargs(args)

        # Get the estimator
        estimator = ml_models.get_model(args['model_name'], init_kwargs=model_init_kwargs)
        model = estimator.model
        
        # Train
        eval_set = (xvl, yvl)
        if args['framework']=='lightgbm':
            model, runtime = trn_lgbm_model(model=model, xtr=xtr, ytr=ytr,
                                            eval_set=eval_set, fit_kwargs=model_fit_kwargs)
        elif args['framework']=='sklearn':
            model, runtime = trn_sklearn_model(model=model, xtr=xtr, ytr=ytr,
                                               eval_set=None, fit_kwargs=model_fit_kwargs)
        elif args['framework']=='keras':
            model, runtime = trn_keras_model(model=model, xtr_sub=xtr, ytr_sub=ytr,
                                             eval_set=eval_set)
        elif args['framework']=='pytorch':
            pass
        else:
            raise ValueError(f'Framework {framework} is not yet supported.')
            
        if model is None:
            continue # sometimes keras fails to train a model (evaluates to nan)

        # Append runtime
        run_times.append(runtime)
        
        # Dump model
        if args['save_model']:
            joblib.dump(model, filename = rout / ('model.'+args['model_name']+'.pkl') )

        # Calc preds and scores
        # ... training set
        y_pred, y_true = calc_preds(model, x=xtr, y=ytr, mltype=mltype)
        tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=mtr, outpath=rout/'preds_tr.csv')
        # ... val set
        y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=mltype)
        vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=mvl, outpath=rout/'preds_vl.csv')
        # ... test set
        y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype=mltype)
        te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=mte, outpath=rout/'preds_te.csv')

        # Add metadata
        tr_scores['run'] = split_id
        vl_scores['run'] = split_id
        te_scores['run'] = split_id

        # Append scores (dicts)
        tr_scores_all.append(tr_scores)
        vl_scores_all.append(vl_scores)
        te_scores_all.append(te_scores)

        # Free space
        del xtr, ytr, mtr, xvl, yvl, mvl, xte, yte, mte, eval_set, model, estimator

        # if i%10 == 0:
        #     print(f'Finished {split_id}')

    file_smp_sz.close()

    # Scores to df
    tr_scores_df = scores_to_df( tr_scores_all )
    vl_scores_df = scores_to_df( vl_scores_all )
    te_scores_df = scores_to_df( te_scores_all )

    tr_scores_df.to_csv(gout/'tr_scores.csv', index=False)
    vl_scores_df.to_csv(gout/'vl_scores.csv', index=False)
    te_scores_df.to_csv(gout/'te_scores.csv', index=False)

    if (time()-t0)//3600 > 0:
        print_fn('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )

    del tr_scores_df, vl_scores_df, te_scores_df

    print_fn('Done.')
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])


