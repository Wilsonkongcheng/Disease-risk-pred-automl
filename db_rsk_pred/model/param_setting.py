import sys
import os

  # 当前项目路径加入到sys.path头部（必须加入头部，append加入尾部依旧会报错）                                                      # os.getcwd() 当前工作路径 working dir
import argparse                                                          # 必须使sys.path头部插入 'F:\\PycharmProject\\dzs_rsk_pred_automl'后续才可import db_rsk_pred中的包
from functools import partial

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from db_rsk_pred.reader.db import *
# from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.util.util import init_logger
from db_rsk_pred.reader.read_data_from_db import read_db
from db_rsk_pred.preprocess.pre_test import PreProcessor


def optuna_objective(train_data, test_data, features, label, trial):
    """Define the objective function"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 5, 20),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'eval_metric': 'mlogloss',
        'class_weight': 'balanced',
        'use_label_encoder': False,
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'monotone_constraints': [1] * len(features)
    }

    # Fit the model
    optuna_model = LGBMClassifier(
        **params)
    optuna_model.fit(train_data[features], train_data[label])

    # Make predictions
    y_pred = optuna_model.predict_proba(test_data[features])[:, 1]
    return test_data.iloc[np.argsort(y_pred)[-10000:]][label].sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='../../data/train_data.csv')
    parser.add_argument("-c", "--cfg", default='../../cfg_sample.ini')
    parser.add_argument('-s','--source',default='csv')
    parser.add_argument('--save')

    args = parser.parse_args()
    cp = args.cfg
    cfg = config_from_ini(
        open(cp, 'rt', encoding='utf-8'), read_from_file=True)

    cols = cfg.source.cols
    cols = [c.strip()for c in cols.split(',') if len(c.strip()) > 0]
    tgt = cfg.source.tgt

    # pos_constraints = cfg.monotonic_constraint.pos
    # if not all([c for c in pos_constraints if c not in cols]):
    #     raise ValueError('constraint features must be a subset of the features required to train the model')
    # pos_constraints = [c.strip()for c in cols.split(',') if len(c.strip()) > 0]
    # monotone_constraints = [1 for _ in pos_constraints]
    # print(monotone_constraints)

    sys.path.append(cfg.preprocess.proc_func_path)

    from Proc import Proc

    if args.source =='csv':
        data = pd.read_csv(f'{args.data}')
    else:
        data = read_db(cfg)

    # cols = [col_mapping[c] for c in cols if c != cfg.source.id]


    processor = PreProcessor(Proc())
    # print(type(processor))
    # print(dir(processor)[-2:])
    data, col_mapping = processor.process(data)
    print(data["吸烟"].unique())
    print(col_mapping)
    # train_data = data.sample(frac=0.8)
    # eval_data = data[~data.index.isin(train_data.index)]
    # objective = partial(optuna_objective, train_data[cols],
    #                     eval_data[cols], monotone_constraints, cols, tgt)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100,timeout=2,n_jobs=-1,show_progress_bar =True)
    # best_trial = study.best_trial
    # best_model = best_trial.user_attrs['model']
    # best_model.save(args.save)
    # logger.info('training finished ! model has been saved to %s',args.save)
