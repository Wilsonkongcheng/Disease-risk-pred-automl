import sys
import argparse
from functools import partial

import optuna
import pandas as pd
# import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from .model import param_setting
from db_rsk_pred.model import param_setting
from db_rsk_pred.database.DB import *
# from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.preprocess.preprocess import PreProcessor
from db_rsk_pred.util.util import init_logger
from db_rsk_pred.database.read_data_from_db import read_db
from db_rsk_pred.database.DB import *
# from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.preprocess.preprocess import PreProcessor
from db_rsk_pred.database.read_data_from_db import read_db
from db_rsk_pred.util.util import logger
import joblib
import os


def train(args):
    cp = args.cfg
    cfg = config_from_ini(
        open(cp, 'rt', encoding='utf-8'), read_from_file=True)

    cols = cfg.source.cols
    cols = [c.strip() for c in cols.split(',') if len(c.strip()) > 0]
    tgt = cfg.source.tgt
    processor = PreProcessor(cfg.preprocess.proc_func_path)
    # if args.source == 'csv':
    #     data = pd.read_csv(f'{args.train_data}')
    # else:
    #     data = read_db(cfg)
    data = pd.read_csv(f'{args.train_data}')
    data, col_mapping = processor.process(data)
    cols = [col_mapping[c] for c in cols if c != cfg.source.id]  # remove user_id then col_name mapping
    pos_constraints = [c.strip() for c in cfg.monotonic_constraint.pos.split(',') if c != "None" and len(c.strip()) > 0]
    neg_constraints = [c.strip() for c in cfg.monotonic_constraint.neg.split(',') if c != "None" and len(c.strip()) > 0]

    if not all([c for c in (pos_constraints + neg_constraints) if c not in cols]):
        raise ValueError('constraint features must be a subset of the features required to train the model')
    monotone_constraints = []

    # construct montone_constraints 0,1,-1
    for col in cols:
        if col in pos_constraints:
            monotone_constraints.append(1)
        elif col in neg_constraints:
            monotone_constraints.append(-1)
        else:
            monotone_constraints.append(0)

    train_data = data.sample(frac=0.8, random_state=0)
    eval_data = data[~data.index.isin(train_data.index)]
    # objective = param_setting.optuna_objective_new(train_data,
    #                     eval_data, cols, tgt, monotone_constraints,trial=optuna.trial._trial.Trial)
    objective = partial(param_setting.optuna_objective_new, train_data,
                        eval_data, cols, tgt, monotone_constraints)  # 偏函数 固定住trial以外的参数
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=-1,
                   show_progress_bar=True)  # n_job=-1 启动所有cpu线程；timeout 学习持续时间（s）到达该时间停止学习
    best_trial = study.best_trial
    best_params = best_trial.params
    print("best_params: ", best_params)
    # params = best_trial['params']
    logger.info('trail ends, now retrain lightgbm with the optimal hyper-parameters')

    # train
    other_params = {'class_weight': 'balanced',  # 针对不平衡数据集自动计算class_weight
                    "random_state": 0,
                    'monotone_constraints': monotone_constraints,
                    'monotone_constraints_method': "advanced"
                    }

    params = {**other_params, **best_params}
    best_model = LGBMClassifier(**params)
    train_log = best_model.fit(train_data[cols], train_data[tgt], eval_set=[(eval_data[cols], eval_data[tgt])],
                               eval_metric=["error", "auc"])

    # save model
    joblib.dump(best_model, os.path.join(args.save, 'model.json'))
    # best_model.booster.save_model(args.save)
    # best_model = best_trial.user_attrs['model']
    # best_model.save_model(os.path.join(args.save,'model.json'))
    logger.info('training finished ! model has been saved to %s', args.save)

    # eval
    y_pred = best_model.predict_proba(eval_data[cols])[:, 1]
    hit_num = eval_data.iloc[np.argsort(y_pred)[-10000:]][tgt].sum()
    eval_count_1 = pd.value_counts(eval_data[tgt])[1]
    eval_total = eval_data.shape[0]
    if args.use_mlflow:
        binary_error = train_log.evals_result_['valid_0']['binary_error']
        auc = train_log.evals_result_['valid_0']['auc']
        binary_logloss = train_log.evals_result_['valid_0']['binary_logloss']
        best_value = best_trial.value
        metrics = {"top10000_hit_num": hit_num, "binary_error": binary_error, "auc": auc,
                   "binary_logloss": binary_logloss,
                   "eval_count_1": eval_count_1, "eval_total": eval_total}
        return params, metrics, best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default='../cfg_lung.ini')
    parser.add_argument("-td", "--train_data", default='./data/train_data.csv')
    parser.add_argument('-s', '--source', default='csv')
    parser.add_argument('--save', default='./')

    args = parser.parse_args()
    train(args)
