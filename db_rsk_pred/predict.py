import argparse
import sys

import numpy as np
import pandas as pd

import xgboost as xgb
# from params import paramsP
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
# import eli5
from db_rsk_pred.database.write_to_db import write_db
import os
# from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.preprocess.preprocess import PreProcessor
from db_rsk_pred.util.util import init_logger
import joblib
from db_rsk_pred.serve.load_model import *
from config import config_from_ini
LOGGER = init_logger()


def weight_filter(x):
    weight_col = x.filter(like="weight")
    top5_weight = weight_col.sort_values(ascending=False)[:5]  # 降序  top5
    top5_index = top5_weight.index
    # other = weight_col[~weight_col.index.isin(top5_index)]
    weight_col[~weight_col.index.isin(top5_index)] = np.nan  # fill others nan
    x[weight_col.index] = weight_col
    return x


def count_rsk(df: pd.DataFrame, cfg):
    rsk = [risk.strip() for risk in cfg.rsk_factor.factor.split(',') if risk != "None" and len(risk.strip()) > 0]
    return df[rsk].sum(axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='../data/full_data.csv')
    parser.add_argument("-c", "--cfg", default='../cfg_sample.ini')
    parser.add_argument("-m", "--model", default='model.json')
    args = parser.parse_args()
    cp = args.cfg
    cfg = config_from_ini(open(cp, 'rt', encoding='utf-8'), read_from_file=True)
    cols = cfg.source.cols

    cols = [c.strip() for c in cols.split(',') if len(c.strip()) > 0]

    #
    # tgt = cfg.source.tgt
    # is_cols = [col for col in cols if col[:2] == "is" or col=="tnb"]
    # dtype = {i: np.int64 for i in is_cols}
    ori_data = pd.read_csv(f'{args.data}')
    print(ori_data.info())
    processor = PreProcessor(cfg.preprocess.proc_func_path)

    data, col_mapping = processor.process(ori_data)
    cols = [col_mapping[c] for c in cols if c != cfg.source.id]  # drop user_id

    model = Model(args.model)

    # predict and shap
    # ori_data = ori_data[:1000]
    # data = data.iloc[:1000]
    preds_proba = model.predict(data)
    shap_value = model.explain(data[cols], preds_proba, 1)

    # process results
    preds_proba_1 = preds_proba[:, 1]
    df_shap = pd.DataFrame(shap_value, columns=[col + '_weight' for col in cols], dtype=np.float16)  # shap
    rsk_count = count_rsk(data, cfg)
    ori_data['pred_proba_1'] = preds_proba_1  # pred_proba of 1
    ori_data['rsk_count'] = rsk_count  # rsk_count
    ori_data = ori_data.reset_index(drop=True)  # reset index

    # merge all result
    new_df = [ori_data, df_shap]
    result_df = pd.concat(new_df, axis=1)

    # fillter shap value: feature value is not null
    for col in cols:
        result_df[col + "_weight"] = result_df.apply((lambda x: np.nan if pd.isna(x[col]) else x[col + "_weight"]),
                                                     axis=1)  # axis=1 apply function to each row
    # top5 weight
    result_df = result_df.apply(weight_filter, axis=1)

    # save to csv
    if not os.path.exists('../data'):
        os.mkdir('../data')
    path = '../data/full_result.csv'
    result_df.to_csv(path)
    LOGGER.info('sample of prediction results')
    print(data.head())
    LOGGER.info(f'{path.split("/")[-1]} saved to local disk')


    #  save to DB
    save_path = '../data/full_result.csv'
    write_db(cfg, save_path)




