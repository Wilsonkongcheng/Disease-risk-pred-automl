import argparse
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import xgboost as xgb
# from params import paramsP
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
# import eli5
from db_rsk_pred.database.DB import *
from db_rsk_pred.database.DB import DB
from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.util.util import init_logger
import joblib
import shap



class Model:

    def __init__(self, model_path) -> None:
        # self.model = joblib.load('lgb.pkl')
        try:
            self.model = joblib.load(model_path)
        except:
            raise FileNotFoundError(''' the model file doesn't exist''')
        # self.feature_names = self.model.booster_.feature_name
        self.feature_names = self.model.booster_.feature_name()
        print(self.feature_names)

    def predict(self, data):
        if isinstance(data, list) and isinstance(data[0], dict):
            features = list(data[0].keys())
            if not all([True for f in self.feature_names if f in features]):
                raise ValueError('some features to run the model are missing in the data')
            else:
                data = pd.DataFrame(data)[self.feature_names]
        elif isinstance(data, pd.DataFrame):
            try:
                data = data[self.feature_names]
            except:
                raise ValueError('some features to run the model are missing in the data')
        else:
            raise TypeError('data type has to be list of dict or pandas dataframe')
        preds = self.model.predict_proba(data)
        return preds

    def explain(self, data, preds, label_index: int):
        explainer = shap.TreeExplainer(self.model)
        shap_values = np.array(explainer.shap_values(data))  # [label,N,feature]
        # explan = explainer.shap_values(data, approximate=True)  # shap value

        preds_proba = preds[:, label_index]
        expected_value = explainer.expected_value[label_index]
        shap_value = shap_values[label_index] - expected_value
        shap_value_output = shap_value + preds_proba.reshape(-1, 1)

        return shap_value_output


def weight_filter(x):
    weight_col = x.filter(like="weight")
    top5_weight = weight_col.sort_values(ascending=False)[:5]  # 降序
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
    parser.add_argument("-d", "--data", default="../../data/process/test_data.csv")
    parser.add_argument("-c", "--cfg", default='../../cfg_lung.ini')
    parser.add_argument("-m", "--model", default="F:\\PycharmProject\\dzs_rsk_pred_automl\\db_rsk_pred\\model.json")
    args = parser.parse_args()
    cp = args.cfg
    cfg = config_from_ini(open(cp, 'rt', encoding='utf-8'), read_from_file=True)
    cols = cfg.source.cols

    cols = [c.strip() for c in cols.split(',') if len(c.strip()) > 0]
    #
    tgt = cfg.source.tgt

    # data = pd.read_csv(f'./data/{args.data}')
    ori_data = pd.read_csv(args.data)

    # preprocess
    processor = PreProcessor(cfg.preprocess.proc_func_path)
    data, col_mapping = processor.process(ori_data)
    cols = [col_mapping[c] for c in cols if c != cfg.source.id]

    model = Model(args.model)

    y_pred_prob = model.predict(data[cols])[:, 1]
    ori_data["pred_proba_1"] = y_pred_prob
    sorted_test_results = ori_data.sort_values(by=('pred_proba_1'), ascending=False)

    # fetch some datas(top 1000;last 1000;mid 1000)
    # index_1_midk = np.argsort(y_pred_prob)[int(len(y_pred_prob)/2)-500:int(len(y_pred_prob)/2)+500]  # mid k index
    index_1_topk = np.argsort(y_pred_prob)[-5:]  # mid k index
    # print(index_1_topk.shape)
    # print(data.index)
    # print(data.info())
    sample = data[(~np.isnan(data.is_last_apky)) & (~np.isnan(data.is_last_xhdb))  # sample with  xhdb and apky and diag
                  & (data.is_cfm_diag == 0) & (data.age <= 70)].sort_values(by=('pred_proba_1'), ascending=False)
    topk_data = sample.iloc[:5]
    # topk_data = data.iloc[index_1_topk]
    for index, row in topk_data.iterrows():
        print(row)
    # calculate pred_prob of 1, shap and rsk_count of topk_data
    preds_proba, shap_values = model.predict(topk_data)
    # preds_proba_1 = preds_proba[:, 1]
    # df_shap = pd.DataFrame(shap_values, columns=[col + '_weight' for col in cols], dtype=np.float16)  # shap
    # rsk_count = count_rsk(data, cfg)
    # data['pred_proba_1'] = preds_proba_1  # pred_proba of 1
    # data['rsk_count'] = rsk_count  # rsk_count
    # data = data.reset_index(drop=True)  # reset index
    #
    #
    # # merge all result
    # new_df = [topk_data, df_shap]
    # result_df = pd.concat(new_df, axis=1)
    # result_df.info()
    #
    # # change label and feature of gdr_num
    # new_cols = list(result_df.columns)
    # i, j = new_cols.index("is_cfm_diag"), new_cols.index("gdr_num")
    # new_cols[i], new_cols[j] = new_cols[j], new_cols[i]
    # result_df = result_df.loc[:, new_cols]
    #
    # # fillter shap value: feature value is not null
    # for col in cols:
    #     result_df[col + "_weight"] = result_df.apply((lambda x:  np.nan if pd.isna(x[col]) else x[col + "_weight"]), axis=1) # axis=1 apply function to each row
    #
    # result_df = result_df.apply(weight_filter, axis=1)
    #
    # # save to csv
    # sorted_test_results.to_csv('../../data/result/test_data_results.csv', index=False)
    # LOGGER.info('result.csv is saved to local disk')


