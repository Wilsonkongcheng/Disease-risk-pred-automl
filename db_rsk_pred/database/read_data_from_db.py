import argparse
import os
import pandas as pd
from config import config_from_ini

from db_rsk_pred.database.DB import DB
from db_rsk_pred.util.util import logger


def read_from_csv(cfg, csv_path, train=0.7):
    cfg = config_from_ini(
        open(cfg, 'rt', encoding='utf-8'), read_from_file=True)
    columns = (cfg.source.cols + ',' + cfg.source.tgt).replace('\n', '').split(',')
    data = pd.read_csv(csv_path, usecols=columns)
    data = data.sample(frac=1.0, random_state=0)
    train_size = int(data.shape[0] * train)
    train_data = data.iloc[0:train_size]
    test_data = data.iloc[train_size:]
    # test_data = data[~data.index.isin(train_data.index)]
    # print(test_data.shape)
    if not os.path.exists('./data'):
        os.mkdir('./data')
    train_data.to_csv('./data/train_data.csv', index=False)  # ignore index
    test_data.to_csv('./data/test_data.csv', index=False)
    logger.info('total train_data size: %s, already save to local disk',
                train_data.shape[0])
    return train_size, test_data


def read_db(cfg, train=0.7, limit=1500000):
    cfg = config_from_ini(
        open(cfg, 'rt', encoding='utf-8'), read_from_file=True)
    db = DB(cfg.db.host, cfg.db.port, cfg.db.user, cfg.db.password, cfg.db.db,
            cfg.source.table, cfg.source.cols, cfg.source.tgt)
    data = db.fetch_data_new(limit=limit)
    print(os.getcwd())
    if not os.path.exists('./data'):
        os.mkdir('./data')
    # save origin data to disk
    data.to_csv('./data/full_data.csv', index=False)  # ignore index
    logger.info('total data size: %s, already save to local disk',
                data.shape[0])
    data = data.sample(frac=1.0)
    train_size = int(data.shape[0] * train)
    train_data = data.iloc[0:train_size]
    test_data = data.iloc[train_size:]
    # test_data = data[~data.index.isin(train_data.index)]
    # print(test_data.shape)
    if not os.path.exists('./data'):
        os.mkdir('./data')
    train_data.to_csv('./data/train_data.csv', index=False)  # ignore index
    test_data.to_csv('./data/test_data.csv', index=False)
    logger.info('train data size: %s, test data size: %s  already save to local disk',
                train_data.shape[0], test_data.shape[0])
    return train_size, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default='../../cfg_lung.ini')
    args = parser.parse_args()
    cfg = args.cfg
    train_size, _ = read_from_csv(cfg,'')
    # train_size, _ = read_db(cfg)
