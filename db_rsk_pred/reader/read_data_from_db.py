import argparse

from config import config_from_ini

from db_rsk_pred.reader.db import DB
from db_rsk_pred.util.util import init_logger

logger = init_logger()


def read_db(cfg,train =0.7,limit=1000):
    cfg = config_from_ini(
        open(cp, 'rt', encoding='utf-8'), read_from_file=True)
    db = DB(cfg.db.host, cfg.db.user, cfg.db.password,
            cfg.source.table, cfg.source.cols, cfg.source.tgt)
    data = db.fetch_data(limit=limit)
    data = data.sample(frac=1.0)
    train_size = int(data.shape[0]*train)
    train_data = data.iloc[0:train_size]
    test_data = data.iloc[train_size:]
    # test_data = data[~data.index.isin(train_data.index)]
    # print(test_data.shape)
    train_data.to_csv('./data/train_data.csv')
    test_data.to_csv('./data/test_data.csv')
    logger.info('total data size: %s, already save to local disk',
                data.shape[0])
    return train_size,test_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default='cfg.ini')
    args = parser.parse_args()
    cp = args.cfg
    cfg = config_from_ini(
        open(cp, 'rt', encoding='utf-8'), read_from_file=True)
    read_db(cfg)
