from db_rsk_pred.database.DB import DB
from config import config_from_ini
from db_rsk_pred.util.util import init_logger
import pandas as pd
from db_rsk_pred.util.util import logger


def write_db(cfg_path, path=None, result_df: pd.DataFrame = None):
    cfg = config_from_ini(
        open(cfg_path, 'rt', encoding='utf-8'), read_from_file=True)
    db = DB(cfg.db.host, cfg.db.port, cfg.db.user, cfg.db.password, cfg.db.db, cfg.target.table, cfg.source.cols,
            cfg.source.tgt, cfg.target)
    if result_df is None:
        result_df = pd.read_csv(path)
    # must replace nan into None before write to DB, and the first step is conveting  to object type
    result_df = result_df.astype(str).where(result_df.notna(), None)  # string类型可以插入None;int,float类型不可以插入None
    print(result_df.info())
    db.write_result(result_df)
    logger.info('Save to DB')


if __name__ == '__main__':
    logger = init_logger()
    cfg = config_from_ini(
        open('../../cfg_lung.ini', 'rt', encoding='utf-8'), read_from_file=True)
    # save to DB
    save_path = '../../data/full_result.csv'
    write_db(cfg, save_path)
