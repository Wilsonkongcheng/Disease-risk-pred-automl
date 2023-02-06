import datetime
import math
from time import strftime

import numpy as np
import pandas as pd
import pymysql
import tqdm
from sqlalchemy import create_engine
from config import config_from_ini


class DB:

    def __init__(self, host, user, password, source, source_cols, target) -> None:
        '''
        数据库连接
        '''
        self.conn = pymysql.connect(
            host=host,
            user=user,
            password=password
        )
        self.source = source
        self.source_cols = source_cols
        self.target = target

    def fetch_data(self, limit=600000):
        # cols = ','.join(c for c in self.source_cols)
        sql = f'select {self.source_cols},{self.target} from {self.source} limit {limit} '
        all_records = pd.read_sql(sql, self.conn, chunksize=10000)
        all_dfs = []
        with tqdm.tqdm(total=60) as pbar:
            pbar.set_description('Processing:')
            for records in all_records:
                all_dfs.append(records)
                pbar.update(1)
        return pd.concat(all_dfs)

    def write_result(self, df, to_write_cols=['USER_ID', 'Pred']):
        cursor = self.conn.cursor()
        sql = f''' replace into dws.db_rsk_preds (USER_ID,SCR)
                            values (%s,%s)'''

        to_write = []
        for row in df[to_write_cols].values.tolist():
            to_write.append(tuple(row))
        cursor.executemany(sql, to_write)
        self.conn.commit()
        self.conn.close()


if __name__ == '__main__':
    cfg = config_from_ini(
        open('cfg.ini', 'rt', encoding='utf-8'), read_from_file=True)
    db = DB(cfg.db.host, cfg.db.user, cfg.db.password,
            cfg.source.table, cfg.source.cols, cfg.source.tgt)
    db.fetch_data()
