import datetime
import math
from time import strftime

import numpy as np
import pandas as pd
import pymysql
import tqdm
from sqlalchemy import create_engine
from config import config_from_ini
import time
import psutil
import os

class DB:

    def __init__(self, host, user, password, source, source_cols, target) -> None:
        '''
        数据库连接
        '''
        self.conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            cursorclass=pymysql.cursors.SSCursor
        )
        self.source = source  # table
        self.source_cols = source_cols  # cols
        self.target = target # tgt

    def fetch_data(self, limit=3000000):
        # cols = ','.join(c for c in self.source_cols)
        chunksize = int(limit/100)
        sql = f'select {self.source_cols},{self.target} from {self.source} limit {limit} '
        all_records = pd.read_sql(sql, self.conn, chunksize=chunksize)
        all_dfs = []
        with tqdm.tqdm(total=100) as pbar:  # 60
            pbar.set_description('Processing:')
            for records in all_records:
                all_dfs.append(records)
                pbar.update(1)
        return pd.concat(all_dfs)

    def fetch_data_new(self, limit=3000000):  # 降低3-5%左右的内存使用，时间与fetch_data相近
        sql = f'select {self.source_cols},{self.target} from {self.source} limit {limit} '
        columns = (self.source_cols+','+self.target).replace('\n', '').split(',')
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall_unbuffered()
            all_dfs = []
            with tqdm.tqdm(total=limit) as pbar:  # 60
                pbar.set_description('Processing')
                for record in result:
                    all_dfs.append(record)
                    pbar.update(1)
            return pd.DataFrame(data=all_dfs, columns=columns)
            # return pd.DataFrame(data=result, columns=columns)




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
        open('../../cfg_sample.ini', 'rt', encoding='utf-8'), read_from_file=True)
    db = DB(cfg.db.host, cfg.db.user, cfg.db.password,
            cfg.source.table, cfg.source.cols, cfg.source.tgt)
    start = time.time()
    data = db.fetch_data()
    end = time.time()
    print("total time:", end-start)
    print(data.info())
    print(data.head())
    # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    # db.fetch_data_new()
