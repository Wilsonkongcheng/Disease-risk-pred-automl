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
import os


class DB:

    def __init__(self, host, port, user, password, db, source, source_cols, target, write=None) -> None:
        '''
        数据库连接
        '''
        self.conn = pymysql.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=db,
            cursorclass=pymysql.cursors.SSCursor
        )
        self.source = source  # table
        self.source_cols = source_cols  # cols
        self.target = target  # tgt
        self.write = write

    def fetch_data(self, limit=3000000):
        # cols = ','.join(c for c in self.source_cols)
        chunksize = int(limit / 100)
        sql = f'select {self.source_cols},{self.target} from {self.source} limit {limit} '
        all_records = pd.read_sql(sql, self.conn, chunksize=chunksize)
        all_dfs = []
        with tqdm.tqdm(total=100) as pbar:  # 60
            pbar.set_description('Processing:')
            for records in all_records:
                all_dfs.append(records)
                pbar.update(1)
        return pd.concat(all_dfs)

    def fetch_data_new(self, sql_str: str = None, limit=1000000):  # 降低3-5%左右的内存使用，时间与fetch_data相近
        if not sql_str:
            sql = f'select {self.source_cols},{self.target} from {self.source} limit {limit} '
        else:
            sql = sql_str
        columns = (self.source_cols + ',' + self.target).replace('\n', '').split(',')
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

    def write_result(self, df):
        # columns str to list
        to_write_pd_cols = self.write.pd_cols.replace('\n', '').split(',')
        # create sql
        values_str = ('%s,' * len(to_write_pd_cols))[:-1]
        # to_write_sql_cols_str = str(to_write_sql_cols)[1:-1].replace("'", '')
        sql = f''' REPLACE INTO {self.write.table}''' \
              + f''' ({self.write.sql_cols})''' + f''' VALUES ({values_str})'''

        # Create a cursor and begin a transaction
        cursor = self.conn.cursor()
        cursor.execute('BEGIN')
        # Split the data into batches of 5000 rows
        batch_size = 5000
        num_batches = df.shape[0] // batch_size + (1 if df.shape[0] % batch_size != 0 else 0)
        # Use tqdm to add a progress bar to the insertion process
        for i in tqdm.tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, df.shape[0])
            batch = df.iloc[start_idx:end_idx]
            to_write = []
            for row in batch[to_write_pd_cols].values.tolist():
                to_write.append(tuple(row))
            cursor.executemany(sql, to_write)

            # commit the changes
            self.conn.commit()
        # close the connection
        self.conn.close()


if __name__ == '__main__':
    cfg = config_from_ini(
        open('../../cfg_sample.ini', 'rt', encoding='utf-8'), read_from_file=True)
    # db = DB(cfg.db.host, cfg.db.port, cfg.db.user, cfg.db.password, cfg.db.db,
    #         cfg.source.table, cfg.source.cols, cfg.source.tgt)
    # start = time.time()
    # data = db.fetch_data_new()
    # end = time.time()
    # print("total time:", end - start)
    # print(data.info())
    # print(data.head())
    # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    # db.fetch_data_new()

    # save to DB
    db = DB(cfg.db.host, cfg.db.port, cfg.db.user, cfg.db.password, cfg.db.db, cfg.target.table, cfg.source.cols,
            cfg.source.tgt, cfg.target)
    db.save_to_db('../../data/full_result.csv')
