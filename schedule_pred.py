from schedule import every, repeat, run_pending
from datetime import datetime, timedelta, time
from db_rsk_pred.database.DB import DB
from config import config_from_ini
from db_rsk_pred.predict import predict
from db_rsk_pred.database.write_to_db import write_db
import os
import argparse
import time


@repeat(every().day)  # 每隔24h
def job():
    # global param
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default='./cfg_lung.ini')
    # global_args = parser.parse_args()

    # pred param
    parser.add_argument("-pd", "--test_data", default='None')
    parser.add_argument("-M", "--model", default='./model.json')
    parser.add_argument("-e", "--explain", default='False', choices=['True', 'False'])
    parser.add_argument("-db", "--to_db", default='True', choices=['True', 'False'])
    args = parser.parse_args()

    # fetch new data from DB
    cfg = config_from_ini(
        open(args.cfg, 'rt', encoding='utf-8'), read_from_file=True)
    db = DB(cfg.db.host, cfg.db.port, cfg.db.user, cfg.db.password, cfg.db.db,
            cfg.source.table, cfg.source.cols, cfg.source.tgt)
    sql = '''select a.id_crd_no,age,is_smk,smk_qty,is_quit_smk,quit_smk_age,quit_smk_drt,is_exzl,is_yjqsqzfa,
        is_mxfbjbs,is_dqwrbl,is_ACEI,is_lung_ca 
    from dws.aggr_肺癌_rsk_idx_rlt_elmt as a left join  acvm_lab.pub_rsk_lung_ca_test  as b
    on a.id_crd_no=b.id_crd_no
    where b.pred_proba_1 is null'''

    data = db.fetch_data_new(sql_str=sql)

    # judge empty dataframe
    if data.empty:
        print("There's no new update datas")
    else:
        # save to db like 20230515_113718.csv
        now = datetime.now()
        save_path = f"./data/{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}.csv"
        data.to_csv(save_path, index=False)

        # predict
        args.test_data = save_path
        result_df = predict(args)
        print(result_df.info())

        # save to csv
        if not os.path.exists('./data'):
            os.mkdir('./data')
        path = f'./data/{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}_result.csv'
        result_df.to_csv(path, index_label='idx')

        #  save to DB
        if eval(args.to_db):
            save_path = path
            write_db(cfg, save_path)


if __name__ == '__main__':
    while True:
        run_pending()
        time.sleep(1)
