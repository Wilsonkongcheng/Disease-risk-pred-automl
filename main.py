from db_rsk_pred.train import train
from db_rsk_pred.predict import predict
from db_rsk_pred.util.util import init_logger
from db_rsk_pred.database.read_data_from_db import read_db, read_from_csv
import argparse
from db_rsk_pred.util.util import logger
import mlflow
import os
from db_rsk_pred.database.write_to_db import write_db

if __name__ == '__main__':

    # global param
    parser = argparse.ArgumentParser()
    parser.add_argument("-ml", "--use_mlflow", action='store_true')  # -ml == -ml true   if not use -ml  use_mlfow=False
    parser.add_argument("-c", "--cfg", default='./cfg_sample.ini')
    parser.add_argument("mode", type=str, choices=['train', 'pred'])
    # global_args = parser.parse_args()

    # train param
    parser.add_argument("-td", "--train_data", default='./data/train_data.csv')
    parser.add_argument('-s', '--source', choices=['csv', 'db'], default='db')
    parser.add_argument('-p', '--path')
    parser.add_argument('--save', default='./')
    # train_args = parser.parse_args()

    # pred param
    parser.add_argument("-pd", "--test_data", default='./data/full_data.csv')
    parser.add_argument("-M", "--model", default='./model.json')
    parser.add_argument("-e", "--explain", default=True, choices=['True', 'False'])
    parser.add_argument("-db", "--to_db", default=True, choices=['True', 'False'])
    args = parser.parse_args()

    # train or pred
    if args.mode == 'train':
        # fetch data
        if args.source == 'csv':
            _, _ = read_from_csv(args.cfg, args.path)
        else:
            _, _ = read_db(args.cfg)

        # mlflow
        if args.use_mlflow:
            mlflow.set_tracking_uri("http://10.123.234.229:5000")  # server addr
            mlflow.set_experiment("YuHuan_lung_rsk_pred_hjh")  # experiment name
            with mlflow.start_run():
                mydata = './data/train_data.csv'  #
                mlflow.log_artifact(mydata)  # data
                hyper_params, metrics, mymodel = train(args)
                mlflow.log_params(hyper_params)  # param
                # metrics
                mlflow.log_metric("top10000_hit_num", metrics["top10000_hit_num"])
                for i in range(len(metrics['auc'])):
                    mlflow.log_metrics({"binary_error": metrics["binary_error"][i],
                                        "binary_logloss": metrics["binary_logloss"][i],
                                        "auc": metrics["auc"][i]}, step=i)
                model_info = mlflow.lightgbm.log_model(mymodel,
                                                       artifact_path="model",
                                                       registered_model_name='hjh_yuhuan_lung_pred_lgb')  # model
                mlflow.end_run()
        else:
            train(args)

    else:
        result_df = predict(args)

        # save to csv
        if not os.path.exists('./data'):
            os.mkdir('./data')
        path = './data/full_result.csv'
        result_df.to_csv(path, index_label='idx')
        logger.info(f'{path.split("/")[-1]} saved to local disk')

        #  save to DB
        if eval(args.to_db):
            save_path = './data/full_result.csv'
            write_db(args.cfg, save_path)
            # logger.info(f'{path.split("/")[-1]}  saved to DB')

