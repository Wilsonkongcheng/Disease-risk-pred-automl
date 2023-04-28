from db_rsk_pred.train import train
from db_rsk_pred.predict import predict
from db_rsk_pred.util.util import init_logger
from db_rsk_pred.database.read_data_from_db import read_db, read_from_csv
import argparse
from db_rsk_pred.util.util import logger
import mlflow

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
    parser.add_argument('--save', default='./')
    # train_args = parser.parse_args()

    # pred param
    parser.add_argument("-pd", "--test_data", default='./data/full_data.csv')
    parser.add_argument("-M", "--model", default='./model.json')
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
            mlflow.set_experiment("yuhuan_lung_rsk_pred")  # experiment name
            with mlflow.start_run():
                mydata = './data/train_data.csv'  #
                mlflow.log_artifact(mydata)  # data
                hyper_params, metrics, mymodel = train(args)
                mlflow.log_params(hyper_params)  # param
                mlflow.log_metrics(metrics)  # metric
                model_info = mlflow.lightgbm.log_model(mymodel,
                                                       artifact_path="model",
                                                       registered_model_name='hjh_yuhuan_lung_pred_lgb')  # model
                mlflow.end_run()
        else:
            train(args)

    else:
        predict(args)
