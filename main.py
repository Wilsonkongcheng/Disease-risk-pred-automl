from db_rsk_pred.train import train
from db_rsk_pred.predict import predict
from config import config_from_ini
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
    parser.add_argument('-s', '--source', choices=['csv', 'db'], default='csv')
    parser.add_argument('-p', '--path', default='./data/full_data.csv')
    parser.add_argument('--save', default='./')
    # train_args = parser.parse_args()

    # pred param
    parser.add_argument("-pd", "--test_data", default='./data/full_data.csv')
    parser.add_argument("-M", "--model", default='./model.json')
    parser.add_argument("-e", "--explain", default='True', choices=['True', 'False'])
    parser.add_argument("-sp", "--save_path", default='./data/pred_result.csv')
    parser.add_argument("-db", "--to_db", default='True', choices=['True', 'False'])
    args = parser.parse_args()
    cfg = config_from_ini(
        open(args.cfg, 'rt', encoding='utf-8'), read_from_file=True)
    # train or pred
    if args.mode == 'train':
        # fetch data
        if args.source == 'csv':
            _, _ = read_from_csv(args.cfg, args.path)
        else:
            _, _ = read_db(args.cfg)

        # mlflow
        if args.use_mlflow:
            mlflow.set_tracking_uri("http://10.123.234.210:5000")  # server addr
            experiment_id = mlflow.create_experiment(name="YuHuan_lung_rsk_pred_hjh",
                                                     artifact_location='YuHuan_lung_rsk_pred_hjh',
                                                     tags={"version": "v1.0"})
            mlflow.set_experiment(experiment_name="YuHuan_lung_rsk_pred_hjh")  # experiment name id
            run_name = "update-all-origin_data-20230613"
            description = "Updata and add new samples in origin data on 2023-06-13"
            with mlflow.start_run(run_name=run_name, description=description):  # run_name description
                # mydata = './data/train_data.csv'  #
                # mlflow.log_artifact(mydata)  # data
                hyper_params, metrics, mymodel = train(args)
                mlflow.log_params(hyper_params)  # param
                # metrics
                mlflow.log_metrics({"top10000_hit_num": metrics["top10000_hit_num"],
                                    "eval_count_1": metrics["eval_count_1"],
                                    "eval_total": metrics["eval_total"]})
                for i in range(len(metrics['auc'])):
                    mlflow.log_metrics({"binary_error": metrics["binary_error"][i],
                                        "binary_logloss": metrics["binary_logloss"][i],
                                        "auc": metrics["auc"][i]}, step=i)
                model_info = mlflow.lightgbm.log_model(mymodel,
                                                       artifact_path="./model",
                                                       registered_model_name='hjh_yuhuan_lung_pred_lgb')  # model
                mlflow.end_run()
        else:
            train(args)

    else:
        result_df = predict(args, ori_data=None)
        print(result_df.info())
        if eval(cfg.result.save):
            # save to csv
            if not os.path.exists('./data'):
                os.mkdir('./data')
            result_df.to_csv(args.save_path, index_label='idx')
            logger.info(f'{args.save_path.split("/")[-1]} saved to local disk')

        #  save to DB
        if eval(args.to_db):
            save_path = args.save_path
            write_db(args.cfg, save_path)
            # logger.info(f'{path.split("/")[-1]}  saved to DB')
