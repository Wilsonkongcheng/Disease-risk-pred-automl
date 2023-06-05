from db_rsk_pred.serve.load_model import *
from db_rsk_pred.serve.load_model import Model
import shap
import numpy as np


def shap_vis(model, data=None):
    # explainer = shap.TreeExplainer(model, data=data, model_output='predict')
    # explan = explainer.shap_values(data, approximate=True)  # shap value
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(data))  # [2,399997,26]

    # global
    plt.title("feature importance")
    shap.plots.bar(explainer(data)[:, :, 1])
    shap.summary_plot(shap_values[1], data.astype("float"), plot_type='bar', max_display=26)  # feature importance
    shap.summary_plot(shap_values[1], data.astype("float"), )  # feature impact on shap value

    # sample vis
    for i in range(5):
        shap.force_plot(explainer.expected_value[1], shap_values[1][i, :], data.iloc[i, :], matplotlib=True,
                        show=True)
        plt.title(f"sample_{i}")
        plt.savefig(f"sample_{i}.jpg", bbox_inches='tight')
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="../../data/process/test_data.csv")
    parser.add_argument("-c", "--cfg", default='../../cfg_lung.ini')
    parser.add_argument("-m", "--model", default="F:\\PycharmProject\\dzs_rsk_pred_automl\\db_rsk_pred\\model.json")
    args = parser.parse_args()
    cp = args.cfg
    cfg = config_from_ini(open(cp, 'rt', encoding='utf-8'), read_from_file=True)
    cols = cfg.source.cols

    cols = [c.strip() for c in cols.split(',') if len(c.strip()) > 0]
    #
    tgt = cfg.source.tgt

    # data = pd.read_csv(f'./data/{args.data}')
    ori_data = pd.read_csv(args.data)

    # preprocess
    processor = PreProcessor(cfg.preprocess.proc_func_path)
    data, col_mapping = processor.process(ori_data)
    cols = [col_mapping[c] for c in cols if c != cfg.source.id]

    model = Model(args.model)
    shap_vis(model.model, data[cols])
