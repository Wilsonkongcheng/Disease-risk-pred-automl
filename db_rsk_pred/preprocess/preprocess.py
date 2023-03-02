import pandas as pd


def sport(x):
    if pd.isna(x):
        return None
    if isinstance(x, int) and x == 0:
        return None
    if isinstance(x, str):
        if x in ['未知']:
            return None
        elif x in ['不运动', '不锻炼']:
            return 1
        else:
            return 0


def smoke(x):
    if pd.isna(x):
        return None
    if isinstance(x, int) and x == 0:
        return None
    if isinstance(x, str):
        if x == '未知':
            return None
        elif x == '从不吸烟':
            return 0
        elif x == '过去吸，已戒烟':
            return 0.5
        elif x == '吸烟':
            return 1


def drink(x):
    if pd.isna(x):
        return None
    if isinstance(x, int) and x == 0:
        return None
    if isinstance(x, str):
        if x == '未知':
            return None
        elif x == '从不':
            return 0
        else:
            return 1


def gender(x):
    if x == '男':
        return 1
    else:
        return 0


class PreProcessor:

    def __init__(self, proc_funcs={smoke: ['吸烟习惯', '吸烟'],
                 drink: ['饮酒习惯', '饮酒'], sport: ['运动状态', '运动'],
                 gender: ['性别', '性别']}) -> None:
        self.proc_funcs = proc_funcs
        # self.proc_col_onms =

    def process(self, df):
        col_mapping = {c: c for c in df.columns}
        for func, nms in self.proc_funcs.items():
            df[nms[1]] = df[nms[0]].apply(func)
            col_mapping[nms[0]] = nms[1]
        return df, col_mapping



if __name__ == '__main__':
    data = pd.read_csv('../../data/train_data.csv')
    print(data.columns)
    print(data.shape)
    processor = PreProcessor()
    data, colmapping = processor.process(data)
    print(data.shape)
    print(data.columns)
    print(colmapping)


