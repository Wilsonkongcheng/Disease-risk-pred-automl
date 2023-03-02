import sys

import pandas as pd
import inspect
import argparse


def create_map_dict(proc_obj):

    # 对象函数
    functions = inspect.getmembers(proc_obj, lambda a: inspect.ismethod(a))
    # print(functions)
    functions = list(filter(lambda x: not x[0].startswith('__'), functions))  # 过滤掉__开头的  bound methond
    # print(functions)
    # # print(functions_tmp)
    functions = list(filter(lambda x: not x[0] == "create_map_dict", functions))
    print(functions)
    #
    # #对象属性
    attributes = inspect.getmembers(proc_obj, lambda a: not (inspect.ismethod(a) or inspect.isfunction(a))) # 过滤出attributes
    print(attributes)
    attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))  # 过滤__开头的内部属性
    print(attributes)


    col_map = {k:v for (k,v) in attributes}
    print(col_map)

    # col_map = {"smoke": ['吸烟习惯', '吸烟'], "drink": ['饮酒习惯', '饮酒'], "sport": ['运动状态', '运动'],
    #        "gender": ['性别', '性别']}
    proc_lib = {}
    for func in functions:
        proc_lib[func[1]] = col_map[str(func[0])[:-2]]

    # proc_lib = {self.sport: col_map["sport"], self.smoke: col_map["smoke"], self.drink: col_map["drink"], self.gender: col_map["gender"]}
    print(proc_lib)
    return proc_lib


class PreProcessor:

    def __init__(self, my_proc) -> None:
        self.proc_funcs = create_map_dict(my_proc)

    def process(self, df):
        col_mapping = {c: c for c in df.columns}
        for func, nms in self.proc_funcs.items():
            df[nms[1]] = df[nms[0]].apply(func).apply(pd.to_numeric)   # value map then to float
            col_mapping[nms[0]] = nms[1]
        return df, col_mapping


if __name__ == '__main__':
    # proc = MyProc()
    # proc.create_map_dict()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--process", default='./')
    args = parser.parse_args()
    sys.path.append(args.process)
    from Proc import Proc
    proc = Proc()
    data = pd.read_csv('../../data/train_data.csv')
    print(data.columns)
    print(data.shape)
    processor = PreProcessor(proc)
    data, colmapping = processor.process(data)
    print(data.shape)
    print(data.columns)
    print(colmapping)
    print(data["吸烟"].unique())
