import sys
import importlib
import pandas as pd
import inspect
import argparse


# 根据自定义py动态导入模块
def import_module_from_file(module_file_path):
    # sys.path.append(module_file_path)
    module_spec = importlib.util.spec_from_file_location("Proc", module_file_path)  # 定义命名空间名称Proc  导入命名空间（py）
    module = importlib.util.module_from_spec(module_spec)  # 从命名空间导入模块
    module_spec.loader.exec_module(module)
    sys.modules["Proc"] = module  # 将模块加入到当前解释器中以便后续以from Proc import * 导入
    return module

# 根据用户自定义的Proc.py导入Proc Class
def fetch_class_from_module(module):
    # print(inspect.getmembers(module, inspect.isclass))
    (cls_name, cls) = inspect.getmembers(module, inspect.isclass)[0]  # 获取模块下的类名，类
    return cls_name, cls


def create_map_dict(proc_obj):
    # 对象函数
    functions = inspect.getmembers(proc_obj, lambda a: inspect.ismethod(a))  # 获取所有对象的bound methond
    functions = list(filter(lambda x: not x[0].startswith('__'), functions))  # 过滤掉__开头的bound methond
    # print(functions)
    #
    # #对象属性
    attributes = inspect.getmembers(proc_obj,
                                    lambda a: not (inspect.ismethod(a) or inspect.isfunction(a)))  # 过滤出attributes
    attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))  # 过滤__开头的内部属性
    # print(attributes)

    col_map = {k: v for (k, v) in
               attributes}  # list to dict   # {'drink': ['饮酒习惯', '饮酒'], 'gender': ['性别', '性别'], 'smoke': ['吸烟习惯', '吸烟']}

    proc_lib = {}
    for func in functions:
        proc_lib[func[1]] = col_map[str(func[0])[:-2]]

    # proc_lib = {self.sport: col_map["sport"], self.smoke: col_map["smoke"], self.drink: col_map["drink"], self.gender: col_map["gender"]}
    return proc_lib




class PreProcessor:

    def __init__(self, module_file_path:str) -> None:
        module = import_module_from_file(module_file_path)
        cls_name, cls = fetch_class_from_module(module)
        user_proc = cls()
        self.proc_funcs = create_map_dict(user_proc)

    def process(self, df):
        col_mapping = {c: c for c in df.columns}
        for func, nms in self.proc_funcs.items():
            if nms[1] in ['sport_flag', 'eat_flag']:
                df[nms[1]] = df[nms[0]].apply(func).astype('category')  # value map then to category
            else:
                df[nms[1]] = df[nms[0]].apply(func).apply(pd.to_numeric)  # value map then to float
            col_mapping[nms[0]] = nms[1]
        return df, col_mapping   # old+new cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--process", default=r'C:\Users\63439\Desktop\Proc.py')
    args = parser.parse_args()  # path of user's proc.py
    module_py_path = args.process
    # module = import_module_from_file(module_py_path)

    # cls_name, cls = fetch_class_from_module(module)

    # proc = cls()
    # print(type(proc))
    data = pd.read_csv('../../data/train_data.csv')
    print(data.columns)
    print(data.shape)
    # print(data["年龄_年"].unique())

    processor = PreProcessor(module_py_path)
    print(sys.modules["Proc"])  # <module 'Proc' from 'C:\\Users\\63439\\Desktop\\Proc.py'>
    data, colmapping = processor.process(data)
    print(data.shape)
    print(data.columns)
    print(colmapping)
    print(data["年龄等级"].unique())
    print(data["年龄_年"].max(), data["年龄_年"].min())
