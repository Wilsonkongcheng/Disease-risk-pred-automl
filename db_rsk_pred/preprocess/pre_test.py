import pandas as pd
import inspect


class Proc:
    def __init__(self):
        self.sport = ['运动状态', '运动']
        self.smoke = ['吸烟习惯', '吸烟']
        self.drink = ['饮酒习惯', '饮酒']
        self.gender = ['性别', '性别']

    def sport_m(self, x):
        # self.sport = ['运动状态', '运动']
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

    def smoke_m(self, x):
        # self.smoke = ['吸烟习惯', '吸烟']
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

    def drink_m(self, x):
        # self.drink = ['饮酒习惯', '饮酒']
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

    def gender_m(self, x):
        # self.gender = ['性别', '性别']
        if x == '男':
            return 1
        else:
            return 0

    @staticmethod
    def create_map_dict(proc_class=None):   # 静态方法
        if not proc_class:
            proc = Proc()
        else:
            proc = proc_class
        # 对象函数
        functions = inspect.getmembers(proc, lambda a: inspect.ismethod(a))
        # print(functions)
        functions = list(filter(lambda x: not x[0].startswith('__'), functions))  # 过滤掉__开头的  bound methond
        # print(functions)
        # # print(functions_tmp)
        functions = list(filter(lambda x: not x[0] == "create_map_dict", functions))
        print(functions)
        #
        # #对象属性
        attributes = inspect.getmembers(proc, lambda a: not (inspect.ismethod(a) or inspect.isfunction(a))) # 过滤出attributes
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



# class MyProc(Proc):
#     def __init__(self):
#         self.test_col = ['a','b']
#     def testfunc_m(self, x):
#         # self.test_col = ['a', 'b']
#         pass
#     @staticmethod
#     def create_map_dict():
#         Proc.create_map_dict(MyProc())



class PreProcessor:

    def __init__(self, proc_funcs=Proc) -> None:
        self.proc_funcs = proc_funcs.create_map_dict()  # 静态方法调用无需创建object

    def process(self, df):
        col_mapping = {c: c for c in df.columns}
        for func, nms in self.proc_funcs.items():
            df[nms[1]] = df[nms[0]].apply(func)
            col_mapping[nms[0]] = nms[1]
        return df, col_mapping


if __name__ == '__main__':
    # proc = MyProc()
    # proc.create_map_dict()


    data = pd.read_csv('../../data/train_data.csv')
    print(data.columns)
    print(data.shape)
    processor = PreProcessor()
    data, colmapping = processor.process(data)
    print(data.shape)
    print(data.columns)
    print(colmapping)
    print(data["吸烟"].unique())
