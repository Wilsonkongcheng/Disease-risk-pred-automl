import pandas as pd

""""

在此编写需要处理的字段和处理的函数
1. __init__下编写需要处理的字段，写法为self.[name] = [old_name,new_name]
2. 编写与字段相关联的处理函数，函数命名以及定义为name_m(self,x)
----------
note：
    1. __init__ 中的字段数量应与下方的转换函数数量一致，命名对齐（ex: self.sport -> def sport_m(self,x)）
    2. Class名称与该文件名称任意

"""


class MyProc:
    def __init__(self):
        # 需要更改的字段
        self.sport = ['运动状态', '运动']
        self.smoke = ['吸烟习惯', '吸烟']
        self.drink = ['饮酒习惯', '饮酒']
        self.gender = ['性别', '性别']
        self.year = ["年龄_年", "年龄等级"]


    # 字段应用的转换函数
    def sport_m(self, x):
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
        if x == '男':
            return 1
        else:
            return 0

    def year_m(self, x):
        if x < 18:
            return 0
        elif 18 <= x <= 45:
            return 0.5
        elif 45 < x <= 65:
            return 0.75
        else:
            return 1
