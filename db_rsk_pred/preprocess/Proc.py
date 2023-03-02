import pandas as pd


class Proc:
    def __init__(self):
        # 需要更改的字段
        self.sport = ['运动状态', '运动']
        self.smoke = ['吸烟习惯', '吸烟']
        self.drink = ['饮酒习惯', '饮酒']
        self.gender = ['性别', '性别']


    # 字段应用的转换函数
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
