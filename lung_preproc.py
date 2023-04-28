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
        self.smk_qty = ['smk_qty', 'smk_qty']

        # 字段应用的转换函数

    def smk_qty_m(self, x):
        if pd.isna(x):
            return 0