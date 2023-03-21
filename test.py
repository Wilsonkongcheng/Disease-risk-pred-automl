import sys
import argparse
from config import config_from_ini
import inspect

import inspect


class CatClass(object):
    name = "cat"
    age = 5
    def __init__(self, name="", age=0):
        self.name = name
        self.age = age

    # def __getattribute__(self, item):
    #     print('正在获取属性{}'.format(item))
    #     return super(CatClass, self).__getattribute__(item)

    def func1(self):
        a = 11111

    def func2(self):
        pass



def func1(a=1):
    a = 11111

#  成员函数
functions = inspect.getmembers(CatClass, lambda a: inspect.isfunction(a))
print(functions)
functions1 = list(filter(lambda x: not x[0].startswith('__'), functions))
print(functions1)

# 成员属性
attributes = inspect.getmembers(CatClass, lambda a: not inspect.isfunction(a))
print(attributes)
attributes1 = list(filter(lambda x: not x[0].startswith('__'), attributes)) # 过滤__开头的内部属性
print(attributes1)

cat = CatClass()
functions = inspect.getmembers(cat, lambda a: inspect.ismethod(a))
functions1 = list(filter(lambda x: not x[0].startswith('__'), functions))

print(inspect.getfullargspec(func1))
# print("*"*30)
# cat = CatClass("Tome", 11)
# print(cat.name)
# print(cat.age)








