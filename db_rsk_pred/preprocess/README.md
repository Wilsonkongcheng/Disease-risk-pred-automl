## README

### preprocess模块使用方法
#### 编写用户自定义预处理函数（user_proc.py），以db_rsk_pred/preprocess/Proc_demo.py为例，只需要做三步修改
##### 1.__init__下为字段映射 格式为 self.[name] = ["old_colunm_name","new_colum_name"]
##### 2.为对应的字段添加映射方法，该方法必须以[name]_m(self,x)命名，不得与上述变量重名；函数体根据用户自定义规则编写（参考Proc_demo.py）
##### 3.将该文件的绝对路径引用到cfg_sample.ini配置文件的preprocess下（参考cfg_sample.ini中的proc_func_path）