import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
df = pd.read_csv('./data/full_data.csv')
print(df.info())
# random sample
df = pd.read_csv('./data/full_result.csv')
print(df.info())
print(pd.value_counts(df.is_lung_ca))

### 普筛
# # sample age>50
# age50 = df[df.age >= 50]
# print(age50.info())
# print(pd.value_counts(age50.is_cfm_diag))


#
test_pred = df.sort_values(by=('pred_proba_1'), ascending=False)  # sort

# test_pred = pd.read_csv('./data/result/test_data_results.csv')
for i in [10000, 50000, 100000, 150000, 200000]:
    count = pd.value_counts(test_pred[:i].is_lung_ca)[1]
    print(i, ":", count, "%.2f" % (count / 3091 * 100))

# count = pd.value_counts(test_pred[:162000].is_lung_ca)[1]
# print(count)


# # ROI result
# step = 5000
# count = []
# portion = []
# x = []
# for i in range(1, df.shape[0] // step + 2):
#     num_cfm_diag = pd.value_counts(test_pred[:step * i].is_cfm_diag)[1]
#     scale_cfm_diag = num_cfm_diag / 1594.0
#     count.append(num_cfm_diag)
#     portion.append(scale_cfm_diag)
#     x.append(step * i)


# # plot ROI curve
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# bar = ax1.bar(x, count, width=2500, label='count')
# ax2.plot(x, portion, 'r.--', label='propotion', linewidth=2)
# ax1.axhline(y=1594, color="black", linestyle='--', linewidth=2)
# ax1.axhline(y=1516, color="green", linestyle=':', linewidth=1)
# ax1.set_xlabel('sample number')
# ax1.set_ylabel('total_count', color='b')
# ax2.set_ylabel('hit_propotion', color='g')
# data = ax2.get_lines()[0].get_xydata()
# ax2.axhline(y=0.9, color="red", linestyle='--', linewidth=0.8, label='0.9')
# ax2.text(114000, 0.9, "(114000,0.9)", fontsize='x-small', bbox=dict(facecolor='red'))
# ax2.axhline(y=0.8, color="yellow", linestyle='--', linewidth=0.8, label='0.8')
# ax2.text(67500, 0.8, "(67500,0.8)",fontsize='x-small',bbox=dict(facecolor='yellow'))
# ax2.axhline(y=0.7, color="skyblue", linestyle='--', linewidth=0.8, label='0.7')
# ax2.text(41350, 0.7, "(41350,0.7)",fontsize='x-small',bbox=dict(facecolor='skyblue'))
# ax2.axhline(y=0.6, color="green", linestyle='--', linewidth=0.8, label='0.6')
# ax2.text(27500, 0.6, "(27500,0.6)",fontsize='x-small',bbox=dict(facecolor='green'))
# ax2.axhline(y=0.5, color="gray", linestyle='--', linewidth=0.8, label='0.5')
# ax2.text(17500, 0.5, "(17500,0.5)",fontsize='x-small',bbox=dict(facecolor='gray'))
#
# # print(np.interp(0.9, portion, x))
# # print(np.interp(0.8, portion, x))
# plt.title("模型性能表现")
# fig.legend()  # 添加全图图注
# plt.show()


# print(test_pred["is_last_apky"].value_counts(dropna=False))
# print(test_pred["is_last_xhdb"].value_counts(dropna=False))
# sample = test_pred[(np.isnan(test_pred.is_last_apky)) & (np.isnan(test_pred.is_last_xhdb))
#                    & (test_pred.is_cfm_diag == 1)]
# print(sample["is_last_apky"].unique())
# print(sample["is_last_xhdb"].unique())
# print(sample.info())
# print(sample.head())

