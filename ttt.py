# 假设txt文件内容如下：
# [1, 2, 3, "apple", "banana"]

# 打开并读取txt文件内容
with open('/Users/shifengfan/desktop/expdata/nmi_cora.txt', 'r', encoding='utf-8') as f:
    content = f.read().strip()

# 将字符串内容转换为Python列表
import ast
loaded_list = ast.literal_eval(content)

# 现在loaded_list就是一个真正的Python列表了
print(loaded_list)


# 折线图模版

import matplotlib.pyplot as plt
import seaborn as sns

xlabel = "epochs"
ylabel = "NMI t-1 / t"

sns.set_theme(style="whitegrid", rc={"axes.linewidth": 1, "axes.edgecolor": "black"})
plt.figure(figsize=(8, 6))
# 或者全局设置字体为Arial
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12

# 数据
# datasets = ['Cora', 'Citeseer', 'Citation', 'ACM', 'Pubmed']
sampling_rounds = range(0, 100)

# 绘制图形
plt.plot(sampling_rounds, loaded_list, lw=4.0)

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# sns.despine()
# 改变X轴刻度标签的字体大小
plt.xticks(fontsize=18)
# 如果需要同时改变Y轴刻度标签
plt.yticks(fontsize=18)
plt.xlabel(xlabel, fontsize=20)
plt.ylabel(ylabel, fontsize=20)
# plt.legend(loc="lower right")
plt.savefig('./test.png', dpi=300, bbox_inches='tight')
plt.show()


# 折线图模版

import matplotlib.pyplot as plt
import seaborn as sns

xlabel = "Ratio"
ylabel = "AUC"

sns.set_theme(style="whitegrid", rc={"axes.linewidth": 1, "axes.edgecolor": "black"})
plt.figure(figsize=(8, 8))
# 或者全局设置字体为Arial
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.size'] = 12

# 数据
datasets = ['Cora', 'Citeseer', 'Citation', 'ACM', 'Pubmed']
# sampling_rounds = range(0, 100)
sampling_rounds = [0.5, 0.6, 0.7, 0.8, 0.9]
auc_values = [
    [0.6895, 0.7284, 0.745, 0.7305, 0.7465],
    [0.7725, 0.7724, 0.7918, 0.7508, 0.7768], 
    [0.6427, 0.6677, 0.6614, 0.6972, 0.6711], 
    [0.6639, 0.6477, 0.7512, 0.6838, 0.7645], 
    [0.7988, 0.8262, 0.8248, 0.8146, 0.7859], 
]

# 绘制图形
plt.plot(sampling_rounds, auc_values[0], label=datasets[0], lw=2.0, marker='^', markersize=6)
plt.plot(sampling_rounds, auc_values[1], label=datasets[1], lw=2.0, marker='X', markersize=6)
plt.plot(sampling_rounds, auc_values[2], label=datasets[2], lw=2.0, marker='d', markersize=6)
plt.plot(sampling_rounds, auc_values[3], label=datasets[3], lw=2.0, marker='s', markersize=6)
plt.plot(sampling_rounds, auc_values[4], label=datasets[4], lw=2.0, marker='o', markersize=6)

plt.xlim([0.49, 0.91])
# plt.ylim([0.0, 1.05])
# sns.despine()
# 改变X轴刻度标签的字体大小
plt.xticks(fontsize=20)
# 如果需要同时改变Y轴刻度标签
plt.yticks(fontsize=20)
plt.xlabel(xlabel, fontsize=24)
plt.ylabel(ylabel, fontsize=24)
plt.legend(loc="lower right", prop={'size': 16, 'family': 'serif', 'weight': 'bold'})
plt.savefig('./test.png', dpi=300, bbox_inches='tight')
plt.show()


# 折线图模版

import matplotlib.pyplot as plt
import seaborn as sns

xlabel = "Number of Clusters"
ylabel = "AUC"

sns.set_theme(style="whitegrid", rc={"axes.linewidth": 1, "axes.edgecolor": "black"})
plt.figure(figsize=(8, 8))
# 或者全局设置字体为Arial
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.size'] = 12

# 数据
datasets = ['Cora', 'Citeseer', 'Citation', 'ACM', 'Pubmed']
# sampling_rounds = range(0, 100)
sampling_rounds = [3, 5, 7, 9, 11]
auc_values = [
    [0.5671, 0.6888, 0.7465, 0.7521, 0.7566], 
    [0.6512, 0.7622, 0.7768, 0.7937, 0.7893],
    [0.6880, 0.6711, 0.6554, 0.6896, 0.6761], 
    [0.6890, 0.6690, 0.6820, 0.7645, 0.7067],
    [0.6769, 0.7512, 0.6841, 0.7859, 0.7731], 
]

# 绘制图形
plt.plot(sampling_rounds, auc_values[0], label=datasets[0], lw=2.0, marker='^', markersize=6)
plt.plot(sampling_rounds, auc_values[1], label=datasets[1], lw=2.0, marker='X', markersize=6)
plt.plot(sampling_rounds, auc_values[2], label=datasets[2], lw=2.0, marker='d', markersize=6)
plt.plot(sampling_rounds, auc_values[3], label=datasets[3], lw=2.0, marker='s', markersize=6)
plt.plot(sampling_rounds, auc_values[4], label=datasets[4], lw=2.0, marker='o', markersize=6)

plt.xlim([2.8, 11.2])
# plt.ylim([0.0, 1.05])
# sns.despine()
# 改变X轴刻度标签的字体大小
plt.xticks([3, 5, 7, 9, 11], fontsize=20)
# 如果需要同时改变Y轴刻度标签
plt.yticks(fontsize=20)
plt.xlabel(xlabel, fontsize=24)
plt.ylabel(ylabel, fontsize=24)
plt.legend(loc="lower right", prop={'size': 16, 'family': 'serif', 'weight': 'bold'})
plt.savefig('./test.png', dpi=300, bbox_inches='tight')
plt.show()


# 折线图模版

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FixedLocator

xlabel = "Embedding Dimension"
ylabel = "AUC"

sns.set_theme(style="whitegrid", rc={"axes.linewidth": 1, "axes.edgecolor": "black"})
plt.figure(figsize=(8, 8))
ax = plt.gca()
# 或者全局设置字体为Arial
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.size'] = 18

# 数据
datasets = ['Cora', 'Citeseer', 'Citation', 'ACM', 'Pubmed']
# sampling_rounds = range(0, 100)
sampling_rounds = [32, 64, 128, 256, 512]
auc_values = [
    [0.7902, 0.7465, 0.7468, 0.7422, 0.7661], 
    [0.7403, 0.7768, 0.7795, 0.7672, 0.7766], 
    [0.67, 0.6711, 0.68, 0.5925, 0.4995],
    [0.6932, 0.7645, 0.7048, 0.6427, 0.6971],
    [0.7525, 0.7859, 0.7376, 0.7192, 0.7384], 
]

# 绘制图形
ax.plot(sampling_rounds, auc_values[0], label=datasets[0], lw=2.0, marker='^', markersize=6)
ax.plot(sampling_rounds, auc_values[1], label=datasets[1], lw=2.0, marker='X', markersize=6)
ax.plot(sampling_rounds, auc_values[2], label=datasets[2], lw=2.0, marker='d', markersize=6)
ax.plot(sampling_rounds, auc_values[3], label=datasets[3], lw=2.0, marker='s', markersize=6)
ax.plot(sampling_rounds, auc_values[4], label=datasets[4], lw=2.0, marker='o', markersize=6)


ax.set_xscale('log', base=2)
ax.xaxis.set_major_locator(FixedLocator([32, 64, 128, 256, 512]))
ax.set_xticklabels(['32', '64', '128', '256', '512'], fontsize=20)

yticks = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
ax.set_yticks(yticks)
ax.set_yticklabels([f'{tick:.2f}' for tick in yticks], fontsize=20)

ax.set_xlim(30, 550)
ax.set_xlabel(xlabel, fontsize=24)
ax.set_ylabel(ylabel, fontsize=24)
ax.tick_params(axis='both', labelsize=20)
legend = ax.legend(loc='lower left', prop={'size': 16, 'family': 'serif', 'weight': 'bold'})

plt.savefig('./test.png', dpi=300, bbox_inches='tight')
plt.show()