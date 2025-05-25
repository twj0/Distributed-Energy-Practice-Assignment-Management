import pandas as pd
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  

# 读取Excel文件
df = pd.read_excel("month.xlsx")    # 同一个文件夹下

# 筛选出月份列中不是全年的数据
df = df[df['月份'] != '全年']

# 将月份列转换为整数类型
df['月份'] = df['月份'].astype(int)

# 计算月平均热电比，热电比 = (空调制冷 + 空调供热 + 生活热水 ) / 电力
monthly_thermoelectric_ratio = (df['空调制冷'] + df['空调供热'] + df['生活热水']) / df['电力']


# 绘制月平均热电比折线图
plt.plot(df['月份'], monthly_thermoelectric_ratio, marker='o')
plt.xlabel('Month')
plt.ylabel('monthly thermoelectric ratio')
plt.title('月平均热电比折线图')
plt.grid(True)
plt.xticks(df['月份']) # 添加x轴刻度标签
plt.show()
