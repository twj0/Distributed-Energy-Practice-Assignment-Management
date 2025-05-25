import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

col_names = ['冬季_空调制冷', '冬季_空调供热', '冬季_生活热水', '冬季_电力', 
                '夏季_空调制冷', '夏季_空调供热', '夏季_生活热水', '夏季_电力', 
                '过渡季_空调制冷', '过渡季_空调供热', '过渡季_生活热水', '过渡季_电力']

# 指定文件路径
df = pd.read_excel("clock.xlsx", header=None, skiprows=1, names=col_names)    # 请根据实际文件路径修改

# 将相关列转换为数值类型
cols = ['冬季_空调制冷', '冬季_空调供热', '冬季_生活热水', '冬季_电力', 
                '夏季_空调制冷', '夏季_空调供热', '夏季_生活热水', '夏季_电力', 
                '过渡季_空调制冷', '过渡季_空调供热', '过渡季_生活热水', '过渡季_电力']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')



df['冬季热负荷'] = df['冬季_空调制冷'] + df['冬季_空调供热'] + df['冬季_生活热水'] + df['冬季_电力']
df['夏季热负荷'] = df['夏季_空调制冷'] + df['夏季_空调供热'] + df['夏季_生活热水'] + df['夏季_电力']
df['过渡季热负荷'] = df['过渡季_空调制冷'] + df['过渡季_空调供热'] + df['过渡季_生活热水'] + df['过渡季_电力']
df['总热负荷'] = df['冬季热负荷'] + df['夏季热负荷'] + df['过渡季热负荷']


# 生成 24 小时的时间序列
hours = list(range(25))

# 创建画布
plt.figure(figsize=(15, 10))


plt.subplot(4, 1, 1)
plt.plot(hours, df['冬季_空调制冷']+df['夏季_空调制冷']+df['过渡季_空调制冷'].head(25), marker='o', color='blue', label='总空调制冷')
plt.plot(hours, df['冬季_空调供热']+df['夏季_空调供热']+df['过渡季_空调供热'].head(25), marker='o', color='red', label='总空调供热')
plt.plot(hours, df['冬季_生活热水']+df['夏季_生活热水']+df['过渡季_生活热水'].head(25), marker='o', color='green', label='总生活热水')
plt.plot(hours, df['冬季_电力']+df['夏季_电力']+df['过渡季_电力'].head(25), marker='o', color='black', label='总电力')
plt.title('各季各项热负荷')
plt.xlabel('时间（小时）')
plt.xticks(hours)  # 设置 x 轴刻度为 24 小时
plt.ylabel('负荷（%）')
plt.legend()
plt.grid(True)


plt.subplot(4, 1, 2)
plt.plot(hours, df['冬季_空调制冷'].head(25), marker='o', color='blue', label='冬季空调制冷')
plt.plot(hours, df['冬季_空调供热'].head(25), marker='o', color='red', label='冬季空调供热')
plt.plot(hours, df['冬季_生活热水'].head(25), marker='o', color='green', label='冬季生活热水')
plt.plot(hours, df['冬季_电力'].head(25), marker='o', color='black', label='冬季电力')
plt.title('冬季各项负荷')
plt.xlabel('时间（小时）')
plt.xticks(hours)  # 设置 x 轴刻度为 24 小时
plt.ylabel('负荷（%）')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(hours, df['夏季_空调制冷'].head(25), marker='o', color='blue', label='夏季空调制冷')
plt.plot(hours, df['夏季_空调供热'].head(25), marker='o', color='red', label='夏季空调供热')
plt.plot(hours, df['夏季_生活热水'].head(25), marker='o', color='green', label='夏季生活热水')
plt.plot(hours, df['夏季_电力'].head(25), marker='o', color='black', label='夏季电力')
plt.title('夏季各项负荷')
plt.xlabel('时间（小时）')
plt.xticks(hours)  # 设置 x 轴刻度为 24 小时
plt.ylabel('负荷（%）')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(hours, df['过渡季_空调制冷'].head(25), marker='o', color='blue', label='过渡季空调制冷')
plt.plot(hours, df['过渡季_空调供热'].head(25), marker='o', color='red', label='过渡季空调供热')
plt.plot(hours, df['过渡季_生活热水'].head(25), marker='o', color='green', label='过渡季生活热水')
plt.plot(hours, df['过渡季_电力'].head(25), marker='o', color='black', label='过渡季电力')
plt.title('过渡季各项负荷')
plt.xlabel('时间（小时）')
plt.xticks(hours)  # 设置 x 轴刻度为 24 小时
plt.ylabel('负荷（%）')
plt.legend()
plt.grid(True)


# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()
