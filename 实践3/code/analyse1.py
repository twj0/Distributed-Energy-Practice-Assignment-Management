import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


dir = 'fig1'
if not os.path.exists(dir):
    os.makedirs(dir)

# ===================== 核心算法模块 =====================
def ahp_weight(matrix):
    """层次分析法权重计算（行和归一化法）"""
    row_sum = np.sum(matrix, axis=1)
    weights = row_sum / np.sum(row_sum)
    return weights

def fuzzy_quantization(data):
    """模糊隶属函数量化（处理定性指标）"""
    def _f(x):
        if x >= 3:
            return 0.3915 * np.log(x) + 0.3699
        else:
            denominator = (x - 0.8942) ** 2 + 1e-9
            return 1 / (1 + 1.1086 / denominator)
    return np.vectorize(_f)(data)

def normalize(data, types):
    """指标标准化处理"""
    normalized = np.zeros_like(data, dtype=float)
    for col in range(data.shape[1]):
        min_val = np.min(data[:, col])
        max_val = np.max(data[:, col])
        if types[col] == 1:  # 正指标
            normalized[:, col] = (data[:, col] - min_val) / (max_val - min_val + 1e-9)
        else:  # 逆指标
            normalized[:, col] = (max_val - data[:, col]) / (max_val - min_val + 1e-9)
    return normalized

def grey_relation(norm_data, weights, rho=0.5):
    """灰色关联度计算"""
    ideal = np.ones(norm_data.shape[1])
    delta = np.abs(norm_data - ideal)
    min_delta = np.min(delta)
    max_delta = np.max(delta)
    
    # 关联系数矩阵
    xi = (min_delta + rho*max_delta) / (delta + rho*max_delta)
    
    # 综合关联度
    return np.dot(xi, weights)

# ===================== 主程序 =====================
def main():
    # ------------ 数据准备 ------------
    # 每个方案包含12个指标的完整数据（不要分割成子列表）
    raw_data = np.array([
        # 方案A: [经济性4, 环境性3, 社会性3, 性能1, 噪声1]
        [535000, 6.42, 480137, 52582, 0.23, 0.45, 400, 0.8, 0.8, 0.6, 1.969, 65],
        # 方案B
        [680000, 6.63, 481374, 28108, 0.223, 0.6, 589, 0.8, 0.8, 0.6, 1.855, 65],
        # 方案C
        [504568, 4.86, 387851, 546633, 0.7, 0.8, 430, 0.6, 0.6, 0.6, 1.594, 80],
        # 方案D 
        [1580000, 8.8, 530131, -403086, 0.007, 0.001, 362, 0.934, 0.934, 0.8, 1.4, 60],
        # 方案E
        [290000, 6.73, 197179, 32136, 3.2, 4, 700, 0.4, 0.4, 0.934, 13.3, 56]
    ])  # shape (5,12) 已确认

    # 指标性质 (0:逆指标，1:正指标)
    index_types = [
        0,0,0,1,   # 经济性(4)
        0,0,0,     # 环境性(3)
        1,1,1,     # 社会性(3)
        0,0        # 性能，噪声(2)
    ]

    # ------------ 权重计算 ------------
    # 一级指标判断矩阵（经济性F1、环境性F2、社会性F3、性能F4、噪声F5）
    L1_matrix = np.array([
        [1, 7, 4, 5, 3],
        [1/7, 1, 1/4, 1/3, 1/5],
        [1/4, 4, 1, 2, 1/3],
        [1/5, 3, 1/2, 1, 1/4],
        [1/3, 5, 3, 4, 1]
    ])
    L1_weights = ahp_weight(L1_matrix)

    # 二级指标权重
    F1_matrix = np.array([  # 经济性
        [1, 3, 5, 7],
        [1/3, 1, 6, 1/4],
        [1/5, 1/6, 1, 1/3],
        [1/7, 4, 3, 1]
    ])
    F1_weights = ahp_weight(F1_matrix) * L1_weights[0]

    F2_matrix = np.array([  # 环境性
        [1, 6, 4],
        [1/6, 1, 1/3],
        [1/4, 3, 1]
    ])
    F2_weights = ahp_weight(F2_matrix) * L1_weights[1]

    F3_matrix = np.array([  # 社会性
        [1, 1/3, 5],
        [3, 1, 7],
        [1/5, 1/7, 1]
    ])
    F3_weights = ahp_weight(F3_matrix) * L1_weights[2]

    # 性能F4和噪声F5作为一级指标直接继承权重
    F4_weights = np.array([L1_weights[3]])
    F5_weights = np.array([L1_weights[4]])

    # 合成总权重
    total_weights = np.concatenate([F1_weights, F2_weights, F3_weights, F4_weights, F5_weights])

    # ------------ 数据处理 ------------
    # 定性指标模糊量化（社会性）
    raw_data[:, 7:10] = fuzzy_quantization(raw_data[:, 7:10])

    # 数据标准化
    norm_data = normalize(raw_data, index_types)

    # ------------ 灰色关联分析 ------------
    grey_scores = grey_relation(norm_data, total_weights)

    # ------------ 结果展示 ------------
    results = pd.DataFrame({
        '方案': ['方案A', '方案B', '方案C', '方案D', '方案E'],
        '灰色关联度': grey_scores
    }).sort_values('灰色关联度', ascending=False)

    print("综合评价结果排序：")
    print(results)

    # 可视化
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['方案'], results['灰色关联度'], color='skyblue')
    plt.title('冷热电联供系统综合评价结果')
    plt.ylabel('灰色关联度')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', 
                va='bottom', ha='center')
    plt.savefig('fig1/综合评价结果.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()