import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

dir = 'fig'
if not os.path.exists(dir):
    os.makedirs(dir)
# 图片支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
try:
    font = FontProperties(family='SimHei')
except:
    font = FontProperties()

def calculate_weights_from_matrix(matrix):
    """
    使用层次分析法（AHP）从判断矩阵计算权重
    
    参数:
        matrix: 判断矩阵，numpy数组
    
    返回:
        权重向量，一致性比率CR
    """
    n = len(matrix)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    eigenvector = eigenvectors[:, max_idx].real
    
    # 归一化特征向量得到权重
    weights = eigenvector / np.sum(eigenvector)
    
    # 计算一致性指标CI
    CI = (lambda_max - n) / (n - 1)
    
    # 随机一致性指标RI
    RI_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_values.get(n, 0)
    
    # 一致性比率CR
    CR = CI / RI if RI != 0 else 0
    
    return weights, CR

def normalize_data(data, types):
    """
    标准化数据
    
    参数:
        data: 原始数据，numpy数组
        types: 指标类型，1表示正指标（越大越好），0表示逆指标（越小越好）
    
    返回:
        标准化后的数据
    """
    norm_data = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        max_val, min_val = np.max(data[:, j]), np.min(data[:, j])
        if max_val == min_val:  # 处理所有值相等的情况
            norm_data[:, j] = np.ones(data.shape[0])
            continue
            
        if types[j] == 1:  # 正指标，越大越好
            norm_data[:, j] = (data[:, j] - min_val) / (max_val - min_val)
        else:  # 逆指标，越小越好
            norm_data[:, j] = (max_val - data[:, j]) / (max_val - min_val)
            
    return norm_data

def grey_relation_analysis(norm_data, weights, rho=0.5):
    """
    灰色关联度分析
    
    参数:
        norm_data: 标准化后的数据
        weights: 指标权重
        rho: 分辨系数，通常为0.5
    
    返回:
        关联系数矩阵，灰色关联度
    """
    # 理想参考序列：标准化后理想值均为1
    ref = np.ones(norm_data.shape[1])
    
    # 计算关联系数
    delta = np.abs(norm_data - ref)
    min_delta = np.min(delta)
    max_delta = np.max(delta)
    
    # 计算关联系数矩阵
    xi = (min_delta + rho * max_delta) / (delta + rho * max_delta)
    
    # 计算加权关联度
    r = np.dot(xi, weights)
    
    return xi, r

def main():
    # 步骤1: 输入原始数据
    data = np.array([
        # 每行对应一个指标，每列对应方案A-E
        [2, 4, 5, 1, 3],    # 初投资/元 (↓)
        [2, 1, 4, 3, 5],    # 投资回收期/年 (↓)
        [2, 5, 3, 4, 1],    # 总费用年值/元 (↓)
        [2, 4, 3, 5, 1],    # 净现值/元 (↑)
        [2, 3, 4, 5, 1],    # 氮氧化物/(g/kW·h) (↓)
        [1, 5, 3, 2, 4],    # CO/(g/kW·h) (↓)
        [3, 5, 4, 1, 2],    # CO₂/(g/kW·h) (↓)
        [5, 3, 2, 1, 4],    # 技术先进性 (↑)
        [1, 5, 2, 3, 4],    # 安全性 (↑)
        [3, 1, 4, 5, 2],    # 维护方便性 (↑)
        [5, 2, 4, 1, 3],    # 一次能源比 (↓)
        [1, 4, 3, 2, 5]     # 噪声/dB (↓)
    ]).T  # 转置为5个方案 × 12个指标

    
    # 指标名称
    index_names = [
        '初投资', '投资回收期', '总费用年值', '净现值',
        '氮氧化物', 'CO', 'CO₂', '技术先进性',
        '安全性', '维护方便性', '一次能源比', '噪声'
    ]
    
    # 指标性质：1为正指标（越大越好），0为逆指标（越小越好）
    index_types = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    
    # 步骤2: 从判断矩阵计算权重
    # 从PDF文件中提取的判断矩阵
    F1_matrix = np.array([
        [1, 3, 5, 7],
        [1/3, 1, 6, 1/4],
        [1/5, 1/6, 1, 1/3],
        [1/7, 4, 3, 1]
    ])
    
    F2_matrix = np.array([
        [1, 6, 4],
        [1/6, 1, 1/3],
        [1/4, 3, 1]
    ])
    
    F3_matrix = np.array([
        [1, 1/3, 5],
        [3, 1, 7],
        [1/5, 1/7, 1]
    ])
    
    # 综合判断矩阵（一级指标）
    comprehensive_matrix = np.array([
        [1, 5, 9, 3, 7],
        [1/5, 1, 6, 3, 1/4],
        [1/9, 1/6, 1, 5, 3],
        [1/3, 1/3, 1/5, 1, 9],
        [1/7, 4, 1/3, 1/9, 1]
    ])
    
    # 计算权重
    W_level1, CR1 = calculate_weights_from_matrix(comprehensive_matrix)
    print(f"一级指标权重计算完成，一致性比率CR = {CR1:.4f}")
    
    W_F1, CR_F1 = calculate_weights_from_matrix(F1_matrix)
    print(f"经济性指标权重计算完成，一致性比率CR = {CR_F1:.4f}")
    
    W_F2, CR_F2 = calculate_weights_from_matrix(F2_matrix)
    print(f"环境性指标权重计算完成，一致性比率CR = {CR_F2:.4f}")
    
    W_F3, CR_F3 = calculate_weights_from_matrix(F3_matrix)
    print(f"社会性指标权重计算完成，一致性比率CR = {CR_F3:.4f}")
    
    # 一级指标各自只有一个二级指标，权重为1
    W_F4 = np.array([1.0])
    W_F5 = np.array([1.0])
    
    # 计算综合权重
    W_total = np.concatenate([
        W_level1[0] * W_F1,  # 经济性指标权重
        W_level1[1] * W_F2,  # 环境性指标权重
        W_level1[2] * W_F3,  # 社会性指标权重
        W_level1[3] * W_F4,  # 性能指标权重
        W_level1[4] * W_F5   # 噪声指标权重
    ])
    
    print("\n综合权重:")
    for i, name in enumerate(index_names):
        print(f"{name}: {W_total[i]:.4f}")
    
    # 步骤3: 标准化处理
    norm_data = normalize_data(data, index_types)
    
    # 步骤4: 灰色关联度计算
    xi, r = grey_relation_analysis(norm_data, W_total)
    
    # 步骤5: 输出结果
    schemes = ['方案A', '方案B', '方案C', '方案D', '方案E']
    result = pd.DataFrame({'方案': schemes, '灰色关联度': r})
    result = result.sort_values(by='灰色关联度', ascending=False)
    
    print("\n灰色关联度计算结果：")
    print(result)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    bars = plt.bar(schemes, r, color='skyblue')
    plt.title('各方案灰色关联度比较', fontproperties=font)
    plt.xlabel('方案', fontproperties=font)
    plt.ylabel('灰色关联度', fontproperties=font)
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fig/grey_relation_result.png', dpi=300)
    plt.close()
    
    # 步骤6: 标准化矩阵和关联系数矩阵
    norm_df = pd.DataFrame(norm_data, index=schemes, columns=index_names)
    xi_df = pd.DataFrame(xi, index=schemes, columns=index_names)
    
    print("\n标准化矩阵：")
    print(norm_df)
    print("\n关联系数矩阵：")
    print(xi_df)
    
    # 输出结果到Excel
    with pd.ExcelWriter('fig/grey_relation_analysis_results.xlsx') as writer:
        result.to_excel(writer, sheet_name='结果排序', index=False)
        norm_df.to_excel(writer, sheet_name='标准化矩阵')
        xi_df.to_excel(writer, sheet_name='关联系数矩阵')
        
        # 创建权重表
        weight_df = pd.DataFrame({
            '指标': index_names,
            '权重': W_total,
            '指标类型': ['逆指标' if t == 0 else '正指标' for t in index_types]
        })
        weight_df.to_excel(writer, sheet_name='指标权重', index=False)

if __name__ == "__main__":
    main()