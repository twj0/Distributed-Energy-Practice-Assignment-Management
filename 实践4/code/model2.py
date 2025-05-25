import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
class CCHPOptimization:
    """微型冷热电联供系统热经济性优化"""
    
    def __init__(self, objective_function='function1', epsilon=1.0e-4, max_iterations=1000):
        """
        初始化优化器
        
        参数:
        - objective_function: 目标函数选择 ('function1', 'function2', 'function3')
        - epsilon: 迭代终止精度
        - max_iterations: 最大迭代次数
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        # 选择目标函数
        if objective_function == 'function1':
            self.objective_function = self.function1
        elif objective_function == 'function2':
            self.objective_function = self.function2
        elif objective_function == 'function3':
            self.objective_function = self.function3
        else:
            raise ValueError("Invalid objective function selection")
        
        # 初始点
        self.x0 = np.array([0.0, 0.0])
        
        # 记录迭代历史
        self.x_history = []
        self.f_history = []
        self.error_history = []
    
    def function1(self, x):
        """目标函数1: f(x) = 10(x1-1)^2 + (x2+1)^4"""
        return 10 * (x[0] - 1)**2 + (x[1] + 1)**4
    
    def function2(self, x):
        """目标函数2: f(x) = 100(x1^2 - x2)^2 + (x1 - 1)^2"""
        return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2
    
    def function3(self, x):
        """目标函数3: f(x) = 100(x1^2 - 3*x2)^2 + (x1 - 1)^2"""
        return 100 * (x[0]**2 - 3*x[1])**2 + (x[0] - 1)**2
    
    def gradient(self, x):
        """计算目标函数在点x处的梯度"""
        h = 1e-6  # 微小步长用于数值梯度计算
        grad = np.zeros_like(x)
        
        # 对每个变量计算偏导数
        for i in range(len(x)):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            grad[i] = (self.objective_function(x_plus_h) - self.objective_function(x)) / h
        
        return grad
    
    def line_search(self, x, search_direction):
        """一维搜索确定步长a"""
        def f_along_line(alpha):
            return self.objective_function(x + alpha * search_direction)
        
        # 使用scipy中的一维最小化函数确定步长
        result = minimize_scalar(f_along_line, bounds=(0, 10), method='bounded')
        return result.x
    
    def optimize(self):
        """使用最速下降法进行优化"""
        start_time = time.time()
        
        x_current = self.x0.copy()
        self.x_history.append(x_current.copy())
        self.f_history.append(self.objective_function(x_current))
        self.error_history.append(0)  # 初始误差设为0
        
        for k in range(self.max_iterations):
            # 计算梯度(搜索方向)
            gradient = self.gradient(x_current)
            search_direction = -gradient  # 负梯度方向
            
            # 确定步长
            alpha = self.line_search(x_current, search_direction)
            
            # 计算新点
            x_new = x_current + alpha * search_direction
            
            # 计算误差
            error = np.linalg.norm(x_new - x_current)
            self.error_history.append(error)
            
            # 更新当前点
            x_current = x_new.copy()
            
            # 记录历史
            self.x_history.append(x_current.copy())
            self.f_history.append(self.objective_function(x_current))
            
            # 判断是否达到收敛条件
            if error < self.epsilon:
                break
        
        end_time = time.time()
        
        # 打印结果
        print(f"优化结束, 共迭代 {k+1} 次, 耗时 {end_time - start_time:.6f} 秒")
        print(f"目标函数值: {self.f_history[-1]:.10f}")
        print(f"最优点: x1 = {x_current[0]:.10f}, x2 = {x_current[1]:.10f}")
        print(f"最终误差: {error:.10f}")
        
        return {
            'iterations': k+1,
            'function_value': self.f_history[-1],
            'optimal_point': x_current,
            'final_error': error,
            'time': end_time - start_time
        }
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 10))
        
        # 迭代次数和函数值图
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.f_history)), self.f_history, 'b-', linewidth=2)
        plt.title('目标函数值随迭代次数的变化')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.grid(True)
        
        # 迭代次数和误差图
        plt.subplot(2, 1, 2)
        plt.semilogy(range(1, len(self.error_history)), self.error_history[1:], 'r-', linewidth=2)
        plt.title('迭代误差随迭代次数的变化')
        plt.xlabel('迭代次数')
        plt.ylabel('误差 (对数尺度)')
        plt.grid(True)
        
        plt.tight_layout()
        return plt

# 微型冷热电联供系统热经济性模型说明
"""
一个典型的微型冷热电联供系统热经济性模型可以表示为:

目标函数：最小化系统全年总费用 (包括投资费用、燃料费用、运行维护费用等)
总费用 = 投资费用 + 燃料费用 + 运行维护费用 - 节能收益

约束条件:
1. 设备容量约束
2. 供需平衡约束
3. 设备运行特性约束
4. 环境排放约束

在本例中，我们使用作业中提供的三个目标函数进行优化演示。
"""

def main():
    print("微型冷热电联供系统热经济性优化")
    print("-"*50)
    
    # 选择目标函数
    print("请选择目标函数:")
    print("1. f(x) = 10(x1-1)^2 + (x2+1)^4")
    print("2. f(x) = 100(x1^2 - x2)^2 + (x1 - 1)^2")
    print("3. f(x) = 100(x1^2 - 3*x2)^2 + (x1 - 1)^2")
    
    choice = input("请输入选择 (1-3): ")
    
    function_map = {
        '1': 'function1',
        '2': 'function2',
        '3': 'function3'
    }
    
    if choice in function_map:
        # 创建优化器
        optimizer = CCHPOptimization(objective_function=function_map[choice])
        
        # 执行优化
        result = optimizer.optimize()
        
        # 绘制收敛曲线
        plt = optimizer.plot_convergence()
        plt.show()
    else:
        print("无效的选择!")

if __name__ == "__main__":
    main()