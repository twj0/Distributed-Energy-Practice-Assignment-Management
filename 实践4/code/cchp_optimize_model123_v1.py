"""
冷热电联供系统优化平台 - 终极优化版
作者：人工智能助手
版本：2.1
最后更新：2023-12-01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
import time

# 设置 matplotlib 支持中文显示，并正确显示负号
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CCHPOptimizer:
    """
    冷热电联供系统严格约束优化器
    
    功能特性：
    1. 智能梯度下降算法：结合动量加速和自适应步长
    2. 双重约束机制：参数硬约束 + 目标函数软约束
    3. 实时参数监控与修正
    4. 多维度优化过程可视化
    5. 自动收敛诊断与预警
    """
    
    # 定义参数的取值范围
    PARAM_BOUNDS = {
        'power_ratio': (0.0, 1.0),    # 发电效率系数范围
        'heat_ratio': (0.0, 1.0)      # 余热回收系数范围
    }
    
    # 优化器超参数配置
    DEFAULT_EPSILON = 1e-4          # 默认收敛阈值
    DEFAULT_MAX_ITERS = 1000         # 默认最大迭代次数
    MAX_STEP_ALPHA = 0.5            # 最大单次步长
    MOMENTUM_FACTOR = 0.9           # 动量因子
    GRAD_EPS = 1e-6                 # 梯度计算步长

    def __init__(self, mode='mode1', epsilon=None, max_iters=None):
        """
        初始化优化器
        :param mode: 运行模式 (mode1/mode2/mode3)
        :param epsilon: 收敛阈值，默认1e-4
        :param max_iters: 最大迭代次数，默认1000
        """
        self.mode = mode.lower()
        self.epsilon = epsilon or self.DEFAULT_EPSILON
        self.max_iters = max_iters or self.DEFAULT_MAX_ITERS
        
        # 初始化运行参数
        self.current_params = np.array([0.0, 0.0])  # [发电效率, 余热回收]
        self.velocity = np.zeros(2)  # 动量项
        
        # 优化过程记录
        self.param_history = []
        self.cost_history = []
        self.violation_history = []
        
        # 配置目标函数
        self._init_objective_function()

    def _init_objective_function(self):
        """根据模式初始化目标函数"""
        func_map = {
            'mode1': self._mode1_objective,
            'mode2': self._mode2_objective,
            'mode3': self._mode3_objective
        }
        if self.mode not in func_map:
            raise ValueError(f"无效模式：{self.mode}，可用模式：{list(func_map.keys())}")
        self.objective = func_map[self.mode]

    def _apply_constraints(self, params):
        """
        应用参数硬约束
        :param params: 待约束的参数数组
        :return: 约束后的参数数组
        """
        # 使用np.clip进行边界约束
        constrained = np.clip(
            params,
            [self.PARAM_BOUNDS['power_ratio'][0], 
             self.PARAM_BOUNDS['heat_ratio'][0]],
            [self.PARAM_BOUNDS['power_ratio'][1],
             self.PARAM_BOUNDS['heat_ratio'][1]]
        )
        
        # 记录违规程度（L2范数）
        violation = np.linalg.norm(params - constrained)
        self.violation_history.append(violation)
        
        return constrained

    def _boundary_penalty(self, params):
        """
        计算边界违规惩罚项
        :param params: 待检查的参数数组
        :return: 惩罚值
        """
        penalty = 0.0
        
        # 发电效率系数违规惩罚
        if params[0] < self.PARAM_BOUNDS['power_ratio'][0]:
            penalty += 1e12 * (self.PARAM_BOUNDS['power_ratio'][0] - params[0])**2
        elif params[0] > self.PARAM_BOUNDS['power_ratio'][1]:
            penalty += 1e12 * (params[0] - self.PARAM_BOUNDS['power_ratio'][1])**2
        
        # 余热回收系数违规惩罚
        if params[1] < self.PARAM_BOUNDS['heat_ratio'][0]:
            penalty += 1e12 * (self.PARAM_BOUNDS['heat_ratio'][0] - params[1])**2
        elif params[1] > self.PARAM_BOUNDS['heat_ratio'][1]:
            penalty += 1e12 * (params[1] - self.PARAM_BOUNDS['heat_ratio'][1])**2
        
        return penalty

    def _mode1_objective(self, params):
        """模式1目标函数：高电负荷运行"""
        base_cost = 10*(params[0]-1)**2 + (params[1]+1)**4
        return base_cost + self._boundary_penalty(params)

    def _mode2_objective(self, params):
        """模式2目标函数：热电平衡运行"""
        base_cost = 100*(params[0]**2 - params[1])**2 + (params[0]-1)**2
        return base_cost + self._boundary_penalty(params)

    def _mode3_objective(self, params):
        """模式3目标函数：余热优先运行"""
        base_cost = 100*(params[0]**2 - 3*params[1])**2 + (params[0]-1)**2
        return base_cost + self._boundary_penalty(params)

    def _compute_gradient(self, params):
        """
        计算带安全保护的梯度
        :param params: 当前参数点
        :return: 梯度向量
        """
        grad = np.zeros(2)
        try:
            for i in range(2):
                # 中心差分法计算梯度
                params_plus = self._apply_constraints(params.copy())
                params_plus[i] += self.GRAD_EPS
                
                params_minus = self._apply_constraints(params.copy())
                params_minus[i] -= self.GRAD_EPS
                
                grad[i] = (self.objective(params_plus) - self.objective(params_minus)) / (2*self.GRAD_EPS)
        except Exception as e:
            print(f"梯度计算异常：{str(e)}")
        return grad

    def _line_search(self, direction):
        """
        安全步长搜索
        :param direction: 搜索方向
        :return: 最优步长
        """
        def cost_at_alpha(alpha):
            # 限制步长在安全范围内
            alpha_clipped = min(alpha, self.MAX_STEP_ALPHA)
            new_params = self.current_params + alpha_clipped * direction
            return self.objective(self._apply_constraints(new_params))
        
        # 使用Bounded方法进行一维搜索
        result = minimize_scalar(
            cost_at_alpha,
            bounds=(0, self.MAX_STEP_ALPHA),
            method='bounded'
        )
        return result.x

    def optimize(self):
        """
        执行优化过程
        :return: 优化结果字典
        """
        # 初始化记录
        self.param_history = [self.current_params.copy()]
        self.cost_history = [self.objective(self.current_params)]
        self.violation_history = []
        
        k = 0
        convergence_status = False
        
        while k < self.max_iters:
            # 计算梯度
            grad = self._compute_gradient(self.current_params)
            
            # 动量加速：v = γv + (1-γ)∇
            self.velocity = self.MOMENTUM_FACTOR * self.velocity + (1 - self.MOMENTUM_FACTOR) * grad
            direction = -self.velocity
            
            # 步长搜索
            alpha = self._line_search(direction)
            
            # 参数更新
            new_params = self.current_params + alpha * direction
            new_params = self._apply_constraints(new_params)
            
            # 记录状态
            self.param_history.append(new_params.copy())
            self.cost_history.append(self.objective(new_params))
            
            # 收敛判断
            delta = np.linalg.norm(new_params - self.current_params)
            if delta < self.epsilon:
                convergence_status = True
                self.current_params = new_params
                break
                
            self.current_params = new_params
            k += 1

        # 生成最终报告
        return self._compile_results(k, convergence_status)

    def _compile_results(self, iterations, converged):
        """
        整理优化结果
        :param iterations: 实际迭代次数
        :param converged: 是否收敛标志
        :return: 结果字典
        """
        return {
            'converged': converged,
            'iterations': iterations,
            'optimal_power': self.current_params[0],
            'optimal_heat': self.current_params[1],
            'final_cost': self.cost_history[-1],
            'param_history': np.array(self.param_history),
            'cost_history': self.cost_history,
            'violation_history': self.violation_history
        }

    def visualize(self, results, show=True, save_path=None):
        """
        可视化优化过程
        :param results: 优化结果字典
        :param show: 是否显示图表
        :param save_path: 图片保存路径
        """
        plt.figure(figsize=(16, 12))
        
        # 参数空间轨迹
        plt.subplot(2, 2, 1)
        plt.plot(results['param_history'][:,0], results['param_history'][:,1], 'b.-', alpha=0.6)
        plt.scatter(results['param_history'][0,0], results['param_history'][0,1], 
                   c='red', label='起点', zorder=2)
        plt.scatter(results['optimal_power'], results['optimal_heat'],
                   c='green', label='最优点', zorder=2)
        plt.xlim(*self.PARAM_BOUNDS['power_ratio'])
        plt.ylim(*self.PARAM_BOUNDS['heat_ratio'])
        plt.xlabel('发电效率系数', fontsize=10)
        plt.ylabel('余热回收系数', fontsize=10)
        plt.title(f'参数优化轨迹 (迭代次数={results["iterations"]})', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 成本收敛曲线
        plt.subplot(2, 2, 2)
        plt.semilogy(results['cost_history'], 'r-', alpha=0.8)
        plt.xlabel('迭代次数', fontsize=10)
        plt.ylabel('综合成本（对数）', fontsize=10)
        plt.title('成本收敛过程', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 参数变化过程
        plt.subplot(2, 2, 3)
        x_axis = np.arange(len(results['param_history']))
        plt.plot(x_axis, results['param_history'][:,0], 'g-', label='发电效率')
        plt.plot(x_axis, results['param_history'][:,1], 'm-', label='余热回收')
        plt.xticks(np.linspace(0, len(x_axis), 5), 
                  np.linspace(0, results['iterations'], 5, dtype=int))
        plt.ylim(-0.1, 1.1)
        plt.xlabel('迭代次数', fontsize=10)
        plt.ylabel('参数值', fontsize=10)
        plt.title('参数迭代过程', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 约束违规记录
        plt.subplot(2, 2, 4)
        plt.semilogy(results['violation_history'], 'k-', alpha=0.8)
        plt.xlabel('迭代次数', fontsize=10)
        plt.ylabel('违规程度（对数）', fontsize=10)
        plt.title('边界约束违规记录', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()

if __name__ == "__main__":
    print("冷热电联供系统优化平台")
    print("="*50)
    
    # 用户交互配置
    mode = input("选择运行模式 (1:高电负荷 2:热电平衡 3:余热优先): ").strip()
    mode_map = {'1':'mode1', '2':'mode2', '3':'mode3'}
    mode = mode_map.get(mode, 'mode2')
    
    custom_init = input("自定义初始参数? (y/n): ").lower()
    if custom_init == 'y':
        x1 = float(input("发电效率初始值(0-1): "))
        x2 = float(input("余热回收初始值(0-1): "))
        initial_params = [max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))]
    else:
        initial_params = [0.0, 0.0]
    
    # 高级配置
    max_iters = int(input(f"最大迭代次数 (默认{CCHPOptimizer.DEFAULT_MAX_ITERS}): ") 
                   or CCHPOptimizer.DEFAULT_MAX_ITERS)
    epsilon = float(input(f"收敛阈值 (默认{CCHPOptimizer.DEFAULT_EPSILON:.0e}): ") 
                  or CCHPOptimizer.DEFAULT_EPSILON)
    
    # 初始化优化器
    optimizer = CCHPOptimizer(
        mode=mode,
        epsilon=epsilon,
        max_iters=max_iters
    )
    optimizer.current_params = np.array(initial_params)
    
    # 执行优化
    start_time = time.time()
    results = optimizer.optimize()
    elapsed = time.time() - start_time
    
    # 显示报告
    print("\n优化结果报告")
    print("="*50)
    print(f"运行模式: \t{mode}")
    print(f"收敛状态: \t{'成功' if results['converged'] else '未收敛'}")
    print(f"迭代次数: \t{results['iterations']} / {max_iters}")
    print(f"计算耗时: \t{elapsed:.2f}秒")
    print(f"最终发电效率: \t{results['optimal_power']:.6f}")
    print(f"最终余热回收: \t{results['optimal_heat']:.6f}")
    print(f"综合运行成本: \t{results['final_cost']:.4e}")
    
    # 可视化并保存结果
    fig_path = os.path.join('fig', f'optimization_{mode}.png')
    optimizer.visualize(results, save_path=fig_path, show=True)