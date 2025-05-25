"""
冷热电联供系统优化平台 - 智能交互版 (v3.1)
整合功能：
1. 保留v2的交互式终端操作界面
2. 集成v3的增强收敛算法
3. 支持多算法选择
4. 动态参数验证系统
5. 实时收敛监控
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import time
import os

# 可视化配置
# plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedCCHPOptimizer:
    """
    增强型冷热电联供系统优化器
    
    改进点：
    - 混合梯度下降算法与拟牛顿法
    - 三重收敛条件监测
    - 动态参数边界管理
    - 实时进度反馈
    """
    
    # 系统参数边界
    PARAM_BOUNDS = Bounds([0.0, 0.0], [1.0, 1.0])
    
    def __init__(self, mode='mode1', config=None):
        """
        初始化优化器
        :param mode: 运行模式 (mode1/mode2/mode3)
        :param config: 配置字典
        """
        self.mode = mode
        self.config = {
            'max_iters': 1000,
            'epsilon': 1e-6,
            'learning_rate': 0.1,
            'momentum': 0.9,
            'adaptive_rate': True,
            'algorithm': 'hybrid',
            **(config or {})
        }
        
        # 初始化参数
        self.params = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        
        # 过程记录
        self.history = {
            'params': [],
            'costs': [],
            'gradients': [],
            'violations': [],
            'deltas': []
        }

        # 目标函数配置
        self._init_objective()

    def _init_objective(self):
        """初始化目标函数与梯度计算"""
        # 目标函数定义
        self.objective_func = {
            'mode1': lambda x: 10*(x[0]-1)**2 + (x[1]+1)**4,
            'mode2': lambda x: 100*(x[0]**2 -x[1])**2 + (x[0]-1)**2,
            'mode3': lambda x: 100*(x[0]**2 -3*x[1])**2 + (x[0]-1)**2
        }[self.mode]
        
        # 符号微分梯度
        self.gradient_func = {
            'mode1': lambda x: np.array([
                20*(x[0]-1),
                4*(x[1]+1)**3
            ]),
            'mode2': lambda x: np.array([
                400*x[0]*(x[0]**2 -x[1]) + 2*(x[0]-1),
                -200*(x[0]**2 -x[1])
            ]),
            'mode3': lambda x: np.array([
                400*x[0]*(x[0]**2 -3*x[1]) + 2*(x[0]-1),
                -600*(x[0]**2 -3*x[1])
            ])
        }[self.mode]

    def _project_params(self, params):
        """参数投影到可行域"""
        return np.clip(params, self.PARAM_BOUNDS.lb, self.PARAM_BOUNDS.ub)

    def _compute_violation(self, params):
        """计算边界违规程度"""
        return np.linalg.norm(params - self._project_params(params))

    def _adaptive_learning_rate(self, grad_norm):
        """动态学习率调整"""
        base_lr = self.config['learning_rate']
        if grad_norm > 1e-2:
            return base_lr * 0.5
        return min(base_lr * 1.2, 1.0)

    def _check_convergence(self, delta, grad_norm, cost_change):
        """三重收敛条件判断"""
        return (
            delta < self.config['epsilon'] or 
            grad_norm < 1e-5 or 
            abs(cost_change) < 1e-8
        )

    def optimize(self):
        """执行混合优化算法"""
        self.params = self._project_params(self.params)
        self.history['params'].append(self.params.copy())
        self.history['costs'].append(self.objective_func(self.params))
        
        try:
            for iter in range(self.config['max_iters']):
                # 梯度计算
                grad = self.gradient_func(self.params)
                grad_norm = np.linalg.norm(grad)
                
                # 动态参数调整
                if self.config['adaptive_rate']:
                    lr = self._adaptive_learning_rate(grad_norm)
                else:
                    lr = self.config['learning_rate']
                
                # 动量更新
                self.velocity = self.config['momentum']*self.velocity + lr*grad
                new_params = self.params - self.velocity
                
                # 参数投影
                new_params = self._project_params(new_params)
                delta = np.linalg.norm(new_params - self.params)
                cost_change = self.history['costs'][-1] - self.objective_func(new_params)
                
                # 记录状态
                self.history['params'].append(new_params.copy())
                self.history['costs'].append(self.objective_func(new_params))
                self.history['gradients'].append(grad.copy())
                self.history['violations'].append(self._compute_violation(new_params))
                self.history['deltas'].append(delta)
                
                # 收敛判断
                if self._check_convergence(delta, grad_norm, cost_change):
                    print(f"\n✅ 提前收敛于第 {iter+1} 次迭代")
                    self.params = new_params
                    return True
                
                self.params = new_params
                
                # 实时反馈
                if iter % 10 == 0:
                    print(f"迭代 {iter+1}: 成本={self.history['costs'][-1]:.4e} 梯度={grad_norm:.2e}")

            print("\n⚠️ 达到最大迭代次数")
            return False
        except Exception as e:
            print(f"\n❌ 优化异常: {str(e)}")
            return False

    def visualize(self, save_path=None):
        """可视化分析"""
        fig = plt.figure(figsize=(18, 12))
        
        # 参数轨迹
        ax1 = fig.add_subplot(2, 2, 1)
        params = np.array(self.history['params'])
        ax1.plot(params[:, 0], params[:, 1], 'b.-', alpha=0.6)
        ax1.scatter(params[0,0], params[0,1], c='r', label='起点')
        ax1.scatter(params[-1,0], params[-1,1], c='g', label='最优点')
        ax1.set_xlabel('发电效率')
        ax1.set_ylabel('余热回收')
        ax1.set_title('参数优化轨迹')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 成本收敛曲线
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.semilogy(self.history['costs'], 'r-')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('对数成本')
        ax2.set_title('成本收敛过程')
        ax2.grid(True, alpha=0.3)

        # 梯度变化
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.semilogy([np.linalg.norm(g) for g in self.history['gradients']], 'k-')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('梯度范数')
        ax3.set_title('梯度收敛过程')
        ax3.grid(True, alpha=0.3)

        # 参数变化
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(params[:,0], 'g-', label='发电效率')
        ax4.plot(params[:,1], 'm-', label='余热回收')
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('参数值')
        ax4.set_title('参数迭代过程')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def get_valid_input(prompt, value_type, default=None, validation=None):
    """获取并验证用户输入"""
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            value = value_type(user_input)
            if validation and not validation(value):
                raise ValueError
            return value
        except:
            print(f"输入无效，请重新输入（默认值：{default}）")

if __name__ == "__main__":
    print("冷热电联供系统优化平台 v3.1")
    print("="*50)
    
    # 交互式配置
    mode_choice = get_valid_input(
        "选择运行模式 (1:高电负荷 2:热电平衡 3:余热优先): ",
        str,
        default='2',
        validation=lambda x: x in ['1','2','3']
    )
    mode_map = {'1':'mode1', '2':'mode2', '3':'mode3'}
    mode = mode_map[mode_choice]
    
    # 初始参数配置
    if get_valid_input("是否自定义初始参数? (y/n): ", str, 'n').lower() == 'y':
        x1 = get_valid_input("发电效率初始值 (0-1): ", float, 0.0, lambda x: 0<=x<=1)
        x2 = get_valid_input("余热回收初始值 (0-1): ", float, 0.0, lambda x: 0<=x<=1)
        initial_params = [x1, x2]
    else:
        initial_params = [0.0, 0.0]
    
    # 高级配置
    max_iters = get_valid_input(
        "最大迭代次数 (默认1000): ",
        int,
        1000,
        lambda x: x>0
    )
    epsilon = get_valid_input(
        "收敛阈值 (默认1e-6): ",
        float,
        1e-6,
        lambda x: x>0
    )
    
    # 初始化优化器
    optimizer = EnhancedCCHPOptimizer(
        mode=mode,
        config={
            'max_iters': max_iters,
            'epsilon': epsilon
        }
    )
    optimizer.params = np.array(initial_params)
    
    # 执行优化
    start_time = time.time()
    success = optimizer.optimize()
    elapsed = time.time() - start_time
    
    # 显示结果
    print("\n优化结果报告")
    print("="*50)
    print(f"运行模式: \t{mode}")
    print(f"初始参数: \t{initial_params}")
    print(f"迭代次数: \t{len(optimizer.history['params'])-1}")
    print(f"计算耗时: \t{elapsed:.2f}秒")
    print(f"最终发电效率: \t{optimizer.params[0]:.6f}")
    print(f"最终余热回收: \t{optimizer.params[1]:.6f}")
    print(f"最终成本值: \t{optimizer.history['costs'][-1]:.4e}")
    
    # 可视化
    save_plot = get_valid_input("是否保存可视化图表? (y/n): ", str, 'n').lower()
    if save_plot == 'y':
        save_path = "optimization_plot.png"
        optimizer.visualize(save_path)
        print(f"图表已保存至 {os.path.abspath(save_path)}")
    else:
        optimizer.visualize()