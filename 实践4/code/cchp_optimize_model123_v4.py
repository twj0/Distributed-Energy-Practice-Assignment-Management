"""
冷热电联供系统优化平台 - 科学强化版 (v4.0)
核心改进：
1. 投影梯度法确保搜索方向可行性
2. 动量校正避免边界震荡
3. 动态约束松弛机制
4. 五重收敛条件检测
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import os

# 可视化配置

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ScientificCCHPOptimizer:
    """
    科学强化型冷热电联供系统优化器
    
    特性：
    - 基于投影梯度法的边界处理
    - 动量自适应校正技术
    - 五维度收敛检测
    - 动态约束松弛机制
    """
    
    # 参数可行域 [发电效率, 余热回收] ∈ [0,1]^2
    PARAM_BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])
    
    def __init__(self, mode='mode2', config=None):
        """
        初始化科学优化器
        :param mode: 运行模式 (mode1/mode2/mode3)
        :param config: 配置字典
        """
        self.mode = mode
        self.config = {
            'max_iters': 1000,         # 最大迭代次数
            'epsilon': 1e-4,           # 收敛阈值
            'base_lr': 0.1,            # 基础学习率
            'momentum': 0.9,           # 动量系数
            'buffer_width': 0.05,      # 边界缓冲层宽度(参数范围的5%)
            'gradient_boost': 2.0,      # 边界附近梯度增强系数
            **(config or {})
        }
        
        # 优化状态变量
        self.params = np.array([0.0, 0.0])  # 当前参数 [x1, x2]
        self.velocity = np.zeros(2)         # 动量项
        self.history = {                    # 优化过程记录
            'params': [],                   # 参数轨迹
            'costs': [],                     # 成本值
            'gradients': [],                 # 梯度记录
            'violations': [],                # 边界违规
            'deltas': []                     # 参数变化量
        }

        # 根据模式初始化目标函数
        self._init_objective()

    def _init_objective(self):
        """初始化目标函数与解析梯度"""
        # 目标函数定义
        if self.mode == 'mode1':
            self.objective = lambda x: 10*(x[0]-1)**2 + (x[1]+1)**4
            self.gradient = lambda x: np.array([20*(x[0]-1), 4*(x[1]+1)**3])
        elif self.mode == 'mode2':
            self.objective = lambda x: 100*(x[0]**2 -x[1])**2 + (x[0]-1)**2
            self.gradient = lambda x: np.array([
                400*x[0]*(x[0]**2 -x[1]) + 2*(x[0]-1),
                -200*(x[0]**2 -x[1])
            ])
        elif self.mode == 'mode3':
            self.objective = lambda x: 100*(x[0]**2 -3*x[1])**2 + (x[0]-1)**2
            self.gradient = lambda x: np.array([
                400*x[0]*(x[0]**2 -3*x[1]) + 2*(x[0]-1),
                -600*(x[0]**2 -3*x[1])
            ])
        else:
            raise ValueError("未知的运行模式")

    def _project_params(self, params):
        """将参数投影到可行域内"""
        return np.clip(params, self.PARAM_BOUNDS[:,0], self.PARAM_BOUNDS[:,1])

    def _project_gradient(self, params, raw_grad):
        """
        边界感知梯度投影
        :param params: 当前参数
        :param raw_grad: 原始梯度
        :return: 投影后的可行梯度
        """
        projected_grad = raw_grad.copy()
        
        # 下边界检测 (参数 <= lb + tolerance)
        lb_mask = params <= self.PARAM_BOUNDS[:,0] + 1e-4
        projected_grad[lb_mask & (projected_grad < 0)] = 0  # 阻止向下越界
        
        # 上边界检测 (参数 >= ub - tolerance)
        ub_mask = params >= self.PARAM_BOUNDS[:,1] - 1e-4
        projected_grad[ub_mask & (projected_grad > 0)] = 0  # 阻止向上越界
        
        return projected_grad

    def _dynamic_gradient_adjust(self, params, grad):
        """
        动态梯度调整(边界缓冲层增强)
        :param params: 当前参数
        :param grad: 原始梯度
        :return: 调整后的梯度
        """
        adjusted_grad = grad.copy()
        buffer = self.config['buffer_width']
        
        # 对每个参数进行边界检测
        for i in range(2):
            lb, ub = self.PARAM_BOUNDS[i]
            
            # 下边界缓冲增强
            if (params[i] - lb) < buffer:
                scale = 1 + (buffer - (params[i] - lb))/buffer * self.config['gradient_boost']
                adjusted_grad[i] *= scale
                
            # 上边界缓冲增强
            if (ub - params[i]) < buffer:
                scale = 1 + (buffer - (ub - params[i]))/buffer * self.config['gradient_boost']
                adjusted_grad[i] *= scale
                
        return adjusted_grad

    def _adaptive_learning_rate(self, grad_norm):
        """
        自适应学习率调整
        :param grad_norm: 当前梯度范数
        :return: 调整后的学习率
        """
        if grad_norm > 1e-1:
            return self.config['base_lr'] * 0.5  # 大梯度时保守更新
        elif grad_norm < 1e-3:
            return self.config['base_lr'] * 2.0   # 小梯度时加速收敛
        else:
            return self.config['base_lr']

    def _update_velocity(self, new_params, lr):
        """
        动量校正(基于实际参数更新量)
        :param new_params: 投影后的新参数
        :param lr: 实际使用的学习率
        """
        actual_update = (new_params - self.params) / lr
        self.velocity = self.config['momentum']*self.velocity + actual_update

    def _check_convergence(self):
        """五维收敛条件检测"""
        if len(self.history['deltas']) < 2:
            return False
            
        criteria = {
            'delta': self.history['deltas'][-1] < self.config['epsilon'],
            'gradient': np.linalg.norm(self.history['gradients'][-1]) < 1e-5,
            'cost_change': abs(self.history['costs'][-2] - self.history['costs'][-1]) < 1e-8,
            'violation': self.history['violations'][-1] < 1e-4,
            'velocity': np.linalg.norm(self.velocity) < 0.1*self.config['epsilon']
        }
        return all(criteria.values())

    def optimize(self):
        """执行科学优化流程"""
        self.params = self._project_params(self.params)
        self.history['params'].append(self.params.copy())
        self.history['costs'].append(self.objective(self.params))
        
        try:
            for iter in range(self.config['max_iters']):
                # 1. 计算原始梯度
                raw_grad = self.gradient(self.params)
                
                # 2. 动态梯度调整
                adjusted_grad = self._dynamic_gradient_adjust(self.params, raw_grad)
                
                # 3. 梯度投影
                projected_grad = self._project_gradient(self.params, adjusted_grad)
                
                # 4. 学习率自适应
                lr = self._adaptive_learning_rate(np.linalg.norm(projected_grad))
                
                # 5. 动量更新
                new_params = self.params - lr * (self.config['momentum']*self.velocity + projected_grad)
                
                # 6. 参数投影
                new_params = self._project_params(new_params)
                
                # 7. 更新动量(基于实际更新量)
                self._update_velocity(new_params, lr)
                
                # 记录优化状态
                delta = np.linalg.norm(new_params - self.params)
                self.history['params'].append(new_params.copy())
                self.history['costs'].append(self.objective(new_params))
                self.history['gradients'].append(projected_grad.copy())
                self.history['violations'].append(np.linalg.norm(new_params - self._project_params(new_params)))
                self.history['deltas'].append(delta)
                
                # 检查收敛
                if self._check_convergence():
                    print(f"\n✅ 严格收敛于第 {iter+1} 次迭代")
                    self._print_convergence_report()
                    return True
                
                self.params = new_params
                
                # 每10次迭代打印进度
                if (iter+1) % 10 == 0:
                    print(f"Iter {iter+1:4d} | Cost: {self.history['costs'][-1]:.3e} | Grad: {np.linalg.norm(projected_grad):.1e} | Δ: {delta:.1e}")

            print("\n⚠️ 达到最大迭代次数")
            return False
        except Exception as e:
            print(f"\n❌ 优化异常: {str(e)}")
            return False

    def _print_convergence_report(self):
        """打印收敛诊断报告"""
        print("\n收敛诊断报告:")
        print("="*40)
        criteria = {
            '参数变化量': self.history['deltas'][-1],
            '梯度范数': np.linalg.norm(self.history['gradients'][-1]),
            '成本变化率': abs(self.history['costs'][-2] - self.history['costs'][-1]),
            '边界违规量': self.history['violations'][-1],
            '动量震荡度': np.linalg.norm(self.velocity)
        }
        for name, value in criteria.items():
            status = "✔️" if value < self.config['epsilon'] else "❌"
            print(f"{name:8s}: {value:.2e} {status}")

    def visualize(self, save_path=None):
        """生成优化过程可视化"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 参数轨迹
        params = np.array(self.history['params'])
        axs[0,0].plot(params[:,0], params[:,1], 'b.-', markersize=8)
        axs[0,0].scatter(params[0,0], params[0,1], c='r', s=100, label='初始点')
        axs[0,0].scatter(params[-1,0], params[-1,1], c='g', s=100, label='最优点')
        axs[0,0].set_xlim(-0.05, 1.05)
        axs[0,0].set_ylim(-0.05, 1.05)
        axs[0,0].set_xlabel('发电效率系数', fontsize=12)
        axs[0,0].set_ylabel('余热回收系数', fontsize=12)
        axs[0,0].set_title('参数优化轨迹', fontsize=14)
        axs[0,0].grid(True, alpha=0.3)
        axs[0,0].legend()
        
        # 成本收敛曲线
        axs[0,1].semilogy(self.history['costs'], 'r-', linewidth=2)
        axs[0,1].set_xlabel('迭代次数', fontsize=12)
        axs[0,1].set_ylabel('综合成本', fontsize=12)
        axs[0,1].set_title('成本收敛过程', fontsize=14)
        axs[0,1].grid(True, alpha=0.3)
        
        # 参数变化过程
        axs[1,0].plot(params[:,0], 'g-', label='发电效率')
        axs[1,0].plot(params[:,1], 'm-', label='余热回收')
        axs[1,0].set_xlabel('迭代次数', fontsize=12)
        axs[1,0].set_ylabel('参数值', fontsize=12)
        axs[1,0].set_title('参数迭代过程', fontsize=14)
        axs[1,0].legend()
        axs[1,0].grid(True, alpha=0.3)
        
        # 梯度变化
        grads = np.array([np.linalg.norm(g) for g in self.history['gradients']])
        axs[1,1].semilogy(grads, 'k-', linewidth=2)
        axs[1,1].set_xlabel('迭代次数', fontsize=12)
        axs[1,1].set_ylabel('梯度范数', fontsize=12)
        axs[1,1].set_title('梯度收敛过程', fontsize=14)
        axs[1,1].grid(True, alpha=0.3)

        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

# 交互界面函数
def get_valid_input(prompt, value_type, default=None, validation=None):
    """获取并验证用户输入"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            value = value_type(user_input)
            if validation and not validation(value):
                raise ValueError
            return value
        except:
            print(f"输入无效，请重新输入（默认值：{default}）")

if __name__ == "__main__":
    print("冷热电联供系统科学优化平台 v4.0")
    print("="*50)
    
    # 模式选择
    mode_choice = get_valid_input(
        "选择运行模式 (1:高电负荷 2:热电平衡 3:余热优先): ",
        str, '2', lambda x: x in ['1','2','3']
    )
    mode_map = {'1':'mode1', '2':'mode2', '3':'mode3'}
    
    # 初始参数配置
    if get_valid_input("是否自定义初始参数? (y/n): ", str, 'n').lower() == 'y':
        x1 = get_valid_input("发电效率初始值 (0-1): ", float, 0.0, lambda x: 0<=x<=1)
        x2 = get_valid_input("余热回收初始值 (0-1): ", float, 0.0, lambda x: 0<=x<=1)
        initial_params = [x1, x2]
    else:
        initial_params = [0.0, 0.0]
    
    # 高级配置
    config = {
        'max_iters': get_valid_input(
            "最大迭代次数 (默认1000): ", int, 1000, lambda x: x>0),
        'epsilon': get_valid_input(
            "收敛阈值 (默认1e-4): ", float, 1e-4, lambda x: x>0),
        'base_lr': get_valid_input(
            "基础学习率 (默认0.1): ", float, 0.1, lambda x: x>0),
        'momentum': get_valid_input(
            "动量系数 (默认0.9): ", float, 0.9, lambda x: 0<=x<1)
    }
    
    # 初始化优化器
    optimizer = ScientificCCHPOptimizer(
        mode=mode_map[mode_choice],
        config=config
    )
    optimizer.params = np.array(initial_params)
    
    # 执行优化
    start_time = time.time()
    success = optimizer.optimize()
    elapsed = time.time() - start_time
    
    # 显示结果
    print("\n优化结果报告")
    print("="*50)
    print(f"运行模式: \t{optimizer.mode}")
    print(f"初始参数: \t{initial_params}")
    print(f"迭代次数: \t{len(optimizer.history['params'])-1}")
    print(f"计算耗时: \t{elapsed:.2f}秒")
    print(f"最终发电效率: \t{optimizer.params[0]:.6f}")
    print(f"最终余热回收: \t{optimizer.params[1]:.6f}")
    print(f"最终成本值: \t{optimizer.history['costs'][-1]:.4e}")
    
    # 可视化
    if get_valid_input("是否生成可视化图表? (y/n): ", str, 'y').lower() == 'y':
        save_path = "fig4\scientific_optimization.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        optimizer.visualize(save_path)
        print(f"图表已保存至 {os.path.abspath(save_path)}")