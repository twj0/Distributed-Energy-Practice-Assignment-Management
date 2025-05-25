"""
冷热电联供系统优化平台 - 增强收敛版
版本：3.0
改进重点：
1. 投影梯度法避免边界震荡
2. 动态动量自适应调整
3. 混合步长搜索策略
4. 三重收敛条件判断
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
import time

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedCCHPOptimizer:
    """
    增强型冷热电联供系统优化器
    新增功能：
    - 投影梯度法
    - 自适应动量控制
    - 混合步长搜索
    - 三重收敛条件
    """
    
    # 系统参数约束
    PARAM_BOUNDS = {
        'power_ratio': (0.0, 1.0),
        'heat_ratio': (0.0, 1.0)
    }
    
    # 优化器超参数
    DEFAULT_EPSILON = 1e-4
    DEFAULT_MAX_ITERS = 200
    MAX_STEP_ALPHA = 0.5
    GRAD_EPS = 1e-6
    GRAD_CLIP = 1e3
    
    def __init__(self, mode='mode2', epsilon=None, max_iters=None):
        self.mode = mode.lower()
        self.epsilon = epsilon or self.DEFAULT_EPSILON
        self.max_iters = max_iters or self.DEFAULT_MAX_ITERS
        
        # 优化状态
        self.current_params = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)  # 动量项
        
        # 过程记录
        self.param_history = []
        self.cost_history = []
        self.grad_history = []
        
        # 配置目标函数
        self._init_objective()

    def _init_objective(self):
        """初始化目标函数"""
        func_map = {
            'mode1': lambda x: 10*(x[0]-1)**2 + (x[1]+1)**4,
            'mode2': lambda x: 100*(x[0]**2 -x[1])**2 + (x[0]-1)**2,
            'mode3': lambda x: 100*(x[0]**2 -3*x[1])**2 + (x[0]-1)**2
        }
        self.objective = lambda x: func_map[self.mode](x) + self._boundary_penalty(x)

    def _boundary_penalty(self, params):
        """边界惩罚函数"""
        penalty = 0.0
        if params[0] < 0 or params[0] > 1:
            penalty += 1e12 * (min(abs(params[0]-0), abs(params[0]-1)))**2
        if params[1] < 0 or params[1] > 1:
            penalty += 1e12 * (min(abs(params[1]-0), abs(params[1]-1)))**2
        return penalty

    def _project_params(self, params):
        """参数投影到可行域"""
        return np.clip(params, [0, 0], [1, 1])

    def _compute_gradient(self):
        """带投影保护的梯度计算"""
        grad = np.zeros(2)
        current = self.current_params.copy()
        
        for i in range(2):
            # 正向扰动
            forward = current.copy()
            forward[i] += self.GRAD_EPS
            forward = self._project_params(forward)
            
            # 负向扰动
            backward = current.copy()
            backward[i] -= self.GRAD_EPS
            backward = self._project_params(backward)
            
            grad[i] = (self.objective(forward) - self.objective(backward)) / (2*self.GRAD_EPS)
        
        return np.clip(grad, -self.GRAD_CLIP, self.GRAD_CLIP)

    def _adaptive_momentum(self, grad_norm):
        """动态动量调整"""
        base_momentum = 0.9
        if grad_norm > 1e-2:
            return max(0.5, base_momentum - 0.3*(min(grad_norm/0.1, 1)))
        return base_momentum

    def _hybrid_line_search(self, direction):
        """混合步长搜索策略"""
        def cost_func(a):
            trial = self._project_params(self.current_params + a*direction)
            return self.objective(trial)
        
        # 第一阶段：精细搜索
        phase1 = minimize_scalar(cost_func, bounds=(0, 0.1), method='bounded')
        # 第二阶段：全局搜索
        phase2 = minimize_scalar(cost_func, bounds=(0.1, self.MAX_STEP_ALPHA), method='bounded')
        
        # 选择最优步长
        return min(phase1.x, phase2.x, key=lambda a: cost_func(a))

    def optimize(self):
        """执行增强优化流程"""
        self.param_history = [self.current_params.copy()]
        self.cost_history = [self.objective(self.current_params)]
        converged = False
        
        for k in range(self.max_iters):
            # 1. 梯度计算
            grad = self._compute_gradient()
            self.grad_history.append(grad.copy())
            
            # 2. 动量更新
            grad_norm = np.linalg.norm(grad)
            momentum = self._adaptive_momentum(grad_norm)
            self.velocity = momentum*self.velocity + (1-momentum)*grad
            
            # 3. 方向确定
            direction = -self.velocity
            
            # 4. 步长搜索
            alpha = self._hybrid_line_search(direction)
            
            # 5. 参数更新
            new_params = self._project_params(self.current_params + alpha*direction)
            
            # 6. 收敛判断
            delta_param = np.linalg.norm(new_params - self.current_params)
            delta_cost = abs(self.cost_history[-1] - self.objective(new_params))
            
            if delta_param < self.epsilon and delta_cost < self.epsilon*1e-3:
                converged = True
                self.current_params = new_params
                self.param_history.append(new_params)
                self.cost_history.append(self.objective(new_params))
                break
            
            # 记录状态
            self.current_params = new_params
            self.param_history.append(new_params)
            self.cost_history.append(self.objective(new_params))
        
        return {
            'converged': converged,
            'iterations': k+1,
            'params': self.current_params,
            'final_cost': self.cost_history[-1],
            'history': {
                'params': np.array(self.param_history),
                'costs': self.cost_history,
                'gradients': np.array(self.grad_history)
            }
        }

    def visualize(self, results, save_path=None):
        """增强可视化分析"""
        plt.figure(figsize=(16, 12))
        
        # 参数空间轨迹
        plt.subplot(2,2,1)
        hist = results['history']['params']
        plt.plot(hist[:,0], hist[:,1], 'b.-', alpha=0.6)
        plt.scatter(hist[0,0], hist[0,1], c='r', label='起点')
        plt.scatter(hist[-1,0], hist[-1,1], c='g', label='终点')
        plt.xlabel('发电效率')
        plt.ylabel('余热回收')
        plt.title('参数优化轨迹')
        plt.grid(True, alpha=0.3)
        
        # 成本收敛曲线
        plt.subplot(2,2,2)
        plt.semilogy(results['history']['costs'], 'r-')
        plt.xlabel('迭代次数')
        plt.ylabel('对数成本')
        plt.title('成本收敛过程')
        plt.grid(True, alpha=0.3)
        
        # 梯度变化
        plt.subplot(2,2,3)
        grads = np.linalg.norm(results['history']['gradients'], axis=1)
        plt.semilogy(grads, 'k-')
        plt.xlabel('迭代次数')
        plt.ylabel('梯度模长')
        plt.title('梯度变化过程')
        plt.grid(True, alpha=0.3)
        
        # 参数变化
        plt.subplot(2,2,4)
        plt.plot(hist[:,0], 'g-', label='发电效率')
        plt.plot(hist[:,1], 'm-', label='余热回收')
        plt.xlabel('迭代次数')
        plt.ylabel('参数值')
        plt.title('参数迭代过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

if __name__ == "__main__":
    # 初始化优化器（默认模式2）
    optimizer = EnhancedCCHPOptimizer(mode='mode2', epsilon=1e-5)
    
    # 执行优化
    start = time.time()
    results = optimizer.optimize()
    elapsed = time.time() - start
    
    # 输出结果
    print(f"\n优化结果 ({'收敛' if results['converged'] else '未收敛'})")
    print("="*50)
    print(f"迭代次数: {results['iterations']}")
    print(f"最终参数: 发电效率={results['params'][0]:.6f}, 余热回收={results['params'][1]:.6f}")
    print(f"最终成本: {results['final_cost']:.4e}")
    print(f"计算耗时: {elapsed:.2f}秒")
    
    # 可视化
    optimizer.visualize(results, save_path='optimization_result.png')