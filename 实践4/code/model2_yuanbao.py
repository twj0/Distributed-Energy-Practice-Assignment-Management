import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CCHPOptimizer:
    """冷热电联供系统严格约束优化器
    
    功能特性：
    1. 双重约束机制：参数硬约束 + 目标函数软约束
    2. 实时参数监控与修正
    3. 自适应边界惩罚
    4. 可视化参数轨迹跟踪
    """
    
    PARAM_BOUNDS = {
        'power_ratio': (0.0, 1.0),    # 发电效率系数范围
        'heat_ratio': (0.0, 1.0)      # 余热回收系数范围
    }
    
    MAX_STEP_ALPHA = 0.5  # 最大单次调节步长（避免剧烈波动）

    def __init__(self, mode='mode1', epsilon=1e-4, max_iters=100):
        """
        初始化优化器
        :param mode: 运行模式 (mode1/mode2/mode3)
        :param epsilon: 收敛阈值
        :param max_iters: 最大迭代次数
        """
        self.mode = mode
        self.epsilon = epsilon
        self.max_iters = max_iters
        
        # 初始化运行参数
        self.current_params = np.array([0.0, 0.0])  # [发电效率, 余热回收]
        
        # 优化过程记录
        self.param_history = []
        self.cost_history = []
        self.violation_history = []
        
        # 配置目标函数
        self._init_objective_function()

    def _init_objective_function(self):
        """初始化带约束的目标函数"""
        if self.mode == 'mode1':
            self.objective = self._mode1_objective
        elif self.mode == 'mode2':
            self.objective = self._mode2_objective
        elif self.mode == 'mode3':
            self.objective = self._mode3_objective
        else:
            raise ValueError("无效的运行模式")

    def _apply_constraints(self, params):
        """应用参数硬约束并记录违规情况"""
        constrained = np.clip(params, 
                            [self.PARAM_BOUNDS['power_ratio'][0], 
                             self.PARAM_BOUNDS['heat_ratio'][0]],
                            [self.PARAM_BOUNDS['power_ratio'][1],
                             self.PARAM_BOUNDS['heat_ratio'][1]])
        
        # 记录违规程度
        violation = np.linalg.norm(params - constrained)
        self.violation_history.append(violation)
        
        return constrained

    def _boundary_penalty(self, params):
        """计算边界违规惩罚项"""
        penalty = 0.0
        
        # 发电效率系数违规
        if params[0] < self.PARAM_BOUNDS['power_ratio'][0]:
            penalty += 1e12 * (self.PARAM_BOUNDS['power_ratio'][0] - params[0])**2
        elif params[0] > self.PARAM_BOUNDS['power_ratio'][1]:
            penalty += 1e12 * (params[0] - self.PARAM_BOUNDS['power_ratio'][1])**2
        
        # 余热回收系数违规
        if params[1] < self.PARAM_BOUNDS['heat_ratio'][0]:
            penalty += 1e12 * (self.PARAM_BOUNDS['heat_ratio'][0] - params[1])**2
        elif params[1] > self.PARAM_BOUNDS['heat_ratio'][1]:
            penalty += 1e12 * (params[1] - self.PARAM_BOUNDS['heat_ratio'][1])**2
        
        return penalty

    def _mode1_objective(self, params):
        """模式1：高电负荷运行"""
        base_cost = 10*(params[0]-1)**2 + (params[1]+1)**4
        return base_cost + self._boundary_penalty(params)

    def _mode2_objective(self, params):
        """模式2：热电平衡运行"""
        base_cost = 100*(params[0]**2 - params[1])**2 + (params[0]-1)**2
        return base_cost + self._boundary_penalty(params)

    def _mode3_objective(self, params):
        """模式3：余热优先运行"""
        base_cost = 100*(params[0]**2 - 3*params[1])**2 + (params[0]-1)**2
        return base_cost + self._boundary_penalty(params)

    def _safeguard_gradient(self, params):
        """带安全保护的梯度计算"""
        try:
            grad = np.zeros(2)
            h = 1e-6
            
            for i in range(2):
                # 创建参数副本并施加约束
                params_plus = self._apply_constraints(params.copy())
                params_plus[i] += h
                
                params_minus = self._apply_constraints(params.copy())
                params_minus[i] -= h
                
                grad[i] = (self.objective(params_plus) - self.objective(params_minus)) / (2*h)
                
            return grad
        except Exception as e:
            print(f"梯度计算错误: {str(e)}")
            return np.zeros(2)

    def _line_search(self, direction):
        """安全步长搜索"""
        def _alpha_objective(alpha):
            # 限制步长范围
            alpha = min(alpha, self.MAX_STEP_ALPHA)
            new_params = self.current_params + alpha * direction
            constrained_params = self._apply_constraints(new_params)
            return self.objective(constrained_params)
        
        # 有界搜索
        result = minimize_scalar(_alpha_objective, bounds=(0, self.MAX_STEP_ALPHA), method='bounded')
        return result.x

    def optimize(self):
        """执行安全约束优化"""
        self.param_history = [self.current_params.copy()]
        self.cost_history = [self.objective(self.current_params)]
        
        for iter in range(self.max_iters):
            # 计算安全梯度
            grad = self._safeguard_gradient(self.current_params)
            direction = -grad
            
            # 步长搜索
            alpha = self._line_search(direction)
            
            # 参数更新
            new_params = self.current_params + alpha * direction
            
            # 施加硬约束
            new_params = self._apply_constraints(new_params)
            
            # 记录状态
            self.param_history.append(new_params.copy())
            self.cost_history.append(self.objective(new_params))
            
            # 检查收敛
            delta = np.linalg.norm(new_params - self.current_params)
            if delta < self.epsilon:
                break
                
            self.current_params = new_params
            
        return self._compile_results(iter+1)

    def _compile_results(self, iterations):
        """整理优化结果"""
        return {
            'optimal_power': self.current_params[0],
            'optimal_heat': self.current_params[1],
            'total_cost': self.cost_history[-1],
            'iterations': iterations,
            'param_history': np.array(self.param_history),
            'cost_history': self.cost_history,
            'violation_history': self.violation_history
        }

    def visualize(self, results):
        """可视化优化过程"""
        plt.figure(figsize=(14, 10))
        
        # 参数轨迹
        plt.subplot(2, 2, 1)
        plt.plot(results['param_history'][:,0], results['param_history'][:,1], 'bo-')
        plt.xlim(*self.PARAM_BOUNDS['power_ratio'])
        plt.ylim(*self.PARAM_BOUNDS['heat_ratio'])
        plt.xlabel('发电效率系数')
        plt.ylabel('余热回收系数')
        plt.title('参数优化轨迹')
        plt.grid(True)
        
        # 成本下降曲线
        plt.subplot(2, 2, 2)
        plt.semilogy(results['cost_history'], 'r-')
        plt.xlabel('迭代次数')
        plt.ylabel('综合成本（对数）')
        plt.title('成本收敛过程')
        plt.grid(True)
        
        # 参数变化过程
        plt.subplot(2, 2, 3)
        plt.plot(results['param_history'][:,0], 'g-', label='发电效率')
        plt.plot(results['param_history'][:,1], 'm-', label='余热回收')
        plt.ylim(-0.1, 1.1)
        plt.xlabel('迭代次数')
        plt.ylabel('参数值')
        plt.title('参数迭代过程')
        plt.legend()
        plt.grid(True)
        
        # 边界违规记录
        plt.subplot(2, 2, 4)
        plt.semilogy(results['violation_history'], 'k-')
        plt.xlabel('迭代次数')
        plt.ylabel('违规程度（对数）')
        plt.title('边界约束违规记录')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("冷热电联供系统优化平台")
    print("="*50)
    print("可用运行模式:")
    print("1. 高电负荷模式 (mode1)")
    print("2. 热电平衡模式 (mode2)")
    print("3. 余热优先模式 (mode3)")
    
    # 用户输入选择
    mode_choice = input("请选择运行模式 (1-3): ").strip()
    mode_map = {'1':'mode1', '2':'mode2', '3':'mode3'}
    
    if mode_choice not in mode_map:
        print("无效的选择，默认使用模式2")
        mode = 'mode2'
    else:
        mode = mode_map[mode_choice]
    
    # 用户自定义参数设置
    custom_init = input("是否自定义初始参数? (y/n): ").lower()
    if custom_init == 'y':
        x1 = float(input("请输入发电效率初始值(0-1): "))
        x2 = float(input("请输入余热回收初始值(0-1): "))
        initial_params = [max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))]
    else:
        initial_params = [0.0, 0.0]
    
    # 创建优化器实例
    optimizer = CCHPOptimizer(mode=mode, epsilon=1e-5, max_iters=100)
    optimizer.current_params = np.array(initial_params)  # 应用自定义初始值
    
    # 执行优化
    results = optimizer.optimize()
    
    # 显示优化报告（保持原样）
    # ...（原输出代码不变）...
    
    # 显示结果
    print("冷热电联供系统优化报告")
    print("="*50)
    print(f"运行模式: {optimizer.mode}")
    print(f"迭代次数: {results['iterations']}")
    print(f"最终发电效率: {results['optimal_power']:.6f}")
    print(f"最终余热回收: {results['optimal_heat']:.6f}")
    print(f"综合运行成本: {results['total_cost']:.6e}")
    print("参数边界合规性验证:")
    print(f"  发电效率: {optimizer.PARAM_BOUNDS['power_ratio'][0]} ≤ {results['optimal_power']:.6f} ≤ {optimizer.PARAM_BOUNDS['power_ratio'][1]}")
    print(f"  余热回收: {optimizer.PARAM_BOUNDS['heat_ratio'][0]} ≤ {results['optimal_heat']:.6f} ≤ {optimizer.PARAM_BOUNDS['heat_ratio'][1]}")
    
    # 可视化
    optimizer.visualize(results)