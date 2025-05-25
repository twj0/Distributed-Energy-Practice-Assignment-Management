import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time
import os

# 设置 matplotlib 支持中文显示，并正确显示负号
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CCHPOptimizer:
    """
    冷热电联供系统严格约束优化器
    
    功能特性：
    1. 双重约束机制：参数硬约束 + 目标函数软约束
    2. 实时参数监控与修正
    3. 自适应边界惩罚
    4. 可视化参数轨迹跟踪
    """
    
    # 定义参数的取值范围
    PARAM_BOUNDS = {
        'power_ratio': (0.0, 1.0),    # 发电效率系数范围
        'heat_ratio': (0.0, 1.0)      # 余热回收系数范围
    }
    
    # 最大单次调节步长，用于避免参数剧烈波动
    MAX_STEP_ALPHA = 0.5  

    def __init__(self, mode='mode1', epsilon=1e-4, max_iters=1145141919810):
        """
        初始化优化器
        :param mode: 运行模式 (mode1/mode2/mode3)
        :param epsilon: 收敛阈值，当参数变化小于该值时认为收敛
        :param max_iters: 最大迭代次数
        """
        self.mode = mode
        self.epsilon = epsilon
        self.max_iters = max_iters
        
        # 初始化运行参数，分别为发电效率和余热回收
        self.current_params = np.array([0.0, 0.0])  
        
        # 用于记录优化过程中的参数、成本和约束违规情况
        self.param_history = []
        self.cost_history = []
        self.violation_history = []
        
        # 配置当前模式对应的目标函数
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
        """应用参数硬约束并记录违规情况
        :param params: 待约束的参数数组
        :return: 满足约束条件的参数数组
        """
        # 使用 np.clip 将参数限制在取值范围内
        constrained = np.clip(params, 
                            [self.PARAM_BOUNDS['power_ratio'][0], 
                             self.PARAM_BOUNDS['heat_ratio'][0]],
                            [self.PARAM_BOUNDS['power_ratio'][1],
                             self.PARAM_BOUNDS['heat_ratio'][1]])
        
        # 计算参数在约束前后的差异，作为违规程度
        violation = np.linalg.norm(params - constrained)
        self.violation_history.append(violation)
        
        return constrained

    def _boundary_penalty(self, params):
        """计算边界违规惩罚项
        :param params: 待检查的参数数组
        :return: 边界违规惩罚值
        """
        penalty = 0.0
        
        # 检查发电效率系数是否违规
        if params[0] < self.PARAM_BOUNDS['power_ratio'][0]:
            penalty += 1e12 * (self.PARAM_BOUNDS['power_ratio'][0] - params[0])**2
        elif params[0] > self.PARAM_BOUNDS['power_ratio'][1]:
            penalty += 1e12 * (params[0] - self.PARAM_BOUNDS['power_ratio'][1])**2
        
        # 检查余热回收系数是否违规
        if params[1] < self.PARAM_BOUNDS['heat_ratio'][0]:
            penalty += 1e12 * (self.PARAM_BOUNDS['heat_ratio'][0] - params[1])**2
        elif params[1] > self.PARAM_BOUNDS['heat_ratio'][1]:
            penalty += 1e12 * (params[1] - self.PARAM_BOUNDS['heat_ratio'][1])**2
        
        return penalty

    def _mode1_objective(self, params):
        """模式1：高电负荷运行
        :param params: 参数数组
        :return: 包含边界惩罚的目标函数值
        """
        base_cost = 10*(params[0]-1)**2 + (params[1]+1)**4
        return base_cost + self._boundary_penalty(params)

    def _mode2_objective(self, params):
        """模式2：热电平衡运行
        :param params: 参数数组
        :return: 包含边界惩罚的目标函数值
        """
        base_cost = 100*(params[0]**2 - params[1])**2 + (params[0]-1)**2
        return base_cost + self._boundary_penalty(params)

    def _mode3_objective(self, params):
        """模式3：余热优先运行
        :param params: 参数数组
        :return: 包含边界惩罚的目标函数值
        """
        base_cost = 100*(params[0]**2 - 3*params[1])**2 + (params[0]-1)**2
        return base_cost + self._boundary_penalty(params)

    def _safeguard_gradient(self, params):
        """带安全保护的梯度计算
        :param params: 参数数组
        :return: 梯度数组
        """
        try:
            grad = np.zeros(2)
            h = 1e-6
            
            for i in range(2):
                # 创建参数副本并施加约束
                params_plus = self._apply_constraints(params.copy())
                params_plus[i] += h
                
                params_minus = self._apply_constraints(params.copy())
                params_minus[i] -= h
                
                # 使用中心差分法计算梯度
                grad[i] = (self.objective(params_plus) - self.objective(params_minus)) / (2*h)
                
            return grad
        except Exception as e:
            print(f"梯度计算错误: {str(e)}")
            return np.zeros(2)

    def _line_search(self, direction):
        """安全步长搜索
        :param direction: 搜索方向
        :return: 最优步长
        """
        def _alpha_objective(alpha):
            # 限制步长范围，不超过最大单次调节步长
            alpha = min(alpha, self.MAX_STEP_ALPHA)
            new_params = self.current_params + alpha * direction
            constrained_params = self._apply_constraints(new_params)
            return self.objective(constrained_params)
        
        # 在指定范围内进行一维搜索，找到最优步长
        result = minimize_scalar(_alpha_objective, bounds=(0, self.MAX_STEP_ALPHA), method='bounded')
        return result.x

    def optimize(self):
        """执行安全约束优化"""
        self.param_history = [self.current_params.copy()]
        self.cost_history = [self.objective(self.current_params)]
        self.violation_history = []  # 清空违规记录
        
        k = 0  # 独立迭代计数器
        while k < self.max_iters:
            # 梯度计算与参数更新
            grad = self._safeguard_gradient(self.current_params)
            direction = -grad
            alpha = self._line_search(direction)
            new_params = self.current_params + alpha * direction
            new_params = self._apply_constraints(new_params)
            
            # 记录当前状态
            self.param_history.append(new_params.copy())
            self.cost_history.append(self.objective(new_params))
            
            # 收敛判断
            delta = np.linalg.norm(new_params - self.current_params)
            if delta < self.epsilon:
                break
                
            self.current_params = new_params
            k += 1  # 更新迭代计数器
            
        return self._compile_results(k)  # 传递实际迭代次数

    def _compile_results(self, iterations):
        """整理优化结果
        :param iterations: 实际迭代次数
        :return: 包含优化结果的字典
        """
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
        """可视化优化过程
        :param results: 优化结果字典
        """
        plt.figure(figsize=(14, 10))
        iterations = results['iterations']  # 获取实际迭代次数
        # 绘制参数优化轨迹
        plt.subplot(2, 2, 1)
        plt.plot(results['param_history'][:,0], results['param_history'][:,1], 'bo-')
        plt.xlim(*self.PARAM_BOUNDS['power_ratio'])
        plt.ylim(*self.PARAM_BOUNDS['heat_ratio'])
        plt.xlabel('发电效率系数')
        plt.ylabel('余热回收系数')
        plt.title('参数优化轨迹')
        plt.grid(True)
        
        # 成本收敛曲线（横轴使用实际迭代次数）
        plt.subplot(2,2,2)
        plt.semilogy(range(len(results['cost_history'])), results['cost_history'], 'r-')
        plt.xlabel(f'迭代次数 (总计={iterations})')
        
        # 参数变化曲线
        plt.subplot(2,2,3)
        x_axis = range(len(results['param_history']))
        plt.plot(x_axis, results['param_history'][:,0], 'g-', label='发电效率')
        plt.plot(x_axis, results['param_history'][:,1], 'm-', label='余热回收')
        plt.xticks(np.linspace(0, len(x_axis), 5), np.linspace(0, iterations, 5, dtype=int))
        
        # 绘制边界约束违规记录
        plt.subplot(2, 2, 4)
        plt.semilogy(results['violation_history'], 'k-')
        plt.xlabel('迭代次数')
        plt.ylabel('违规程度（对数）')
        plt.title('边界约束违规记录')
        plt.grid(True)
        
        plt.tight_layout()
# 保存可视化结果
        # 检查 fig 文件夹是否存在，不存在则创建
        fig_dir = 'fig'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        # 生成图片文件名，包含运行模式
        filename = os.path.join(fig_dir, f'optimization_{self.mode}.png')
        plt.savefig(filename)
        plt.close()  # 关闭图片，避免显示


if __name__ == "__main__":
    print("冷热电联供系统优化平台")
    print("="*50)
    print("可用运行模式:")
    print("1. 高电负荷模式 (mode1)")
    print("2. 热电平衡模式 (mode2)")
    print("3. 余热优先模式 (mode3)")
    
    # 获取用户输入的运行模式选择
    mode_choice = input("请选择运行模式 (1-3): ").strip()
    mode_map = {'1':'mode1', '2':'mode2', '3':'mode3'}
    
    if mode_choice not in mode_map:
        print("无效的选择，默认使用模式2")
        mode = 'mode2'
    else:
        mode = mode_map[mode_choice]
    
    # 获取用户是否自定义初始参数的选择
    custom_init = input("是否自定义初始参数? (y/n): ").lower()
    if custom_init == 'y':
        x1 = float(input("请输入发电效率初始值(0-1): "))
        x2 = float(input("请输入余热回收初始值(0-1): "))
        initial_params = [max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))]
    else:
        initial_params = [0.0, 0.0]
    
    # 创建优化器实例
    optimizer = CCHPOptimizer(mode=mode, epsilon=1e-5, max_iters=1145141919810)
    optimizer.current_params = np.array(initial_params)  # 应用自定义初始值
    
    # 执行优化
    results = optimizer.optimize()
    
    # 显示优化报告
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
    
    # 可视化优化过程
    optimizer.visualize(results)
