import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

class CCHPThermalEconomicModel:
    """
    微型冷热电联供系统的热经济性模型
    
    基于实际工程应用，建立系统的热经济性数学模型，包括目标函数和约束条件。
    目标函数为系统年化总成本最小化，约束条件包括能量平衡、设备容量限制等。
    """
    
    def __init__(self):
        # 系统参数设置
        self.params = {
            # 经济参数
            'r': 0.08,                 # 贴现率
            'lifetime': 15,            # 系统使用寿命(年)
            'electricity_price': 0.8,  # 电价(元/kWh)
            'gas_price': 3.0,          # 天然气价格(元/m³)
            'maintenance_rate': 0.05,  # 维护成本率(占初始投资的百分比)
            
            # 初始投资成本系数(元/kW)
            'PGU_cost': 8000,          # 原动机成本
            'AC_cost': 1500,           # 吸收式制冷机成本
            'EC_cost': 1200,           # 电制冷机成本
            'boiler_cost': 400,        # 锅炉成本
            
            # 设备效率
            'PGU_electrical_efficiency': 0.33,  # 原动机电效率
            'PGU_thermal_efficiency': 0.45,     # 原动机热效率
            'AC_COP': 0.7,                      # 吸收式制冷机性能系数
            'EC_COP': 3.5,                      # 电制冷机性能系数
            'boiler_efficiency': 0.85,          # 锅炉效率
            
            # 能量转换系数
            'gas_heating_value': 10.0,  # 天然气热值(kWh/m³)
            
            # 负荷数据(kW) - 通常从历史数据获取或预测
            'max_electrical_load': 100,  # 最大电负荷
            'max_heating_load': 120,     # 最大热负荷
            'max_cooling_load': 150,     # 最大冷负荷
            
            # 系统运行时间
            'operation_hours': 8760,     # 年运行时间(小时)
        }
        
        # 负荷需求数据(可以从文件导入或预测生成)
        # 示例：使用典型日负荷曲线进行简化计算
        self.generate_example_load_profiles()
    
    def generate_example_load_profiles(self):
        """生成示例负荷曲线（24小时）"""
        hours = np.arange(24)
        
        # 电负荷曲线（典型商业建筑）
        electrical_load = 60 + 40 * np.sin((hours - 12) * np.pi / 12)
        electrical_load[hours < 8] = 40
        electrical_load[hours > 18] = 45
        
        # 热负荷曲线（冬季供暖）
        heating_load = 80 + 40 * np.sin((hours - 10) * np.pi / 12)
        heating_load[hours < 6] = 50
        heating_load[hours > 22] = 60
        
        # 冷负荷曲线（夏季制冷）
        cooling_load = 90 + 60 * np.sin((hours - 14) * np.pi / 12)
        cooling_load[hours < 10] = 40
        cooling_load[hours > 20] = 50
        
        # 创建DataFrame存储负荷数据
        self.load_profiles = pd.DataFrame({
            'Hour': hours,
            'ElectricalLoad': electrical_load,
            'HeatingLoad': heating_load,
            'CoolingLoad': cooling_load
        })
    
    def capital_recovery_factor(self):
        """计算资金回收系数"""
        r = self.params['r']
        n = self.params['lifetime']
        return r * (1 + r)**n / ((1 + r)**n - 1)
    
    def annual_investment_cost(self, capacities):
        """
        计算系统年化投资成本
        
        参数:
        - capacities: [PGU容量, AC容量, EC容量, 锅炉容量]
        """
        PGU_capacity, AC_capacity, EC_capacity, boiler_capacity = capacities
        
        # 各设备初始投资
        PGU_investment = PGU_capacity * self.params['PGU_cost']
        AC_investment = AC_capacity * self.params['AC_cost']
        EC_investment = EC_capacity * self.params['EC_cost']
        boiler_investment = boiler_capacity * self.params['boiler_cost']
        
        # 总初始投资
        total_investment = PGU_investment + AC_investment + EC_investment + boiler_investment
        
        # 年化投资成本
        CRF = self.capital_recovery_factor()
        annual_investment = total_investment * CRF
        
        return annual_investment
    
    def annual_operation_cost(self, capacities):
        """
        计算系统年化运行成本（燃料成本 + 维护成本 - 电网收益）
        
        参数:
        - capacities: [PGU容量, AC容量, EC容量, 锅炉容量]
        """
        PGU_capacity, AC_capacity, EC_capacity, boiler_capacity = capacities
        
        total_gas_consumption = 0
        total_electricity_purchase = 0
        total_electricity_sell = 0
        
        # 遍历24小时负荷数据计算年化成本(简化计算)
        for _, row in self.load_profiles.iterrows():
            electrical_load = row['ElectricalLoad']
            heating_load = row['HeatingLoad']
            cooling_load = row['CoolingLoad']
            
            # 设备出力计算
            PGU_output = min(PGU_capacity, electrical_load)
            PGU_heat = PGU_output * self.params['PGU_thermal_efficiency'] / self.params['PGU_electrical_efficiency']
            
            # 剩余电量计算
            remaining_electricity = electrical_load - PGU_output
            
            # 电网购电/售电计算
            if remaining_electricity > 0:
                total_electricity_purchase += remaining_electricity
            else:
                total_electricity_sell -= remaining_electricity
            
            # 热负荷分配
            remaining_heat = heating_load - PGU_heat
            if remaining_heat > 0:
                boiler_heat = min(boiler_capacity, remaining_heat)
                boiler_gas = boiler_heat / self.params['boiler_efficiency']
                total_gas_consumption += boiler_gas
            
            # 冷负荷分配
            AC_cooling = min(AC_capacity, cooling_load)
            remaining_cooling = cooling_load - AC_cooling
            EC_cooling = min(EC_capacity, remaining_cooling)
            
            # 制冷电力消耗
            EC_electricity = EC_cooling / self.params['EC_COP']
            total_electricity_purchase += EC_electricity
            
            # PGU燃气消耗
            PGU_gas = PGU_output / self.params['PGU_electrical_efficiency']
            total_gas_consumption += PGU_gas
        
        # 年化运行成本(简化为24小时 * 365天)
        annual_factor = 365
        gas_cost = total_gas_consumption * self.params['gas_price'] * annual_factor / self.params['gas_heating_value']
        electricity_cost = total_electricity_purchase * self.params['electricity_price'] * annual_factor
        electricity_income = total_electricity_sell * (self.params['electricity_price'] * 0.8) * annual_factor  # 假设售电价格为购电价格的80%
        
        # 维护成本
        maintenance_cost = self.annual_investment_cost(capacities) * self.params['maintenance_rate']
        
        return gas_cost + electricity_cost + maintenance_cost - electricity_income
    
    def objective_function(self, capacities):
        """
        目标函数：系统年化总成本最小化
        
        参数:
        - capacities: [PGU容量, AC容量, EC容量, 锅炉容量]
        """
        annual_investment = self.annual_investment_cost(capacities)
        annual_operation = self.annual_operation_cost(capacities)
        
        return annual_investment + annual_operation
    
    def constraint_energy_balance(self, capacities):
        """
        约束条件：能量平衡约束
        确保系统能够满足所有负荷需求
        
        返回所有时段能量平衡满足的程度(非负值表示满足)
        """
        PGU_capacity, AC_capacity, EC_capacity, boiler_capacity = capacities
        
        # 能量平衡余量
        electrical_margins = []
        heating_margins = []
        cooling_margins = []
        
        # 遍历负荷数据
        for _, row in self.load_profiles.iterrows():
            electrical_load = row['ElectricalLoad']
            heating_load = row['HeatingLoad']
            cooling_load = row['CoolingLoad']
            
            # 电负荷平衡
            max_electrical_output = PGU_capacity + self.params['max_electrical_load']  # 假设可以从电网购买最大电负荷
            electrical_margin = max_electrical_output - electrical_load
            electrical_margins.append(electrical_margin)
            
            # 热负荷平衡
            PGU_heat = PGU_capacity * self.params['PGU_thermal_efficiency'] / self.params['PGU_electrical_efficiency']
            max_heat_output = PGU_heat + boiler_capacity
            heating_margin = max_heat_output - heating_load
            heating_margins.append(heating_margin)
            
            # 冷负荷平衡
            max_cooling_output = AC_capacity + EC_capacity
            cooling_margin = max_cooling_output - cooling_load
            cooling_margins.append(cooling_margin)
        
        # 返回最小余量(最不满足的情况)
        return min(min(electrical_margins), min(heating_margins), min(cooling_margins))
    
    def optimize(self):
        """执行系统优化"""
        # 初始猜测值
        initial_capacities = [
            self.params['max_electrical_load'] * 0.7,  # PGU容量初始值
            self.params['max_cooling_load'] * 0.5,     # AC容量初始值
            self.params['max_cooling_load'] * 0.5,     # EC容量初始值
            self.params['max_heating_load'] * 0.5      # 锅炉容量初始值
        ]
        
        # 容量约束(下限和上限)
        bounds = [
            (10, self.params['max_electrical_load']),   # PGU容量范围
            (10, self.params['max_cooling_load']),      # AC容量范围
            (10, self.params['max_cooling_load']),      # EC容量范围
            (10, self.params['max_heating_load'])       # 锅炉容量范围
        ]
        
        # 能量平衡约束
        constraints = [
            {'type': 'ineq', 'fun': self.constraint_energy_balance}
        ]
        
        # 执行优化
        result = minimize(
            self.objective_function,
            initial_capacities,
            method='SLSQP',  # 序列最小二乘规划法
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000}
        )
        
        # 输出优化结果
        if result.success:
            optimized_capacities = result.x
            annual_cost = result.fun
            
            print("\n优化结果:")
            print(f"PGU容量: {optimized_capacities[0]:.2f} kW")
            print(f"吸收式制冷机容量: {optimized_capacities[1]:.2f} kW")
            print(f"电制冷机容量: {optimized_capacities[2]:.2f} kW")
            print(f"锅炉容量: {optimized_capacities[3]:.2f} kW")
            print(f"系统年化总成本: {annual_cost:.2f} 元")
            
            # 计算详细成本
            investment_cost = self.annual_investment_cost(optimized_capacities)
            operation_cost = self.annual_operation_cost(optimized_capacities)
            
            print("\n成本细分:")
            print(f"年化投资成本: {investment_cost:.2f} 元")
            print(f"年化运行成本: {operation_cost:.2f} 元")
            
            # 分析能量平衡
            energy_balance = self.constraint_energy_balance(optimized_capacities)
            print(f"\n能量平衡满足度: {energy_balance:.2f} kW (正值表示满足所有需求)")
            
            return {
                'success': True,
                'capacities': optimized_capacities,
                'annual_cost': annual_cost,
                'investment_cost': investment_cost,
                'operation_cost': operation_cost,
                'energy_balance': energy_balance
            }
        else:
            print("\n优化失败!")
            print(f"原因: {result.message}")
            return {'success': False, 'message': result.message}
    
    def sensitivity_analysis(self, parameter_name, range_values):
        """
        对指定参数进行敏感性分析
        
        参数:
        - parameter_name: 参数名称
        - range_values: 参数取值范围
        """
        # 存储不同参数值下的优化结果
        results = []
        original_value = self.params[parameter_name]
        
        for value in range_values:
            # 修改参数值
            self.params[parameter_name] = value
            
            # 执行优化
            result = self.optimize()
            
            if result['success']:
                results.append({
                    'parameter_value': value,
                    'annual_cost': result['annual_cost'],
                    'capacities': result['capacities']
                })
        
        # 恢复原始参数值
        self.params[parameter_name] = original_value
        
        # 绘制敏感性分析结果
        plt.figure(figsize=(10, 6))
        
        parameter_values = [r['parameter_value'] for r in results]
        annual_costs = [r['annual_cost'] for r in results]
        
        plt.plot(parameter_values, annual_costs, 'o-', linewidth=2)
        plt.title(f'参数 {parameter_name} 敏感性分析')
        plt.xlabel(f'{parameter_name} 取值')
        plt.ylabel('系统年化总成本 (元)')
        plt.grid(True)
        
        return results, plt

def main():
    print("微型冷热电联供系统热经济性优化模型")
    print("="*50)
    
    # 创建模型
    model = CCHPThermalEconomicModel()
    
    # 执行优化
    print("\n执行系统优化...")
    result = model.optimize()
    
    if result['success']:
        print("\n是否执行敏感性分析? (y/n)")
        choice = input()
        
        if choice.lower() == 'y':
            print("\n选择要分析的参数:")
            print("1. 电价")
            print("2. 天然气价格")
            print("3. 原动机电效率")
            
            param_choice = input("请输入选择 (1-3): ")
            
            param_map = {
                '1': ('electricity_price', np.linspace(0.6, 1.0, 5)),
                '2': ('gas_price', np.linspace(2.0, 4.0, 5)),
                '3': ('PGU_electrical_efficiency', np.linspace(0.25, 0.4, 5))
            }
            
            if param_choice in param_map:
                param_name, param_range = param_map[param_choice]
                print(f"\n执行 {param_name} 的敏感性分析...")
                _, plt = model.sensitivity_analysis(param_name, param_range)
                plt.show()
            else:
                print("无效的选择!")
    
    print("\n程序结束")

if __name__ == "__main__":
    main()