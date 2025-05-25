from pyomo.environ import *

# 创建模型
model = ConcreteModel()

# 参数设置
model.eta_gt = Param(initialize=0.3)     # 燃气轮机发电效率
model.eta_hr = Param(initialize=0.8)     # 余热回收效率
model.eta_boiler = Param(initialize=0.85) # 辅助锅炉效率
model.COP_abs = Param(initialize=1.2)    # 吸收式制冷COP
model.COP_elec = Param(initialize=3.0)   # 电制冷COP
model.gas_price = Param(initialize=0.5)  # 燃气价格（元/kWh）
model.elec_price = Param(initialize=1.0) # 外购电价（元/kWh）
model.E_load = Param(initialize=100)     # 电负荷（kW）
model.H_load = Param(initialize=80)      # 热负荷（kW）
model.C_load = Param(initialize=60)      # 冷负荷（kW）
model.P_gt_max = Param(initialize=150)   # 燃气轮机最大发电量（kW）
model.E_grid_max = Param(initialize=200)  # 最大外购电量（kW）
model.Q_aux_max = Param(initialize=100)  # 辅助锅炉最大供热量（kW）
model.C_elec_max = Param(initialize=50)  # 电制冷机最大制冷量（kW）

# 决策变量
model.P_gt = Var(domain=NonNegativeReals)       # 燃气轮机发电量
model.E_purchased = Var(domain=NonNegativeReals) # 外购电量
model.Q_aux = Var(domain=NonNegativeReals)      # 辅助锅炉供热量
model.Q_hr_heat = Var(domain=NonNegativeReals)  # 余热供热
model.Q_hr_cool = Var(domain=NonNegativeReals)  # 余热制冷
model.C_elec = Var(domain=NonNegativeReals)     # 电制冷量

# 目标函数：总成本最小
def total_cost_rule(model):
    fuel_gt = model.P_gt / model.eta_gt
    cost_gt = fuel_gt * model.gas_price
    cost_purchased = model.E_purchased * model.elec_price
    fuel_aux = model.Q_aux / model.eta_boiler
    cost_aux = fuel_aux * model.gas_price
    cost_cooling = (model.C_elec / model.COP_elec) * model.elec_price
    return cost_gt + cost_purchased + cost_aux + cost_cooling
model.total_cost = Objective(rule=total_cost_rule, sense=minimize)

# 约束条件
def power_balance(model):
    return model.P_gt + model.E_purchased == model.E_load + (model.C_elec / model.COP_elec)
model.con_power = Constraint(rule=power_balance)

def heat_recovery_total(model):
    Q_hr_total = model.P_gt * (1 - model.eta_gt) * model.eta_hr
    return model.Q_hr_heat + model.Q_hr_cool <= Q_hr_total
model.con_hr_total = Constraint(rule=heat_recovery_total)

def heat_balance(model):
    return model.Q_hr_heat + model.Q_aux == model.H_load
model.con_heat = Constraint(rule=heat_balance)

# 定义一个函数cooling_balance，用于计算冷却平衡
def cooling_balance(model):
    # 返回模型中的Q_hr_cool乘以COP_abs加上C_elec等于C_load
    return model.Q_hr_cool * model.COP_abs + model.C_elec == model.C_load
model.con_cooling = Constraint(rule=cooling_balance)

def p_gt_max(model):
    return model.P_gt <= model.P_gt_max
model.con_p_gt_max = Constraint(rule=p_gt_max)

def e_purchased_max(model):
    return model.E_purchased <= model.E_grid_max
model.con_e_purchased_max = Constraint(rule=e_purchased_max)

def q_aux_max(model):
    return model.Q_aux <= model.Q_aux_max
model.con_q_aux_max = Constraint(rule=q_aux_max)

def c_elec_max(model):
    return model.C_elec <= model.C_elec_max
model.con_c_elec_max = Constraint(rule=c_elec_max)
# ... 已有代码 ...

# 求解模型
# 请根据实际安装路径修改
solver_path = "D:\winglpk-4.65\glpk-4.65\w64\glpsol.exe" 

solver = SolverFactory('glpk', executable=solver_path)
result = solver.solve(model)

# ... 已有代码 ...

# 输出结果
print("优化结果：")
print(f"总成本: {model.total_cost():.2f} 元")
print(f"燃气轮机发电量: {model.P_gt():.2f} kW")
print(f"外购电量: {model.E_purchased():.2f} kW")
print(f"辅助锅炉供热量: {model.Q_aux():.2f} kW")
print(f"余热供热: {model.Q_hr_heat():.2f} kW")
print(f"余热制冷: {model.Q_hr_cool():.2f} kW")
print(f"电制冷量: {model.C_elec():.2f} kW")