import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# Create output directory for figures
output_dir = os.path.join(os.getcwd(), 'fig')
os.makedirs(output_dir, exist_ok=True)

# Data from the first table 
# 燃气轮机发电机组的热电性能
data1 = {
    'Rated_Power': [30, 60, 70, 100, 350, 1000, 5000, 10000, 25000, 40000],
    'Power_Gen_Efficiency': [20.4, 20.4, 20.2, 20.0, 20.0, 20.9, 20.1, 20, 30.3, 30.0],
    'Thermoelectric_Efficiency': [70, 70 , 60, 70, 70, 60, 60, 70, 70, 70],
    'Maintenance_Cost': [],
    'Fuel_Cost': [0.030, 0.030, 0.030, 0.030, 0.030, 0.0307, 0.030, 0.0307, 0.0307, 0.0309],
    'Unit_Price': [213.0, 413.06, 513.08, 613.0, 2130.5, 5130.5, 11300, 31300, 91300, 113040]
}

# Calculate Maintenance_Cost for data1
power_gen_eff_1 = data1['Power_Gen_Efficiency']
thermoelectric_eff_1 = data1['Thermoelectric_Efficiency']
data1['Maintenance_Cost'] = [gen_eff / (thermo_eff - gen_eff) for gen_eff, thermo_eff in zip(power_gen_eff_1, thermoelectric_eff_1)]

# Data from the second table 
# 燃气内燃机发电机组的热电性能
data2 = {
    'Rated_Power': [11, 30, 50, 75, 100, 300, 800, 3000, 5000],
    'Power_Gen_Efficiency': [20.0, 20.5, 20.2, 20.6, 30.6, 30.1, 30.3, 30.0, 30.0],
    'Thermoelectric_Efficiency': [80.0, 80.8, 80.7, 80.0, 80.0, 70.0, 70.0, 70.0, 70.0],
    'Maintenance_Cost': [],
    'Fuel_Cost': [0.3028, 0.3028, 0.3028, 0.3028, 0.3027, 0.3062, 0.0305, 0.0302, 0.0302],
    'Unit_Price': [11.30, 113.0, 113.0, 213.0, 213.0, 513.0, 1130.16, 9130, 11305]
}

# Calculate Maintenance_Cost for data2
power_gen_eff_2 = data2['Power_Gen_Efficiency']
thermoelectric_eff_2 = data2['Thermoelectric_Efficiency']
data2['Maintenance_Cost'] = [gen_eff / (thermo_eff - gen_eff) for gen_eff, thermo_eff in zip(power_gen_eff_2, thermoelectric_eff_2)]

# 定义三次多项式拟合函数
def cubic_polynomial(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 定义拟合函数
def fit_data(data):
    x_data = np.array(data['Rated_Power'])
    dependent_vars = ['Power_Gen_Efficiency', 'Thermoelectric_Efficiency', 'Maintenance_Cost', 'Fuel_Cost']
    fit_params = {}
    for var in dependent_vars:
        y_data = np.array(data[var])
        # 进行曲线拟合
        popt, _ = curve_fit(cubic_polynomial, x_data, y_data)
        fit_params[var] = popt
    return fit_params

# 对燃气轮机和燃气内燃机数据进行拟合
fit_params1 = fit_data(data1)
fit_params2 = fit_data(data2)

# 打印拟合参数
print("燃气轮机拟合参数:")
for var, params in fit_params1.items():
    print(f"{var}: {params}")

print("\n燃气内燃机拟合参数:")
for var, params in fit_params2.items():
    print(f"{var}: {params}")

# 绘制拟合曲线
def plot_fit(data, fit_params, label):
    x_data = np.array(data['Rated_Power'])
    dependent_vars = ['Power_Gen_Efficiency', 'Thermoelectric_Efficiency', 'Maintenance_Cost', 'Fuel_Cost']
    plt.figure(figsize=(12, 8))
    for i, var in enumerate(dependent_vars):
        y_data = np.array(data[var])
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = cubic_polynomial(x_fit, *fit_params[var])
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_data, y_data, label='原始数据')
        plt.plot(x_fit, y_fit, 'r-', label='拟合曲线')
        plt.title(f'{label} - {var} 拟合')
        plt.xlabel('Rated_Power')
        plt.ylabel(var)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_fit.png'))
    plt.close()

# 绘制燃气轮机和燃气内燃机的拟合曲线
plot_fit(data1, fit_params1, '燃气轮机')
plot_fit(data2, fit_params2, '燃气内燃机')

print(f"拟合曲线已保存到 {output_dir}")