import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
# 设置 matplotlib 使用支持中文的字体
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

# EPTE Calculation Parameters
electricity_prices = [0.6, 0.8]  # yuan/kWh
ng_prices = [2.5, 3.5]  # yuan/m³
reference_ng_price = 2.5  # Assume original fuel cost based on 2.5 yuan/m³
lifetime = 20  # years
operating_hours = 8000  # hours/year
total_hours = lifetime * operating_hours  # total operating hours

# Function to calculate EPTE
def calculate_epte(data, electricity_prices, ng_prices, reference_ng_price, total_hours):
    epte_elec_results = {price: [] for price in electricity_prices}
    epte_ng_results = {price: [] for price in ng_prices}
    for i, power in enumerate(data['Rated_Power']):
        gen_eff = data['Power_Gen_Efficiency'][i] / 100
        fuel_cost = data['Fuel_Cost'][i]
        maint_cost = data['Maintenance_Cost'][i]
        unit_price = data['Unit_Price'][i]
        
        total_energy = power * total_hours
        # 计算总成本
        total_cost = (fuel_cost + maint_cost) * total_energy
        unit_price = data['Unit_Price'][i]

        # 计算不同电价下的 EPTE
        for elec_price in electricity_prices:
            revenue = elec_price * gen_eff * total_energy
            net_profit = revenue - total_cost - unit_price
            epte = net_profit / unit_price
            epte_elec_results[elec_price].append(epte)

        # 计算不同天然气价格下的 EPTE
        for ng_price in ng_prices:
            adjusted_fuel_cost = fuel_cost * (ng_price / reference_ng_price)
            total_cost = (adjusted_fuel_cost + maint_cost) * total_energy
            revenue = 0.6 * gen_eff * total_energy
            net_profit = revenue - total_cost - unit_price
            epte = net_profit / unit_price
            epte_ng_results[ng_price].append(epte)
    
    return epte_elec_results, epte_ng_results

# Calculate EPTE for both data sets
epte_elec_1, epte_ng_1 = calculate_epte(data1, electricity_prices, ng_prices, reference_ng_price, total_hours)
epte_elec_2, epte_ng_2 = calculate_epte(data2, electricity_prices, ng_prices, reference_ng_price, total_hours)

# Plot EPTE for different electricity prices
plt.figure(figsize=(8, 6))
for elec_price in electricity_prices:
    plt.plot(data1['Rated_Power'], epte_elec_1[elec_price], label=f'Gas Turbine, Elec Price = {elec_price} yuan/kWh')
    plt.plot(data2['Rated_Power'], epte_elec_2[elec_price], label=f'Gas IC Engine, Elec Price = {elec_price} yuan/kWh')
plt.xlabel('Rated Power (kW)')
plt.ylabel('EPTE (无量纲)')
plt.title('EPTE vs. Rated Power for Different Electricity Prices')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'epte_electricity_prices.png'))
plt.close()

# Plot EPTE for different natural gas prices
plt.figure(figsize=(8, 6))
for ng_price in ng_prices:
    plt.plot(data1['Rated_Power'], epte_ng_1[ng_price], label=f'Gas Turbine, NG Price = {ng_price} yuan/m³')
    plt.plot(data2['Rated_Power'], epte_ng_2[ng_price], label=f'Gas IC Engine, NG Price = {ng_price} yuan/m³')
plt.xlabel('Rated Power (kW)')
plt.ylabel('EPTE (无量纲)')
plt.title('EPTE vs. Rated Power for Different Natural Gas Prices')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'epte_natural_gas_prices.png'))
plt.close()

print(f"Plots saved to {output_dir}")

