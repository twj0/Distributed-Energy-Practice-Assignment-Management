import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math

# --- Configuration & Constants ---
DATA_FILE = 'order-63.txt'
OUTPUT_DIR = 'fig_综合评价结果_v2' # Changed output directory
RHO_GRA = 0.5

ALPHA_FUZZY = 1.1086
BETA_FUZZY = 0.8942
A_FUZZY = 0.3915
B_FUZZY = 0.3699

# Matplotlib setup
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    font = FontProperties(family='SimHei')
except:
    print("SimHei font not found, using default font for plots.")
    font = FontProperties()

# --- Helper Functions ---

def fuzzy_membership_function(x_qualitative):
    """
    Quantifies qualitative indicators using the fuzzy membership function.
    Input x_qualitative is a rating (1 to 5).
    f(1)=0.01, f(3)=0.8, f(5)=1.0
    Formula from PDF page 11:
    - [1 + alpha * (x - beta)^(-2)]^(-1)  for 1 <= x <= 3
    - a * ln(x) + b                       for 3 < x <= 5
    """
    x = float(x_qualitative) # Ensure float for calculations

    if not (1.0 <= x <= 5.0):
        # This should catch any out-of-range raw inputs if they occur
        raise ValueError(f"Qualitative input x={x} is out of the expected range [1, 5]")

    value = np.nan # Initialize to NaN to catch unhandled cases

    if 1.0 <= x <= 3.0: # This correctly includes x=1, x=2, and x=3
        # Term (x - BETA_FUZZY)^2
        # For x=1: (1 - 0.8942)^2 = (0.1058)^2 = 0.01119364
        # For x=2: (2 - 0.8942)^2 = (1.1058)^2 = 1.22279364
        # For x=3: (3 - 0.8942)^2 = (2.1058)^2 = 4.43438564
        # None of these are zero.
        denominator_of_power_term = (x - BETA_FUZZY)**2
        
        if abs(denominator_of_power_term) < 1e-12:
            # This case should not be hit with integer x in [1,3] and BETA_FUZZY=0.8942
            # If it were, it implies x is almost exactly BETA_FUZZY.
            # alpha / (small_number) -> large number
            # 1 / (1 + large_number) -> approaches 0
            print(f"Warning: (x - BETA_FUZZY)^2 is near zero for x={x}. Result may be imprecise.")
            value = 0.0 
        else:
            # Formula: 1.0 / (1.0 + ALPHA_FUZZY * (x - BETA_FUZZY)^(-2.0))
            # which is 1.0 / (1.0 + ALPHA_FUZZY / denominator_of_power_term)
            value = 1.0 / (1.0 + ALPHA_FUZZY / denominator_of_power_term)
            
    elif 3.0 < x <= 5.0: # This correctly includes x=4 and x=5
        # For x=4: 0.3915 * ln(4) + 0.3699 = 0.3915 * 1.38629 + 0.3699 = 0.54276 + 0.3699 = 0.91266
        # For x=5: 0.3915 * ln(5) + 0.3699 = 0.3915 * 1.60943 + 0.3699 = 0.63010 + 0.3699 = 1.00000
        value = A_FUZZY * math.log(x) + B_FUZZY
    
    if np.isnan(value):
        # This means x was in [1,5] but didn't match either if/elif block.
        # This should not happen if the conditions are 1<=x<=3 and 3<x<=5
        print(f"Error: No fuzzy value computed for x={x}. Check conditional logic.")
        
    return value

def calculate_ahp_weights(judgement_matrix):
    n = judgement_matrix.shape[0]
    if n == 0: return np.array([]), 0.0, 0.0
    if n == 1: return np.array([1.0]), 0.0, 1.0 # Single criterion

    eigenvalues, eigenvectors = np.linalg.eig(judgement_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_eigenvalue_index].real
    w = eigenvectors[:, max_eigenvalue_index].real
    weights = w / np.sum(w)
    
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri_values = {1:0, 2:0, 3:0.52, 4:0.89, 5:1.11, 6:1.25, 7:1.35, 8:1.40, 9:1.45, 10:1.49, 11:1.51, 12:1.52} # Updated RI from common sources
    ri = ri_values.get(n, 1.52) 
    if n > len(ri_values) and ri == ri_values[max(ri_values.keys())] : # Using RI for largest n if not found
         print(f"Warning: RI for n={n} not explicitly defined in lookup, using RI for n={max(ri_values.keys())}. CR accuracy may be affected.")

    cr = ci / ri if ri != 0 else (0 if ci == 0 else float('inf')) # Handle RI=0 for n=1,2
    return weights, cr, lambda_max

def normalize_quantitative_data(data_subset, indicator_types_subset):
    norm_data = np.zeros_like(data_subset, dtype=float)
    for j in range(data_subset.shape[1]):
        col_data = data_subset[:, j]
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        
        if abs(max_val - min_val) < 1e-9: # Handles float comparison for equality
            # All values in the column are (practically) the same.
            # Normalized value should be 1.0 if the identical value is optimal,
            # or some mid-value (0.5) if it's neutral.
            # Given GRA expects higher is better, and if all are same, they are equally good/bad relative to each other.
            # If it's a positive indicator and all values are say 5, then (5-5)/(5-5) is problem.
            # Let's assign 1.0, implying they are all at the same (potentially optimal or non-differentiable) level.
            norm_data[:, j] = 1.0 
        elif indicator_types_subset[j] == 1:
            norm_data[:, j] = (col_data - min_val) / (max_val - min_val)
        else:
            norm_data[:, j] = (max_val - col_data) / (max_val - min_val)
    return norm_data

def grey_relational_analysis(processed_data, global_weights, rho=0.5):
    num_schemes, num_indicators = processed_data.shape
    if np.isnan(processed_data).any():
        print("Error: NaN values detected in data provided to GRA.")
        # You might want to return NaN arrays or raise an error
        nan_gra_coeffs = np.full((num_schemes, num_indicators), np.nan)
        nan_gra_degrees = np.full(num_schemes, np.nan)
        return nan_gra_coeffs, nan_gra_degrees

    reference_sequence = np.ones(num_indicators)
    abs_diff = np.abs(processed_data - reference_sequence)
    min_abs_diff_all = np.min(abs_diff)
    max_abs_diff_all = np.max(abs_diff)

    if abs(max_abs_diff_all) < 1e-9: # All processed data is identical to reference (all 1s)
        ksi = np.ones_like(processed_data)
    else:
        ksi = (min_abs_diff_all + rho * max_abs_diff_all) / (abs_diff + rho * max_abs_diff_all)
    
    if global_weights.ndim > 1: global_weights = global_weights.flatten()
    if len(global_weights) != num_indicators:
        raise ValueError(f"Weight/indicator mismatch for GRA: {len(global_weights)} vs {num_indicators}")
    gra_degree = np.dot(ksi, global_weights)
    return ksi, gra_degree

# --- Main Processing ---
def main():
    print("--- Comprehensive Evaluation of Distributed Energy Systems (v2) ---")

    raw_indicator_data_transposed = np.loadtxt(DATA_FILE, dtype=float)
    raw_indicator_data = raw_indicator_data_transposed.T
    scheme_names = [f'方案{chr(65+i)}' for i in range(raw_indicator_data.shape[0])]
    
    base_indicator_names = [
        '初投资', '投资回收期', '总费用年值', '净现值',
        '氮氧化物', 'CO', 'CO₂',
        '技术先进性', '安全性', '维护方便性',
        '一次能源比', '噪声'
    ]
    base_indicator_types = [0,0,0,1, 0,0,0, 1,1,1, 0,0] # Original nature
    qualitative_indicator_indices = [7, 8, 9]

    print(f"\nLoaded raw data for {raw_indicator_data.shape[0]} schemes x {raw_indicator_data.shape[1]} indicators.")
    df_raw_data = pd.DataFrame(raw_indicator_data, index=scheme_names, columns=base_indicator_names)
    print("Raw Data:")
    print(df_raw_data)

    # --- Indicator Preprocessing ---
    processed_data = np.zeros_like(raw_indicator_data, dtype=float)
    quantitative_indices = [i for i in range(len(base_indicator_names)) if i not in qualitative_indicator_indices]
    
    print("\n--- Fuzzifying Qualitative Indicators ---")
    for idx_qual in qualitative_indicator_indices:
        indicator_name_qual = base_indicator_names[idx_qual]
        # print(f"Processing Qualitative Indicator: {indicator_name_qual} (index {idx_qual})")
        for i_scheme in range(raw_indicator_data.shape[0]):
            scheme_name_qual = scheme_names[i_scheme]
            raw_value = raw_indicator_data[i_scheme, idx_qual]
            fuzzified_value = fuzzy_membership_function(raw_value)
            # print(f"  Scheme: {scheme_name_qual}, Raw: {raw_value}, Fuzzified: {fuzzified_value:.6f}")
            processed_data[i_scheme, idx_qual] = fuzzified_value
    
    raw_quantitative_data = raw_indicator_data[:, quantitative_indices]
    quantitative_types_subset = [base_indicator_types[i] for i in quantitative_indices]
    normalized_quantitative_data = normalize_quantitative_data(raw_quantitative_data, quantitative_types_subset)
    
    current_quant_col = 0
    for j in range(raw_indicator_data.shape[1]):
        if j not in qualitative_indicator_indices:
            processed_data[:, j] = normalized_quantitative_data[:, current_quant_col]
            current_quant_col += 1
            
    df_processed_data = pd.DataFrame(processed_data, index=scheme_names, columns=base_indicator_names)
    print("\nProcessed (Normalized/Fuzzified) Indicator Data (All higher-is-better):")
    print(df_processed_data)
    if df_processed_data.isnull().values.any():
        print("ERROR: NaN values found in processed_data BEFORE GRA. Debug fuzzification/normalization.")
        return


    # --- AHP Weight Calculation ---
    print("\n--- AHP Weight Calculation ---")
    # !! CRITICAL: User must verify these judgement matrices for their specific problem !!
    # Level 1: Criteria (Economic, Environmental, Social, Performance, Noise)
    criteria_judgement_matrix = np.array([ # Based on old code's "comprehensive_matrix"
        [1,     5,    9,    3,    7],    # Economic (vs E, Env, Soc, P, N)
        [1/5,   1,    6,    1/3,  1/4],  # Environmental
        [1/9,   1/6,  1,    1/5,  1/3],  # Social
        [1/3,   3,    5,    1,    9],    # Performance
        [1/7,   4,    3,    1/9,  1]     # Noise
    ])
    level1_weights, cr1, lm1 = calculate_ahp_weights(criteria_judgement_matrix)
    print(f"Level 1 Criteria Weights (E, Env, Soc, P, N): {np.round(level1_weights,4)}")
    print(f"L1: Lambda_max={lm1:.4f}, CR={cr1:.4f} {'(Consistent)' if cr1 < 0.1 else '(INCONSISTENT - REVIEW L1 MATRIX!)'}")

    # Level 2: Sub-criteria
    economic_judgement_matrix = np.array([ [1,3,5,7], [1/3,1,6,1/4], [1/5,1/6,1,1/3], [1/7,4,3,1] ])
    w_econ, cr_econ, lm_econ = calculate_ahp_weights(economic_judgement_matrix)
    print(f"\nEconomic Sub-weights: {np.round(w_econ,4)}")
    print(f"Econ: Lambda_max={lm_econ:.4f}, CR={cr_econ:.4f} {'(Consistent)' if cr_econ < 0.1 else '(INCONSISTENT - REVIEW Econ MATRIX!)'}")

    environmental_judgement_matrix = np.array([ [1,6,4], [1/6,1,1/3], [1/4,3,1] ])
    w_env, cr_env, lm_env = calculate_ahp_weights(environmental_judgement_matrix)
    print(f"\nEnvironmental Sub-weights: {np.round(w_env,4)}")
    print(f"Env: Lambda_max={lm_env:.4f}, CR={cr_env:.4f} {'(Consistent)' if cr_env < 0.1 else '(INCONSISTENT - REVIEW Env MATRIX!)'}")

    social_judgement_matrix = np.array([ [1,1/3,5], [3,1,7], [1/5,1/7,1] ])
    w_social, cr_social, lm_social = calculate_ahp_weights(social_judgement_matrix)
    print(f"\nSocial Sub-weights: {np.round(w_social,4)}")
    print(f"Social: Lambda_max={lm_social:.4f}, CR={cr_social:.4f} {'(Consistent)' if cr_social < 0.1 else '(INCONSISTENT - REVIEW Social MATRIX!)'}")

    w_perf = np.array([1.0]) # Single indicator
    w_noise = np.array([1.0])# Single indicator

    global_weights = np.concatenate([
        level1_weights[0] * w_econ, level1_weights[1] * w_env,
        level1_weights[2] * w_social, level1_weights[3] * w_perf,
        level1_weights[4] * w_noise
    ])
    global_weights = global_weights / np.sum(global_weights) 
    print("\nGlobal Weights for Base Indicators (Sum to 1):")
    df_global_weights = pd.DataFrame({'Indicator': base_indicator_names, 'GlobalWeight': global_weights})
    print(df_global_weights.to_string())
    print(f"Sum of Global Weights: {np.sum(global_weights):.4f}")

    # --- Grey Relational Analysis ---
    print("\n--- Grey Relational Analysis ---")
    gra_coefficients, gra_degrees = grey_relational_analysis(processed_data, global_weights, rho=RHO_GRA)
    
    df_gra_coeffs = pd.DataFrame(gra_coefficients, index=scheme_names, columns=base_indicator_names)
    print("\nGrey Relational Coefficients Matrix (ksi):")
    print(df_gra_coeffs.round(4))
    
    df_results = pd.DataFrame({'Scheme': scheme_names, 'GRA_Degree': gra_degrees})
    df_results = df_results.sort_values(by='GRA_Degree', ascending=False).reset_index(drop=True)
    df_results['Rank'] = df_results.index + 1
    
    print("\nFinal Evaluation Results (Ranked by GRA Degree):")
    print(df_results.round(4))

    if not df_results.empty and 'Scheme' in df_results.columns:
        optimal_scheme = df_results.loc[0, 'Scheme']
        print(f"\nOptimal Scheme based on GRA: {optimal_scheme}")
    else:
        print("\nCould not determine optimal scheme due to errors in GRA results.")


    # --- Output and Visualization ---
    if not df_results['GRA_Degree'].isnull().all(): # Plot only if GRA degrees are not all NaN
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_results['Scheme'], df_results['GRA_Degree'], color='skyblue')
        plt.xlabel('方案 (Scheme)', fontproperties=font)
        plt.ylabel('综合灰色关联度 (GRA Degree)', fontproperties=font)
        plt.title('各方案综合评价灰色关联度 (GRA Evaluation Results)', fontproperties=font)
        plt.xticks(rotation=45)
        for bar_widget in bars: # bar is a widget, not a numerical value
            yval = bar_widget.get_height()
            if not np.isnan(yval):
                 plt.text(bar_widget.get_x() + bar_widget.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'GRA_Results_Comparison.png'), dpi=300)
        # plt.show() # Optionally show plot
        plt.close()
    else:
        print("Skipping plot generation due to NaN GRA degrees.")

    excel_path = os.path.join(OUTPUT_DIR, 'Comprehensive_Evaluation_Results.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        df_raw_data.to_excel(writer, sheet_name='Raw_Data')
        df_processed_data.to_excel(writer, sheet_name='Processed_Data')
        df_global_weights.to_excel(writer, sheet_name='Global_Weights', index=False)
        df_gra_coeffs.to_excel(writer, sheet_name='GRA_Coefficients')
        df_results.to_excel(writer, sheet_name='Final_Ranking_GRA', index=False)
    print(f"\nAll results saved to: {excel_path}")

if __name__ == "__main__":
    main()