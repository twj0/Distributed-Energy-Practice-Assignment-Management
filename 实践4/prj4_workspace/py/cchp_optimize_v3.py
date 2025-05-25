# cchp_optimize_v3.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar
import os
import datetime

# --- Global Variables for User Choices (will be set in main) ---
# These are set here to allow plot_optimization_progress to access them
# if they are not passed explicitly, though passing is preferred.
SELECTED_FUNCTION_NUM_GLOBAL = 2
EPSILON_GLOBAL = 1e-4

# --- Objective Functions and Gradients ---
def f1(x):
    x1, x2 = x
    return 10 * (x1 - 1)**2 + (x2 + 1)**4

def grad_f1(x):
    x1, x2 = x
    return np.array([20 * (x1 - 1), 4 * (x2 + 1)**3])

def f2(x):
    x1, x2 = x
    return 100 * (x1**2 - x2)**2 + (x1 - 1)**2

def grad_f2(x):
    x1, x2 = x
    return np.array([400 * x1 * (x1**2 - x2) + 2 * (x1 - 1), -200 * (x1**2 - x2)])

def f3(x):
    x1, x2 = x
    return 100 * (x1**2 - 3*x2)**2 + (x1 - 1)**2

def grad_f3(x):
    x1, x2 = x
    return np.array([400 * x1 * (x1**2 - 3*x2) + 2 * (x1 - 1), -600 * (x1**2 - 3*x2)])

FUNC_MAPPING = {
    1: {"func": f1, "grad": grad_f1, "name": "f(x) = 10*(x1-1)^2 + (x2+1)^4"},
    2: {"func": f2, "grad": grad_f2, "name": "f(x) = 100*(x1^2 - x2)^2 + (x1-1)^2"}, # Rosenbrock
    3: {"func": f3, "grad": grad_f3, "name": "f(x) = 100*(x1^2 - 3*x2)^2 + (x1-1)^2"},
}

# --- Line Search Helper Functions ---
def phi(a, x_k, b_k, objective_func_phi):
    """Helper function for 1D minimization: f(x_k + a*b_k)"""
    return objective_func_phi(x_k + a * b_k)

def phi_prime(a, x_k, b_k, gradient_func_phi):
    """Derivative of phi(a) w.r.t. a: grad_f(x_k + a*b_k) . b_k"""
    x_new = x_k + a * b_k
    grad_at_x_new = gradient_func_phi(x_new)
    return np.dot(grad_at_x_new, b_k)

# --- Robust Line Search Implementations ---
def line_search_fsolve(x_k, b_k, objective_func, gradient_func):
    initial_guess_a = 0.001 
    try:
        a_optimal_roots = fsolve(lambda a_val: phi_prime(a_val, x_k, b_k, gradient_func),
                                 initial_guess_a, xtol=1e-7, maxfev=500) # More precise fsolve
        
        positive_roots = [r for r in np.atleast_1d(a_optimal_roots) if r > 1e-9] 

        if not positive_roots:
            # print(f"Debug (fsolve): No positive root. Roots: {a_optimal_roots}. Fallback.")
            return line_search_backtracking(x_k, b_k, objective_func, gradient_func, alpha_init=0.1)

        best_a = -1
        min_phi_val = phi(0, x_k, b_k, objective_func) # f(x_k)
        
        # Check roots and find the one that minimizes phi(a) the most
        candidate_as = sorted([r for r in positive_roots if r < 50.0]) # Filter large steps
        if not candidate_as: # If all positive roots were too large
             return line_search_backtracking(x_k, b_k, objective_func, gradient_func, alpha_init=0.01)

        for r_val in candidate_as:
            current_phi_val = phi(r_val, x_k, b_k, objective_func)
            if current_phi_val < min_phi_val:
                min_phi_val = current_phi_val
                best_a = r_val
        
        if best_a > 1e-9 : # A valid step was found
             return np.clip(best_a, 1e-9, 20.0) # Clip to reasonable bounds
        else:
            # print(f"Debug (fsolve): No root improved f(x). Fallback.")
            return line_search_backtracking(x_k, b_k, objective_func, gradient_func, alpha_init=0.01)

    except Exception as e:
        # print(f"Error in line_search_fsolve: {e}. Fallback.")
        return line_search_backtracking(x_k, b_k, objective_func, gradient_func, alpha_init=0.01)

def line_search_backtracking(x_k, b_k, objective_func, gradient_func, alpha_init=1.0, beta=0.5, c=1e-4):
    alpha = alpha_init
    f_k = objective_func(x_k)
    # grad_k = gradient_func(x_k) # For Armijo, use grad at x_k
    # slope = np.dot(grad_k, b_k) # b_k is -grad_k, so slope = -||grad_k||^2
    # For simplicity, if b_k is already -grad_k, then grad_k.b_k = -grad_k.grad_k
    slope = np.dot(gradient_func(x_k), b_k)


    if slope > -1e-12 : # Should be significantly negative for descent
        # print(f"Warning (backtracking): Direction b_k (slope: {slope:.2e}) not a strong descent. Using small fixed step.")
        return 1e-5 # Return a very small step

    max_backtrack_iters = 50
    for _ in range(max_backtrack_iters):
        if objective_func(x_k + alpha * b_k) <= f_k + c * alpha * slope:
            return alpha
        alpha *= beta
        if alpha < 1e-10: # Prevent infinitely small steps
            # print("Warning (backtracking): Alpha too small. Using minimal step.")
            return 1e-9 
    # print("Warning (backtracking): Max backtrack iterations reached. Using last alpha.")
    return alpha # Or a very small fixed value if max iters hit

def line_search_minimize_scalar(x_k, b_k, objective_func, gradient_func_for_fallback):
    try:
        res = minimize_scalar(lambda a_val: phi(a_val, x_k, b_k, objective_func),
                              bounds=(0, 20), method='bounded', options={'xatol': 1e-7, 'maxiter':100}) # Wider bounds, more precise
        if res.success and res.x > 1e-9:
            return res.x
        else:
            # print(f"Warning (minimize_scalar): Failed (Success: {res.success}, x: {res.x:.2e}). Fallback.")
            return line_search_backtracking(x_k, b_k, objective_func, gradient_func_for_fallback, alpha_init=0.01)
    except Exception as e:
        # print(f"Error in line_search_minimize_scalar: {e}. Fallback.")
        return line_search_backtracking(x_k, b_k, objective_func, gradient_func_for_fallback, alpha_init=0.01)

# --- Gradient Descent Algorithm ---
def gradient_descent(selected_func_num, x_initial_coords, epsilon_val, max_iters_val, line_search_method_str):
    objective_func = FUNC_MAPPING[selected_func_num]["func"]
    gradient_func = FUNC_MAPPING[selected_func_num]["grad"]
    func_name_str = FUNC_MAPPING[selected_func_num]["name"]

    x_current = np.array(x_initial_coords, dtype=float)
    history = {
        'x_values': [x_current.copy()], 'f_values': [objective_func(x_current)],
        'errors': [], 'gradients_norm': [], 'step_lengths': []
    }
    iterations_completed = 0

    print(f"\n🚀 Optimizing: {func_name_str}")
    print(f"Initial: x0 = {x_current}, f(x0) = {history['f_values'][0]:.6e}")
    print(f"Settings: ε = {epsilon_val:.1e}, MaxIters = {max_iters_val}, LineSearch = {line_search_method_str}")
    print("-" * 95)
    header = f"{'Iter':<5} | {'x1':<13} | {'x2':<13} | {'f(x)':<16} | {'||∇f(x)||':<13} | {'Error':<13} | {'Step (a)':<10}"
    print(header)
    print("-" * 95)

    for k_iter in range(max_iters_val):
        iterations_completed = k_iter + 1
        grad = gradient_func(x_current)
        grad_norm = np.linalg.norm(grad)
        history['gradients_norm'].append(grad_norm)

        b_k = -grad 

        if grad_norm < epsilon_val * 0.001: # More aggressive stop if gradient is tiny
            print(f"\n✅ Gradient norm ({grad_norm:.2e}) is very small. Optimization likely converged.")
            break
        
        if line_search_method_str == 'fsolve':
            a = line_search_fsolve(x_current, b_k, objective_func, gradient_func)
        elif line_search_method_str == 'minimize_scalar':
            a = line_search_minimize_scalar(x_current, b_k, objective_func, gradient_func)
        else: # Default to backtracking
            a = line_search_backtracking(x_current, b_k, objective_func, gradient_func)
        
        history['step_lengths'].append(a)

        if a < 1e-10: # If step length is effectively zero
            print("\n⚠️ Step length 'a' is extremely small. Stopping to prevent stagnation.")
            iterations_completed -=1 
            break

        x_next = x_current + a * b_k
        error = np.linalg.norm(x_next - x_current)

        history['x_values'].append(x_next.copy())
        history['f_values'].append(objective_func(x_next))
        history['errors'].append(error)
        
        # Print progress: first 15, then every 20% of max_iters, or if converged
        if k_iter < 15 or (k_iter + 1) % (max_iters_val // 20 if max_iters_val > 40 else 1) == 0 or error < epsilon_val:
            print(f"{iterations_completed:<5} | {x_next[0]:<13.6f} | {x_next[1]:<13.6f} | {history['f_values'][-1]:<16.6e} | {grad_norm:<13.2e} | {error:<13.6e} | {a:<10.4e}")

        if error < epsilon_val:
            print(f"\n🏁 Convergence achieved at iteration {iterations_completed}: Error < Epsilon.")
            break
        
        x_current = x_next

    print("-" * 95)
    if iterations_completed == max_iters_val and (not history['errors'] or history['errors'][-1] >= epsilon_val):
        print(f"\n⚠️ Maximum iterations ({max_iters_val}) reached. Full convergence may not be achieved.")
    
    final_x = history['x_values'][-1]
    final_f = history['f_values'][-1]
    
    print("\n--- Optimization Summary ---")
    print(f"Function Optimized: {selected_func_num} ({func_name_str})")
    print(f"Initial Point x0: {x_initial_coords}")
    print(f"Epsilon (Accuracy): {epsilon_val:.1e}")
    print(f"Line Search Method: {line_search_method_str}")
    print("-----------------------------")
    print(f"Total Iterations: {iterations_completed}")
    print(f"Optimal x = ({final_x[0]:.8f}, {final_x[1]:.8f})")
    print(f"Optimal f(x) = {final_f:.8e}")
    if history['errors']:
        print(f"Final Error (||Δx||) = {history['errors'][-1]:.8e}")
    if history['gradients_norm']:
         print(f"Final Gradient Norm ||∇f(x)|| = {history['gradients_norm'][-1]:.8e}")

    return history

# --- Plotting and Saving ---
def plot_and_save_results(history, selected_func_num, x_initial_coords, line_search_method_str, epsilon_plot):
    objective_func_plot = FUNC_MAPPING[selected_func_num]["func"]
    func_name_str_plot = FUNC_MAPPING[selected_func_num]["name"]

    if not history['errors'] and len(history['f_values']) <= 1:
        print("\n📉 Not enough data for detailed plotting.")
        return

    num_main_plots = 2 # f(x) vs iter, error vs iter
    x_coords = np.array(history['x_values'])
    plot_contour = x_coords.shape[0] > 1 # Only plot contour if we have iterations
    
    if plot_contour:
        num_main_plots = 3
        
    fig = plt.figure(figsize=(7 * num_main_plots, 5.5)) 

    # Plot 1: Objective Function Value
    ax1 = fig.add_subplot(1, num_main_plots, 1)
    ax1.plot(range(len(history['f_values'])), history['f_values'], marker='.', linestyle='-', color='dodgerblue')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x) value")
    ax1.set_title("Objective Function Convergence")
    if any(f_val > 0 for f_val in history['f_values']):
        try:
            ax1.set_yscale('log')
        except ValueError: # Handle cases where values are too small or non-positive for log
            ax1.set_yscale('linear')
    ax1.grid(True, which="both", ls=":", alpha=0.7)

    # Plot 2: Error
    ax2 = fig.add_subplot(1, num_main_plots, 2)
    if history['errors']:
        ax2.plot(range(1, len(history['errors']) + 1), history['errors'], marker='.', linestyle='-', color='orangered')
        ax2.axhline(y=epsilon_plot, color='k', linestyle='--', label=f'ε = {epsilon_plot:.1e}')
        ax2.set_ylabel("Error ||x_k+1 - x_k||")
        ax2.set_title("Error Convergence")
        ax2.set_yscale('log')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No error data", ha='center', va='center')
        ax2.set_title("Error Convergence")
    ax2.set_xlabel("Iteration")
    ax2.grid(True, which="both", ls=":", alpha=0.7)
    
    # Plot 3: Contour plot
    if plot_contour:
        ax3 = fig.add_subplot(1, num_main_plots, 3)
        x1_path = x_coords[:, 0]
        x2_path = x_coords[:, 1]
        
        # Auto-range for contour plot
        x1_min, x1_max = x1_path.min(), x1_path.max()
        x2_min, x2_max = x2_path.min(), x2_path.max()
        x1_range_delta = abs(x1_max - x1_min)
        x2_range_delta = abs(x2_max - x2_min)

        margin_x = max(x1_range_delta * 0.2, 0.5) # Ensure some minimum margin
        margin_y = max(x2_range_delta * 0.2, 0.5)
        
        grid_x1 = np.linspace(x1_min - margin_x, x1_max + margin_x, 100)
        grid_x2 = np.linspace(x2_min - margin_y, x2_max + margin_y, 100)
        
        if np.allclose(grid_x1, grid_x1[0]) or np.allclose(grid_x2, grid_x2[0]):
             ax3.text(0.5,0.5, "Path too small for contour.", ha='center', va='center')
        else:
            X1_grid, X2_grid = np.meshgrid(grid_x1, grid_x2)
            Z_grid = objective_func_plot(np.array([X1_grid, X2_grid]))
            
            levels = 25
            try: # Log levels for functions with large value ranges
                f_vals_positive = [f for f in history['f_values'] if f > 1e-9]
                if f_vals_positive and max(f_vals_positive) / min(f_vals_positive) > 100:
                    levels = np.logspace(np.log10(min(f_vals_positive)), np.log10(max(f_vals_positive)), 20)
            except: pass # Use linear levels if log fails
                
            cp = ax3.contour(X1_grid, X2_grid, Z_grid, levels=levels, cmap='viridis', alpha=0.7)
            fig.colorbar(cp, ax=ax3, label="f(x)", shrink=0.8)
            
            ax3.plot(x1_path, x2_path, 'o-', color='red', markersize=2, linewidth=1, label='Path')
            ax3.plot(x_initial_coords[0], x_initial_coords[1], 'X', color='blue', markersize=8, label='Start')
            ax3.plot(x1_path[-1], x2_path[-1], '*', color='lime', markersize=10, markeredgecolor='black',label='End')
            ax3.set_xlabel("x1")
            ax3.set_ylabel("x2")
            ax3.set_title("Optimization Path")
            ax3.legend(fontsize='small')
            ax3.axis('tight')
            ax3.grid(True, ls=":", alpha=0.5)

    suptitle_text = (f"Optimization: {func_name_str_plot}\n"
                     f"Initial: ({x_initial_coords[0]:.2f}, {x_initial_coords[1]:.2f}), "
                     f"LS: {line_search_method_str}, ε: {epsilon_plot:.1e}")
    fig.suptitle(suptitle_text, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93]) 

    # Saving the figure
    fig_dir = "fig"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print(f"✅ Created directory: ./{fig_dir}")

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    x0_str_fn = f"{str(x_initial_coords[0]).replace('.', 'p')}_{str(x_initial_coords[1]).replace('.', 'p')}"
    file_name = f"opt_F{selected_func_num}_x0_{x0_str_fn}_LS_{line_search_method_str}_{time_stamp}.png"
    file_path = os.path.join(fig_dir, file_name)

    try:
        plt.savefig(file_path, dpi=150)
        print(f"📊 Figure saved to: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"❌ Error saving figure: {e}")
    
    plt.show()

# --- Main Execution with User Input ---
if __name__ == "__main__":
    print("--- Gradient Descent Optimization (v3) ---")
    
    while True:
        try:
            prompt = "🎯 Select objective function (1, 2, or 3, default 2):\n"
            for i, data in FUNC_MAPPING.items():
                prompt += f"   {i}: {data['name']}\n"
            prompt += "Choice: "
            SELECTED_FUNCTION_NUM_GLOBAL = int(input(prompt) or "2")
            if SELECTED_FUNCTION_NUM_GLOBAL in FUNC_MAPPING:
                break
            print("Invalid selection. Please enter 1, 2, or 3.")
        except ValueError: print("Invalid input. Please enter a number.")

    while True:
        try:
            x0_input_str = input("📍 Enter initial point x0 as 'x1,x2' (e.g., 0,0 or -1.2,1, default 0,0): ") or "0,0"
            user_x_initial = [float(x.strip()) for x in x0_input_str.split(',')]
            if len(user_x_initial) == 2: break
            print("Please enter two comma-separated values for x1 and x2.")
        except ValueError: print("Invalid format. Use 'x1,x2' (e.g., 0.0,0.0).")
    
    while True:
        try:
            EPSILON_GLOBAL = float(input(f"🔍 Enter convergence epsilon ε (e.g., 1e-4, default {1e-4:.1e}): ") or f"{1e-4:.1e}")
            if EPSILON_GLOBAL > 0: break
            print("Epsilon must be positive.")
        except ValueError: print("Invalid number for epsilon.")

    while True:
        try:
            user_max_iterations = int(input(f"🔄 Enter maximum iterations (e.g., 1000, default 1000): ") or "1000")
            if user_max_iterations > 0: break
            print("Maximum iterations must be positive.")
        except ValueError: print("Invalid integer for maximum iterations.")

    while True:
        user_line_search_method = input(
            "🌊 Select line search: 'fsolve', 'backtracking', 'minimize_scalar' (default 'backtracking'): "
        ).lower() or "backtracking"
        if user_line_search_method in ['fsolve', 'backtracking', 'minimize_scalar']: break
        print("Invalid method. Choose from the list.")

    # Call the optimization
    hist = gradient_descent(SELECTED_FUNCTION_NUM_GLOBAL, user_x_initial, EPSILON_GLOBAL, 
                              user_max_iterations, user_line_search_method)
    
    # Plot and save results
    plot_and_save_results(hist, SELECTED_FUNCTION_NUM_GLOBAL, user_x_initial, 
                          user_line_search_method, EPSILON_GLOBAL)

    print("\n--- Expected Optimal Solutions (Approximate) ---")
    expected = {
        1: "x ≈ (1.0, -1.0), f(x) ≈ 0",
        2: "x ≈ (1.0, 1.0), f(x) ≈ 0 (Rosenbrock)",
        3: "x ≈ (1.0, 0.333), f(x) ≈ 0"
    }
    print(f"For Function {SELECTED_FUNCTION_NUM_GLOBAL}: {expected.get(SELECTED_FUNCTION_NUM_GLOBAL, 'N/A')}")
    print("\n✨ Optimization complete.")