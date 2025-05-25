import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar # For line search options

# --- Configuration (will be set by user input later) ---
# SELECTED_FUNCTION = 2
# X_INITIAL = np.array([0.0, 0.0])
# EPSILON = 1.0e-4
# MAX_ITERATIONS = 1000
# LINE_SEARCH_METHOD = 'fsolve' # or 'backtracking' or 'minimize_scalar'

# --- Objective Functions and Gradients (same as before) ---
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

# Global dictionary to hold selected functions
FUNC_MAPPING = {
    1: {"func": f1, "grad": grad_f1, "name": "f(x) = 10*(x1-1)^2 + (x2+1)^4"},
    2: {"func": f2, "grad": grad_f2, "name": "f(x) = 100*(x1^2 - x2)^2 + (x1-1)^2"},
    3: {"func": f3, "grad": grad_f3, "name": "f(x) = 100*(x1^2 - 3*x2)^2 + (x1-1)^2"},
}
# --- Line Search Implementations ---

def phi(a, x_k, b_k, objective_func_phi):
    """Helper function for 1D minimization: f(x_k + a*b_k)"""
    return objective_func_phi(x_k + a * b_k)

def phi_prime(a, x_k, b_k, gradient_func_phi):
    """Derivative of phi(a) w.r.t. a: grad_f(x_k + a*b_k) . b_k"""
    x_new = x_k + a * b_k
    grad_at_x_new = gradient_func_phi(x_new)
    return np.dot(grad_at_x_new, b_k)

def line_search_fsolve(x_k, b_k, objective_func, gradient_func):
    """Finds step length 'a' by solving phi'(a) = 0 using fsolve."""
    initial_guess_a = 0.001 # Start with a small positive guess
    try:
        # Using lambda to correctly pass current objective_func and gradient_func
        a_optimal_roots = fsolve(lambda a_val: phi_prime(a_val, x_k, b_k, gradient_func),
                                 initial_guess_a, xtol=1e-6) # Reduced xtol for fsolve
        
        positive_roots = [r for r in np.atleast_1d(a_optimal_roots) if r > 1e-8] # Slightly more tolerance for positive

        if not positive_roots:
            # print(f"Warning (fsolve): No positive root for 'a'. Roots: {a_optimal_roots}. Trying backtracking.")
            return line_search_backtracking(x_k, b_k, objective_func, gradient_func) # Fallback

        # Select the root that gives the best reduction in f, or smallest positive
        best_a = -1
        min_phi_val = float('inf')
        for r_val in positive_roots:
            if r_val > 100: continue # Avoid extremely large steps
            current_phi_val = phi(r_val, x_k, b_k, objective_func)
            if current_phi_val < min_phi_val:
                min_phi_val = current_phi_val
                best_a = r_val
        
        if best_a > 0:
             # Check if this 'a' actually reduces the function value compared to a=0
            if min_phi_val < phi(0, x_k, b_k, objective_func) - 1e-9: # Ensure some decrease
                 return np.clip(best_a, 1e-8, 10.0) # Clip to reasonable bounds
            else:
                # print(f"Warning (fsolve): Root a={best_a:.2e} does not improve f. Trying backtracking.")
                return line_search_backtracking(x_k, b_k, objective_func, gradient_func)
        else:
            # print(f"Warning (fsolve): No suitable positive 'a' from roots. Trying backtracking.")
            return line_search_backtracking(x_k, b_k, objective_func, gradient_func)

    except Exception as e:
        # print(f"Error in line_search_fsolve: {e}. Trying backtracking.")
        return line_search_backtracking(x_k, b_k, objective_func, gradient_func)

def line_search_backtracking(x_k, b_k, objective_func, gradient_func, alpha=1.0, beta=0.5, c=1e-4):
    """Performs backtracking line search to satisfy Armijo condition."""
    f_k = objective_func(x_k)
    grad_k = gradient_func(x_k) # Gradient at x_k, not b_k
    slope = np.dot(grad_k, b_k) # Directional derivative using grad_k and b_k (b_k is -grad_k)

    if slope > -1e-9 : # If direction is not a descent direction (should not happen if b_k = -grad_k and grad_k is not zero)
        # This can happen if numerical precision issues make grad_k dot -grad_k appear positive.
        # print(f"Warning (backtracking): Direction b_k is not a descent direction (slope: {slope:.2e}). Using small step.")
        return 1e-5


    while objective_func(x_k + alpha * b_k) > f_k + c * alpha * slope:
        alpha *= beta
        if alpha < 1e-9: # Prevent infinitely small steps
            # print("Warning (backtracking): Alpha too small. Using minimal step.")
            return 1e-8
    return alpha

def line_search_minimize_scalar(x_k, b_k, objective_func, gradient_func_fallback): # Added gradient_func_fallback
    """Uses scipy.optimize.minimize_scalar for a more robust 1D search."""
    res = minimize_scalar(lambda a_val: phi(a_val, x_k, b_k, objective_func),
                          bounds=(0, 10), method='bounded', options={'xatol': 1e-6})
    if res.success and res.x > 1e-8:
        return res.x
    else:
        return line_search_backtracking(x_k, b_k, objective_func, gradient_func_fallback)


# --- Gradient Descent Algorithm ---
def gradient_descent(objective_func, gradient_func, func_name_str, x_initial, epsilon_val, max_iters_val, line_search_method_str):
    x_current = np.array(x_initial, dtype=float) # Ensure it's a float array for calculations
    history = {
        'x_values': [x_current.copy()],
        'f_values': [objective_func(x_current)],
        'errors': [],
        'gradients_norm': [],
        'step_lengths': []
    }
    iterations = 0

    print(f"\nOptimizing function: {func_name_str}")
    print(f"Initial point: x0 = {x_current}, f(x0) = {history['f_values'][0]:.6e}")
    print(f"Epsilon: {epsilon_val}, Max Iterations: {max_iters_val}, Line Search: {line_search_method_str}")
    print("-" * 90)
    print(f"{'Iter':<5} | {'x1':<12} | {'x2':<12} | {'f(x)':<15} | {'||grad||':<12} | {'Error':<12} | {'Step (a)':<10}")
    print("-" * 90)

    for k in range(max_iters_val):
        iterations = k + 1
        grad = gradient_func(x_current)
        grad_norm = np.linalg.norm(grad)
        history['gradients_norm'].append(grad_norm)

        b_k = -grad # Search direction

        if grad_norm < epsilon_val * 0.01: # If gradient is already very small
            print(f"\nGradient norm ({grad_norm:.2e}) is very small. Likely at a minimum or saddle point.")
            break
        
        if line_search_method_str == 'fsolve':
            a = line_search_fsolve(x_current, b_k, objective_func, gradient_func)
        elif line_search_method_str == 'backtracking':
            a = line_search_backtracking(x_current, b_k, objective_func, gradient_func)
        elif line_search_method_str == 'minimize_scalar':
            a = line_search_minimize_scalar(x_k, b_k, objective_func)
        else: # Default to backtracking
            a = line_search_backtracking(x_current, b_k, objective_func, gradient_func)

        history['step_lengths'].append(a)

        if a < 1e-9:
            print("\nStep length 'a' is too small (<1e-9). Stopping to prevent stagnation.")
            # No new x_value to add if we stop here before update
            iterations -=1 # Correct iteration count as this one didn't complete
            break

        x_next = x_current + a * b_k
        error = np.linalg.norm(x_next - x_current)

        history['x_values'].append(x_next.copy())
        history['f_values'].append(objective_func(x_next))
        history['errors'].append(error)
        
        if k < 15 or k % (max_iters_val // 20 if max_iters_val > 20 else 1) == 0 or error < epsilon_val:
            print(f"{iterations:<5} | {x_next[0]:<12.6f} | {x_next[1]:<12.6f} | {history['f_values'][-1]:<15.6e} | {grad_norm:<12.2e} | {error:<12.6e} | {a:<10.4e}")

        if error < epsilon_val:
            print("\nConvergence achieved: Error < Epsilon.")
            break
        
        x_current = x_next

    print("-" * 90)
    if iterations == max_iters_val and (not history['errors'] or history['errors'][-1] >= epsilon_val):
        print("\nMaximum iterations reached without full convergence based on error criteria.")
    
    final_x = history['x_values'][-1]
    final_f = history['f_values'][-1]
    
    print("\n--- Optimization Summary ---")
    print(f"Selected Function: {SELECTED_FUNCTION} ({func_name_str})")
    print(f"Initial Point x0: {x_initial}")
    print(f"Epsilon (Accuracy): {epsilon_val}")
    print(f"Line Search Method: {line_search_method_str}")
    print("-----------------------------")
    print(f"Iterations: {iterations}") # iterations completed
    print(f"Optimal x1: {final_x[0]:.8f}")
    print(f"Optimal x2: {final_x[1]:.8f}")
    print(f"Optimal f(x): {final_f:.8e}")
    if history['errors']:
        print(f"Final Error (||x_k+1 - x_k||): {history['errors'][-1]:.8e}")
    if history['gradients_norm']:
         print(f"Final Gradient Norm: {history['gradients_norm'][-1]:.8e}")

    return history

# --- Plotting ---
def plot_optimization_progress(history, func_name_str, x_initial_coords, objective_func_plot):
    if not history['errors'] and len(history['f_values']) <=1 :
        print("\nNot enough data to plot.")
        return

    num_plots = 2
    x_coords = np.array(history['x_values'])
    if x_coords.shape[0] > 1: # Only plot contour if we have iterations
        num_plots = 3 
        
    plt.figure(figsize=(8 * num_plots, 6))

    # Plot 1: Objective Function Value vs. Iterations
    ax1 = plt.subplot(1, num_plots, 1)
    ax1.plot(range(len(history['f_values'])), history['f_values'], marker='.', linestyle='-')
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Objective Function Value f(x)")
    ax1.set_title("f(x) vs. Iterations")
    if any(f_val > 0 for f_val in history['f_values']): # Use log if values are positive
        ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # Plot 2: Error vs. Iterations
    ax2 = plt.subplot(1, num_plots, 2)
    if history['errors']:
        # Iterations for error start from 1
        ax2.plot(range(1, len(history['errors']) + 1), history['errors'], marker='o', linestyle='-')
        ax2.axhline(y=EPSILON, color='r', linestyle='--', label=f'Epsilon = {EPSILON:.1e}')
        ax2.set_ylabel("Error (||x_k+1 - x_k||)")
        ax2.set_title("Error vs. Iterations")
        ax2.set_yscale('log') 
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No error data (0 iterations or error not calculated)", ha='center', va='center')
        ax2.set_title("Error vs. Iterations")
    ax2.set_xlabel("Iteration Number")
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    
    # Plot 3: Contour plot with optimization path
    if num_plots == 3:
        ax3 = plt.subplot(1, num_plots, 3)
        x1_vals = x_coords[:, 0]
        x2_vals = x_coords[:, 1]

        # Determine plot range based on trajectory and expected minimum
        # Expected mins for visual centering if needed:
        expected_mins = {1: (1,-1), 2: (1,1), 3: (1, 1/3)}
        exp_min = expected_mins.get(SELECTED_FUNCTION, x_initial_coords)

        x1_min, x1_max = min(x1_vals.min(), exp_min[0]), max(x1_vals.max(), exp_min[0])
        x2_min, x2_max = min(x2_vals.min(), exp_min[1]), max(x2_vals.max(), exp_min[1])
        
        margin_x = (x1_max - x1_min) * 0.2 + 0.1 # Add some margin
        margin_y = (x2_max - x2_min) * 0.2 + 0.1
        
        X1, X2 = np.meshgrid(np.linspace(x1_min - margin_x, x1_max + margin_x, 100),
                             np.linspace(x2_min - margin_y, x2_max + margin_y, 100))
        Z = objective_func_plot(np.array([X1, X2]))

        # Use log scale for levels if function values vary a lot (like Rosenbrock)
        contour_levels = 20
        if func_name_str.startswith("f(x) = 100"): # Heuristic for Rosenbrock-like
            min_f_val = max(1e-3, min(history['f_values'])) # Avoid log(0)
            max_f_val = max(history['f_values'])
            if max_f_val > min_f_val :
                contour_levels = np.logspace(np.log10(min_f_val), np.log10(max_f_val), 20)

        cp = ax3.contour(X1, X2, Z, levels=contour_levels, cmap='viridis')
        # plt.clabel(cp, inline=True, fontsize=8) # Can be too crowded
        plt.colorbar(cp, ax=ax3, label="f(x) value")
        
        ax3.plot(x1_vals, x2_vals, 'r.-', label='Optimization Path', markersize=3, linewidth=1)
        ax3.plot(x_initial_coords[0], x_initial_coords[1], 'bo', label='Start')
        ax3.plot(x1_vals[-1], x2_vals[-1], 'go', markersize=7, label='End')
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_title("Optimization Path on Contour Plot")
        ax3.legend()
        ax3.axis('equal') # Helps visualize banana shape of Rosenbrock
        ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Optimization for: {func_name_str}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- Main Execution with User Input ---
if __name__ == "__main__":
    print("--- Gradient Descent Optimization ---")
    
    # 1. Select Objective Function
    while True:
        try:
            SELECTED_FUNCTION = int(input(f"Select objective function (1, 2, or 3, default 2):\n"
                                          f"1: {FUNC_MAPPING[1]['name']}\n"
                                          f"2: {FUNC_MAPPING[2]['name']}\n"
                                          f"3: {FUNC_MAPPING[3]['name']}\nChoice: ") or "2")
            if SELECTED_FUNCTION in FUNC_MAPPING:
                objective_func = FUNC_MAPPING[SELECTED_FUNCTION]["func"]
                gradient_func = FUNC_MAPPING[SELECTED_FUNCTION]["grad"]
                func_name = FUNC_MAPPING[SELECTED_FUNCTION]["name"]
                break
            else:
                print("Invalid selection. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 2. Initial Point
    while True:
        try:
            x0_str = input("Enter initial point x0 as 'x1,x2' (e.g., '0,0', default '0,0'): ") or "0,0"
            X_INITIAL = [float(x.strip()) for x in x0_str.split(',')]
            if len(X_INITIAL) == 2:
                break
            else:
                print("Please enter two comma-separated values for x1 and x2.")
        except ValueError:
            print("Invalid format. Please use 'x1,x2'.")
    
    # 3. Epsilon
    while True:
        try:
            EPSILON = float(input("Enter convergence epsilon (e.g., 1e-4, default 1e-4): ") or "1e-4")
            if EPSILON > 0:
                break
            else:
                print("Epsilon must be positive.")
        except ValueError:
            print("Invalid number for epsilon.")

    # 4. Max Iterations
    while True:
        try:
            MAX_ITERATIONS = int(input("Enter maximum iterations (e.g., 1000, default 1000): ") or "1000")
            if MAX_ITERATIONS > 0:
                break
            else:
                print("Maximum iterations must be positive.")
        except ValueError:
            print("Invalid integer for maximum iterations.")

    # 5. Line Search Method
    while True:
        method_choice = input("Select line search method ('fsolve', 'backtracking', 'minimize_scalar', default 'backtracking'): ").lower() or "backtracking"
        if method_choice in ['fsolve', 'backtracking', 'minimize_scalar']:
            LINE_SEARCH_METHOD = method_choice
            break
        else:
            print("Invalid method. Choose 'fsolve', 'backtracking', or 'minimize_scalar'.")

    optimization_history = gradient_descent(objective_func, gradient_func, func_name, 
                                            X_INITIAL, EPSILON, MAX_ITERATIONS, LINE_SEARCH_METHOD)
    
    plot_optimization_progress(optimization_history, func_name, X_INITIAL, objective_func)

    print("\n--- Expected solutions (approximate) ---")
    if SELECTED_FUNCTION == 1:
        print("f1: x -> (1, -1), f(x) -> 0")
    elif SELECTED_FUNCTION == 2:
        print("f2 (Rosenbrock): x -> (1, 1), f(x) -> 0")
    elif SELECTED_FUNCTION == 3:
        print("f3: x -> (1, 1/3), f(x) -> 0")