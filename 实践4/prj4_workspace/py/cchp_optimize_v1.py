import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve # For solving phi'(a) = 0

# --- Configuration ---
SELECTED_FUNCTION = 2  # Choose 1, 2, or 3
X_INITIAL = np.array([0.0, 0.0])
EPSILON = 1.0e-4
MAX_ITERATIONS = 1000

# --- Objective Functions ---
def f1(x):
    """f(x) = 10*(x1-1)**2 + (x2+1)**4"""
    x1, x2 = x
    return 10 * (x1 - 1)**2 + (x2 + 1)**4

def f2(x):
    """f(x) = 100*(x1**2 - x2)**2 + (x1 - 1)**2"""
    x1, x2 = x
    return 100 * (x1**2 - x2)**2 + (x1 - 1)**2

def f3(x):
    """f(x) = 100*(x1**2 - 3*x2)**2 + (x1 - 1)**2"""
    x1, x2 = x
    return 100 * (x1**2 - 3*x2)**2 + (x1 - 1)**2

# --- Gradient Functions ---
def grad_f1(x):
    """Gradient for f1(x)"""
    x1, x2 = x
    df_dx1 = 20 * (x1 - 1)
    df_dx2 = 4 * (x2 + 1)**3
    return np.array([df_dx1, df_dx2])

def grad_f2(x):
    """Gradient for f2(x)"""
    x1, x2 = x
    df_dx1 = 200 * (x1**2 - x2) * (2 * x1) + 2 * (x1 - 1)
    df_dx2 = -200 * (x1**2 - x2)
    return np.array([df_dx1, df_dx2])

def grad_f3(x):
    """Gradient for f3(x)"""
    x1, x2 = x
    df_dx1 = 200 * (x1**2 - 3*x2) * (2 * x1) + 2 * (x1 - 1)
    df_dx2 = -200 * (x1**2 - 3*x2) * 3 # -600 * (x1**2 - 3*x2)
    return np.array([df_dx1, df_dx2])

# --- Select active function and gradient ---
if SELECTED_FUNCTION == 1:
    objective_function = f1
    gradient_function = grad_f1
    func_name = "f(x) = 10*(x1-1)^2 + (x2+1)^4"
elif SELECTED_FUNCTION == 2:
    objective_function = f2
    gradient_function = grad_f2
    func_name = "f(x) = 100*(x1^2 - x2)^2 + (x1-1)^2"
elif SELECTED_FUNCTION == 3:
    objective_function = f3
    gradient_function = grad_f3
    func_name = "f(x) = 100*(x1^2 - 3*x2)^2 + (x1-1)^2"
else:
    raise ValueError("Invalid SELECTED_FUNCTION. Choose 1, 2, or 3.")

# --- Line Search: Find step length 'a' ---
def phi_prime(a, x_k, b_k, objective_func, gradient_func_phi):
    """
    Calculates the derivative of phi(a) = f(x_k + a*b_k) with respect to 'a'.
    phi'(a) = grad_f(x_k + a*b_k) . b_k  (dot product)
    """
    x_new = x_k + a * b_k
    grad_at_x_new = gradient_func_phi(x_new)
    return np.dot(grad_at_x_new, b_k)

def line_search_step_length(x_k, b_k):
    """
    Finds the step length 'a' by solving phi'(a) = 0.
    Uses scipy.optimize.fsolve to find the root.
    """
    # fsolve needs an initial guess for 'a'.
    # A small positive value is usually a good start.
    # We expect 'a' to be positive because b_k is a descent direction.
    initial_guess_a = 0.01
    
    # Try to find a positive root.
    # We pass current x_k, search direction b_k, and the gradient function
    # to phi_prime.
    try:
        # We search for 'a' that makes phi_prime zero
        # args passes additional arguments to phi_prime
        a_optimal_roots = fsolve(phi_prime, initial_guess_a, args=(x_k, b_k, objective_function, gradient_function), xtol=1e-5)
        
        # fsolve can return multiple roots or non-positive roots.
        # We want the smallest positive 'a' that represents a minimum.
        # A robust way would be to evaluate f(x_k + a*b_k) for valid 'a's.
        # For simplicity here, we take the first positive result if available.
        # If fsolve returns an array, we look for positive values.
        positive_roots = [r for r in np.atleast_1d(a_optimal_roots) if r > 1e-7] # Avoid zero or negative step
        
        if not positive_roots:
            # Fallback if no positive root found or fsolve fails to converge to a good one.
            # This might happen if the initial guess is bad or the function is tricky.
            # A very small fixed step or a more robust 1D minimizer could be used.
            # print(f"Warning: Line search did not find a suitable positive 'a'. Roots: {a_optimal_roots}. Using fallback.")
            # Fallback: Try a very small step or a simple backtracking if fsolve fails
            # For this assignment, we'll try a simpler approach if fsolve is problematic:
            # test a few small 'a' values or use a very small 'a'
            # This specific requirement "令一阶导数为0" makes this part tricky if fsolve is not robust enough
            # or if multiple roots exist.
            
            # A simple heuristic: if fsolve gives negative or zero, try a small positive a
            if len(a_optimal_roots) > 0 and a_optimal_roots[0] <= 1e-7:
                 return 1e-3 # Small fixed step as fallback
            elif len(positive_roots) > 0:
                # If multiple positive roots, picking the smallest one is a heuristic
                # that often works for the first minimum along the direction.
                return min(positive_roots)
            else: # No roots or only non-positive roots
                return 1e-3 

        a_optimal = min(positive_roots) # Smallest positive root
        
        # Ensure 'a' is not excessively large, which can happen if phi_prime is very flat
        if a_optimal > 10.0: # Heuristic upper bound for 'a'
            # print(f"Warning: Large step length a={a_optimal:.2e} found. Capping to 1.0.")
            return 1.0
        
        return a_optimal

    except Exception as e:
        print(f"Error in line_search_step_length with fsolve: {e}. Using fallback step.")
        # Fallback to a very small step if fsolve has issues
        return 1e-4


# --- Gradient Descent Algorithm ---
def gradient_descent():
    x_current = X_INITIAL.copy()
    history = {
        'x_values': [x_current.copy()],
        'f_values': [objective_function(x_current)],
        'errors': [],
        'gradients': [],
        'step_lengths': []
    }
    iterations = 0

    print(f"Optimizing function: {func_name}")
    print(f"Initial point: x0 = {x_current}, f(x0) = {history['f_values'][0]:.6e}")
    print("-" * 70)
    print(f"{'Iter':<5} | {'x1':<12} | {'x2':<12} | {'f(x)':<15} | {'Error':<12} | {'Step (a)':<10}")
    print("-" * 70)

    for k in range(MAX_ITERATIONS):
        iterations = k + 1
        grad = gradient_function(x_current)
        history['gradients'].append(grad)

        # Search direction b_k = -grad
        b_k = -grad

        # Check for near-zero gradient (already at/near minimum)
        if np.linalg.norm(grad) < EPSILON * 0.1: # More stringent check for gradient
            print(f"\nGradient norm ({np.linalg.norm(grad):.2e}) is very small. Likely at a minimum.")
            break
            
        # Line search for step length 'a'
        a = line_search_step_length(x_current, b_k)
        history['step_lengths'].append(a)

        if a < 1e-9: # If step length becomes too small
            print("\nStep length 'a' is too small. Stopping to prevent stagnation.")
            break

        x_next = x_current + a * b_k
        
        error = np.linalg.norm(x_next - x_current)

        history['x_values'].append(x_next.copy())
        history['f_values'].append(objective_function(x_next))
        history['errors'].append(error)
        
        if k < 10 or k % (MAX_ITERATIONS // 20 if MAX_ITERATIONS > 20 else 1) == 0 or error < EPSILON : # Print more often for first few iterations
             print(f"{iterations:<5} | {x_next[0]:<12.6f} | {x_next[1]:<12.6f} | {history['f_values'][-1]:<15.6e} | {error:<12.6e} | {a:<10.4e}")

        if error < EPSILON:
            print("\nConvergence achieved: Error < Epsilon.")
            break
        
        x_current = x_next

    print("-" * 70)
    if iterations == MAX_ITERATIONS and error >= EPSILON:
        print("\nMaximum iterations reached without full convergence.")
    
    final_x = history['x_values'][-1]
    final_f = history['f_values'][-1]
    
    print("\n--- Optimization Summary ---")
    print(f"Selected Function: {SELECTED_FUNCTION} ({func_name})")
    print(f"Initial Point x0: {X_INITIAL}")
    print(f"Epsilon (Accuracy): {EPSILON}")
    print("-----------------------------")
    print(f"Iterations: {iterations}")
    print(f"Optimal x1: {final_x[0]:.8f}")
    print(f"Optimal x2: {final_x[1]:.8f}")
    print(f"Optimal f(x): {final_f:.8e}")
    if history['errors']:
        print(f"Final Error: {history['errors'][-1]:.8e}")
    else:
        print("Final Error: N/A (0 iterations or immediate convergence)")
    
    return history

# --- Plotting ---
def plot_results(history):
    if not history['errors']:
        print("\nNo error history to plot (0 iterations or immediate convergence).")
        return

    # Plot 1: Error vs. Iterations
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Iterations for error start from 1 (error is between k and k+1)
    plt.plot(range(1, len(history['errors']) + 1), history['errors'], marker='o', linestyle='-')
    plt.axhline(y=EPSILON, color='r', linestyle='--', label=f'Epsilon = {EPSILON}')
    plt.xlabel("Iteration Number")
    plt.ylabel("Error (||x_k+1 - x_k||)")
    plt.title("Error vs. Iterations")
    plt.yscale('log') # Log scale often useful for errors
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # Plot 2: Objective Function Value vs. Iterations
    plt.subplot(1, 2, 2)
    plt.plot(range(len(history['f_values'])), history['f_values'], marker='.', linestyle='-')
    plt.xlabel("Iteration Number")
    plt.ylabel("Objective Function Value f(x)")
    plt.title("f(x) vs. Iterations")
    plt.yscale('log') # Log scale can be useful here too
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.suptitle(f"Optimization for Function {SELECTED_FUNCTION}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

    # Optional: Plot x1 and x2 values over iterations
    x_vals_arr = np.array(history['x_values'])
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(x_vals_arr)), x_vals_arr[:, 0], label='x1', marker='.')
    plt.plot(range(len(x_vals_arr)), x_vals_arr[:, 1], label='x2', marker='.')
    plt.xlabel("Iteration Number")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Values (x1, x2) vs. Iterations")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    optimization_history = gradient_descent()
    plot_results(optimization_history)

    print("\nExpected solutions (approximate, may vary based on problem and parameters):")
    if SELECTED_FUNCTION == 1:
        print("f1: x -> (1, -1), f(x) -> 0")
    elif SELECTED_FUNCTION == 2:
        print("f2: x -> (1, 1), f(x) -> 0 (Rosenbrock function)")
    elif SELECTED_FUNCTION == 3:
        print("f3: x -> (1, 1/3), f(x) -> 0")