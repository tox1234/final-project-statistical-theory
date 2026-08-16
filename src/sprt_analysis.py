import numpy as np
import matplotlib.pyplot as plt

def run_wald_sequential_test():
    """
    Evaluates Wald's Sequential Probability Ratio Test (SPRT).
    Generates stochastic trajectories determining strict optimal stopping 
    times (N) to minimize Expected Sample Numbers (ASN).
    """
    print("\n--- Part G: Wald's SPRT (Sequential Most Powerful Method) ---")
    
    # Defining a priori Error Tolerance Boundaries
    alpha = 0.05  # False Alarm Probability (Type I)
    beta = 0.10   # Missed Hazard Probability (Type II)
    
    # Theoretical Means
    mu0 = 0.90    # Safe equilibrium
    mu1 = 1.20    # Hazardous threshold
    
    # Historical Process Variance
    sigma2 = 0.0625  
    sigma = np.sqrt(sigma2)
    s_drift = (mu0 + mu1) / 2.0
    
    # Exact Boundary Functions
    a_boundary = (sigma2 / (mu1 - mu0)) * np.log(beta / (1 - alpha))
    b_boundary = (sigma2 / (mu1 - mu0)) * np.log((1 - beta) / alpha)
    
    print(f"Acceptance Bound (a) : {a_boundary:.4f}")
    print(f"Rejection Bound (b)  : {b_boundary:.4f}")
    print(f"Expected Drift (s)   : {s_drift:.4f}")
    
    def execute_random_walk(true_mean, iterations=1000):
        stopping_times = []
        decisions = []
        for _ in range(iterations):
            cum_sum = 0.0
            n = 0
            while True:
                n += 1
                x = np.random.normal(true_mean, sigma)
                cum_sum += (x - s_drift)
                if cum_sum <= a_boundary:
                    stopping_times.append(n)
                    decisions.append(0)  # Accepted H0
                    break
                elif cum_sum >= b_boundary:
                    stopping_times.append(n)
                    decisions.append(1)  # Rejected H0 -> Hazardous
                    break
        return np.mean(stopping_times), np.mean(decisions)

    np.random.seed(42)
    asn_h0, err_h0 = execute_random_walk(mu0)
    asn_h1, err_h1 = execute_random_walk(mu1)
    
    print(f"\nSimulation ASN under Normal State (\u03bc0) : {asn_h0:.2f} reads (False Positives: {err_h0:.4f})")
    print(f"Simulation ASN under Hazard State (\u03bc1) : {asn_h1:.2f} reads (True Positives : {err_h1:.4f})")
    
    np.random.seed(111)
    trajectory = [0.0]
    c_sum = 0.0
    while True:
        c_sum += (np.random.normal(mu1, sigma) - s_drift)
        trajectory.append(c_sum)
        if c_sum <= a_boundary or c_sum >= b_boundary:
            break
            
    plt.figure(figsize=(10, 5))
    plt.plot(trajectory, marker='o', color='indigo', lw=2, label='Log-Likelihood Ratio Cumulative Sum ($S_n$)')
    plt.axhline(y=b_boundary, color='darkred', linestyle='--', label=f'Upper Threshold (Trigger Alert) = {b_boundary:.2f}')
    plt.axhline(y=a_boundary, color='darkgreen', linestyle='--', label=f'Lower Threshold (Safe Status) = {a_boundary:.2f}')
    
    plt.title("Sequential Monitoring Trajectory via Wald's SPRT Framework")
    plt.xlabel("Continuous Sequence Index ($N$)")
    plt.ylabel("Cumulative Sum Statistic ($S_n$)")
    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('sprt_trajectory.png', dpi=300)
    plt.close()
    print("Sequential walk saved as 'sprt_trajectory.png'.")
