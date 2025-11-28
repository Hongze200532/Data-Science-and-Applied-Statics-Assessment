import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
# Data Set
n = 100  # Number of Geiger counters
r = 0.4  # Detection rate (per second)
t = 10  # Time interval (seconds)
lambda_param = r * t  # Poisson distribution parameter = 4


# ==================== Problem (a): Simulate Y (α=1) ====================
def problem_a():
    """
    Problem (a): Simulate 1000 samples of Y when α=1
    Distribution choice: Since particles are detected independently at a constant rate, Xi ~ Poisson(λ = r t = 4)
    When α=1, all Wi=1, so Y = ΣXi ~ Poisson(nλ = 400)
    """
    print("=" * 60)
    print("Problem (a): Simulate Y (α=1)")
    print("=" * 60)

    n_samples = 1000

    # Simulate detection counts for each counter Xi ~ Poisson(4)
    X = np.random.poisson(lambda_param, size=(n_samples, n))

    # Simulate successful transmission Wi ~ Bernoulli(α=1), i.e., all Wi=1
    W = np.random.binomial(1, 1.0, size=(n_samples, n))

    # Compute Y = Σ(Wi * Xi)
    Y = np.sum(W * X, axis=1)

    # Plot empirical distribution and CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Empirical distribution (histogram)
    axes[0].hist(Y, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Y value')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Empirical distribution of Y (α=1, n={n}, λ={lambda_param})')
    axes[0].grid(True, alpha=0.3)

    # Overlay theoretical Poisson distribution
    y_range = np.arange(Y.min(), Y.max() + 1)
    poisson_theory = stats.poisson.pmf(y_range, n * lambda_param)
    axes[0].plot(y_range, poisson_theory, 'r-', linewidth=2, label=f'Poisson(λ={n * lambda_param})')
    axes[0].legend()

    # Right: Empirical CDF
    sorted_Y = np.sort(Y)
    empirical_cdf = np.arange(1, len(sorted_Y) + 1) / len(sorted_Y)
    axes[1].plot(sorted_Y, empirical_cdf, 'b-', linewidth=2, label='Empirical CDF')

    # Overlay theoretical CDF
    y_range_cdf = np.arange(0, Y.max() + 1)
    poisson_cdf = stats.poisson.cdf(y_range_cdf, n * lambda_param)
    axes[1].plot(y_range_cdf, poisson_cdf, 'r--', linewidth=2, label=f'Poisson(λ={n * lambda_param}) CDF')
    axes[1].set_xlabel('Y value')
    axes[1].set_ylabel('Cumulative probability')
    axes[1].set_title('Empirical cumulative distribution function of Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_a_distribution.png', dpi=300, bbox_inches='tight')
    print("Statistics of Y:")
    print(f"  Mean: {np.mean(Y):.2f} (theoretical value: {n * lambda_param})")
    print(f"  Standard deviation: {np.std(Y):.2f} (theoretical value: {np.sqrt(n * lambda_param):.2f})")
    print(f"  Minimum: {np.min(Y)}, Maximum: {np.max(Y)}")
    print("Figure saved as: problem_a_distribution.png\n")

    return Y


# ==================== Problem (b): Approximate simulation models ====================
def problem_b(Y_exact):
    """
    Problem (b): Implement two approximate simulation models
    1. Normal approximation: According to the central limit theorem, Y is approximately normal
    2. Poisson approximation: Directly use Poisson(nλ) distribution
    """
    print("=" * 60)
    print("Problem (b): Approximate simulation models")
    print("=" * 60)

    n_samples = 1000

    # Approximate model 1: Normal approximation
    # By the central limit theorem, Y ~ N(nλ, nλ)
    mean_normal = n * lambda_param
    std_normal = np.sqrt(n * lambda_param)
    Y_normal = np.random.normal(mean_normal, std_normal, n_samples)
    Y_normal = np.maximum(0, np.round(Y_normal))  # Ensure non-negative integers

    # Approximate model 2: Poisson approximation
    # Directly use Poisson(nλ) distribution
    Y_poisson = np.random.poisson(n * lambda_param, n_samples)

    # Plot CDF comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # CDF of exact model
    sorted_Y_exact = np.sort(Y_exact)
    empirical_cdf_exact = np.arange(1, len(sorted_Y_exact) + 1) / len(sorted_Y_exact)
    ax.plot(sorted_Y_exact, empirical_cdf_exact, 'b-', linewidth=2, label='Exact model (a)')

    # Normal approximation CDF
    sorted_Y_normal = np.sort(Y_normal)
    empirical_cdf_normal = np.arange(1, len(sorted_Y_normal) + 1) / len(sorted_Y_normal)
    ax.plot(sorted_Y_normal, empirical_cdf_normal, 'g--', linewidth=2, label='Approximate model 1: Normal approximation')

    # Poisson approximation CDF
    sorted_Y_poisson = np.sort(Y_poisson)
    empirical_cdf_poisson = np.arange(1, len(sorted_Y_poisson) + 1) / len(sorted_Y_poisson)
    ax.plot(sorted_Y_poisson, empirical_cdf_poisson, 'r:', linewidth=2, label='Approximate model 2: Poisson approximation')

    ax.set_xlabel('Y value')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('Comparison of CDFs of approximate models (α=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_b_approximations.png', dpi=300, bbox_inches='tight')

    # Accuracy metrics (KS statistic)
    from scipy.stats import ks_2samp

    ks_normal, p_normal = ks_2samp(Y_exact, Y_normal)
    ks_poisson, p_poisson = ks_2samp(Y_exact, Y_poisson)

    print("Accuracy analysis of approximate models:")
    print(f"  Normal approximation KS statistic: {ks_normal:.4f}, p-value: {p_normal:.4f}")
    print(f"  Poisson approximation KS statistic: {ks_poisson:.4f}, p-value: {p_poisson:.4f}")
    print("\nDiscussion:")
    print("  1. Poisson approximation: Since Y is the sum of 100 independent Poisson(4) random variables,")
    print("     theoretically Y ~ Poisson(400), so the Poisson approximation is extremely accurate.")
    print("  2. Normal approximation: By the central limit theorem, when n is large,")
    print("     Y is approximately normal N(400, 20²). For large λ, the Poisson distribution is also close to normal.")
    print("     Both approximations are accurate, but the Poisson approximation is more precise because it is the exact distribution.")
    print("Figure saved as: problem_b_approximations.png\n")

    return Y_normal, Y_poisson


# ==================== Problem (c): Hypothesis testing ====================
def problem_c(Y):
    """
    Problem (c): Hypothesis testing
    H0: μ ≤ 400 vs H1: μ > 400
    Significance level: 5%
    """
    print("=" * 60)
    print("Problem (c): Hypothesis testing")
    print("=" * 60)

    mu0 = 400  # Value under null hypothesis
    alpha_level = 0.05  # Significance level

    # One-sample t-test (right-tailed)
    t_stat, p_value = stats.ttest_1samp(Y, mu0, alternative='greater')

    # Ensure p_value is a scalar
    p_value_float = float(np.asarray(p_value).item())

    # z-test can also be used (since sample size is large)
    sample_mean = np.mean(Y)
    sample_std = np.std(Y, ddof=1)
    n_samples = len(Y)
    z_stat = (sample_mean - mu0) / (sample_std / np.sqrt(n_samples))
    z_p_value = 1 - stats.norm.cdf(z_stat)

    print(f"Null hypothesis H0: μ ≤ {mu0}")
    print(f"Alternative hypothesis H1: μ > {mu0}")
    print(f"Significance level: {alpha_level * 100}%\n")

    print("Sample statistics:")
    print(f"  Sample mean: {sample_mean:.2f}")
    print(f"  Sample standard deviation: {sample_std:.2f}")
    print(f"  Sample size: {n_samples}\n")

    print("t-test results:")
    print(f"  t statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value_float:.4f}")
    print(f"  Conclusion: {'Reject H0' if p_value_float < alpha_level else 'Do not reject H0'}\n")

    print("z-test results (large-sample approximation):")
    print(f"  z statistic: {z_stat:.4f}")
    print(f"  p-value: {z_p_value:.4f}")
    print(f"  Conclusion: {'Reject H0' if z_p_value < alpha_level else 'Do not reject H0'}\n")

    print("Discussion on using hypothesis tests on computer-simulated data:")
    print("  1. Simulated data are deterministic (given a random seed), but statistical inference is still valid.")
    print("  2. Hypothesis testing can help verify whether the simulation model matches theoretical expectations.")
    print("  3. In simulations, we can generate many samples, leading to high test power.")
    print("  4. Note: Simulated data may not satisfy certain assumptions (e.g., independence),")
    print("     but in this problem the counters are independent, so the assumptions hold.")
    print("  5. If the p-value is very small, it may indicate an error in the simulation implementation or an inaccurate theoretical model.\n")

    return t_stat, p_value


# ==================== Problem (d): Discussion (text explanation) ====================
def problem_d():
    """
    Problem (d): Discuss how the answer to (b) changes when α=0.5 while keeping r=0.4 unchanged
    """
    print("=" * 60)
    print("Problem (d): Discussion (α=0.5, r=0.4)")
    print("=" * 60)
    print("""
    When α=0.5, each counter's report has only a 50% probability of being successfully transmitted.

    Impact on the approximate models in problem (b):

    1. Change in exact distribution:
       - When α=0.5, Y = Σ(Wi * Xi), where Wi ~ Bernoulli(0.5)
       - Expectation of each Wi * Xi: E[Wi * Xi] = E[Wi] * E[Xi] = 0.5 * 4 = 2
       - Expectation of Y: E[Y] = n * 0.5 * 4 = 200
       - Variance of Y: Var(Y) = n * [Var(Wi*Xi)] = n * [E[Wi]Var(Xi) + Var(Wi)E[Xi]²]
                    = n * [0.5*4 + 0.25*16] = n * [2 + 4] = 600

    2. Normal approximation:
       - Y is approximately ~ N(200, √600²) = N(200, 24.49²)
       - Still valid, because the central limit theorem still applies
       - But the variance increases because transmission failures introduce extra variability

    3. Poisson approximation:
       - Due to the presence of Wi, Y is no longer a simple Poisson distribution
       - But it can be approximated as Poisson(200), since E[Y] = 200
       - The accuracy of this approximation decreases because the actual variance (600) is greater than the Poisson variance (200)
       - The actual distribution is "zero-inflated" because some counters' reports may be completely lost

    4. Conclusion:
       - The normal approximation is still applicable, but the parameters need to be adjusted
       - The accuracy of the Poisson approximation decreases significantly because the actual distribution deviates from Poisson
       - More complex approximations may be needed, such as a negative binomial mixture or other mixture distributions
    """)


# ==================== Problem (e): Bimodal simulation ====================
def problem_e():
    """
    Problem (e): When α=0.5, with 50 counters having r=0.4 and 50 counters having r=1
    Simulate 1000 seconds of count reports
    """
    print("=" * 60)
    print("Problem (e): Bimodal simulation (α=0.5, mixed r)")
    print("=" * 60)

    alpha = 0.5
    t_simulation = 1000  # Simulate 1000 seconds

    # 50 counters with r=0.4, 50 counters with r=1
    r_values = np.concatenate([np.full(50, 0.4), np.full(50, 1.0)])
    lambda_values = r_values * t  # Poisson parameter for each counter

    # Store counts at each time point
    counts_per_second = []

    for second in range(t_simulation):
        # Detection counts of each counter
        X = np.array([np.random.poisson(lam) for lam in lambda_values])

        # Successful/failed transmission
        W = np.random.binomial(1, alpha, size=n)

        # Total count in this second
        total_count = np.sum(W * X)
        counts_per_second.append(total_count)

    counts_per_second = np.array(counts_per_second)

    # Plot distribution and CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Empirical distribution
    axes[0].hist(counts_per_second, bins=50, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Counts per second')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Bimodal distribution (α=0.5, 50×r=0.4 + 50×r=1.0)')
    axes[0].grid(True, alpha=0.3)

    # Right: Empirical CDF
    sorted_counts = np.sort(counts_per_second)
    empirical_cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1].plot(sorted_counts, empirical_cdf, 'b-', linewidth=2)
    axes[1].set_xlabel('Counts per second')
    axes[1].set_ylabel('Cumulative probability')
    axes[1].set_title('Empirical cumulative distribution function')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_e_bimodal.png', dpi=300, bbox_inches='tight')

    print("Statistics:")
    print(f"  Mean: {np.mean(counts_per_second):.2f}")
    print(f"  Standard deviation: {np.std(counts_per_second):.2f}")
    print(f"  Minimum: {np.min(counts_per_second)}, Maximum: {np.max(counts_per_second)}")
    print(f"  Skewness: {stats.skew(counts_per_second):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(counts_per_second):.4f}")
    print("Figure saved as: problem_e_bimodal.png\n")

    return counts_per_second


# ==================== Problem (f): Bimodal distribution fitting ====================
def bimodal_log_likelihood(params, data):
    """
    Negative log-likelihood function for a bimodal normal distribution
    params: [mu1, sigma1, mu2, sigma2, w]
    w is the weight of the first component
    """
    mu1, sigma1, mu2, sigma2, w = params

    # Ensure parameters are in a reasonable range
    if sigma1 <= 0 or sigma2 <= 0 or w < 0 or w > 1:
        return 1e10

    # Likelihood of the mixture distribution
    likelihood = (w * stats.norm.pdf(data, mu1, sigma1) +
                  (1 - w) * stats.norm.pdf(data, mu2, sigma2))

    # Avoid numerical issues
    likelihood = np.clip(likelihood, 1e-10, None)

    return -np.sum(np.log(likelihood))


def problem_f(counts_data):
    """
    Problem (f): Use MLE to fit a bimodal normal distribution
    """
    print("=" * 60)
    print("Problem (f): Bimodal distribution fitting (MLE)")
    print("=" * 60)

    # Initial parameter estimates (based on data quantiles)
    data_sorted = np.sort(counts_data)

    # Use quantiles to initialize the two modes
    mu1_init = np.percentile(data_sorted, 25)
    mu2_init = np.percentile(data_sorted, 75)
    sigma1_init = np.std(data_sorted[data_sorted < np.median(data_sorted)])
    sigma2_init = np.std(data_sorted[data_sorted >= np.median(data_sorted)])
    w_init = 0.5

    initial_params = [mu1_init, sigma1_init, mu2_init, sigma2_init, w_init]

    print("Initial parameter estimates:")
    print(f"  μ1: {mu1_init:.2f}, σ1: {sigma1_init:.2f}")
    print(f"  μ2: {mu2_init:.2f}, σ2: {sigma2_init:.2f}")
    print(f"  Weight w: {w_init:.2f}\n")

    # MLE optimization
    bounds = [(None, None), (1e-3, None), (None, None), (1e-3, None), (0, 1)]
    result = minimize(bimodal_log_likelihood, initial_params, args=(counts_data,),
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})

    mu1_est, sigma1_est, mu2_est, sigma2_est, w_est = result.x

    print("MLE estimation results:")
    print(f"  μ1: {mu1_est:.2f}, σ1: {sigma1_est:.2f}")
    print(f"  μ2: {mu2_est:.2f}, σ2: {sigma2_est:.2f}")
    print(f"  Weight w: {w_est:.4f}")
    print(f"  Negative log-likelihood: {result.fun:.2f}\n")

    # Plot fitted results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram + fitted distribution
    axes[0].hist(counts_data, bins=50, density=True, alpha=0.7,
                 color='skyblue', edgecolor='black', label='Data')

    x_range = np.linspace(counts_data.min(), counts_data.max(), 1000)
    fitted_pdf = (w_est * stats.norm.pdf(x_range, mu1_est, sigma1_est) +
                  (1 - w_est) * stats.norm.pdf(x_range, mu2_est, sigma2_est))
    axes[0].plot(x_range, fitted_pdf, 'r-', linewidth=2, label='Fitted bimodal distribution')

    # Plot the two components
    comp1 = w_est * stats.norm.pdf(x_range, mu1_est, sigma1_est)
    comp2 = (1 - w_est) * stats.norm.pdf(x_range, mu2_est, sigma2_est)
    axes[0].plot(x_range, comp1, 'g--', linewidth=1.5, alpha=0.7, label=f'Component 1 (w={w_est:.3f})')
    axes[0].plot(x_range, comp2, 'orange', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=f'Component 2 (w={1 - w_est:.3f})')

    axes[0].set_xlabel('Counts per second')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Bimodal distribution fitting results')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: CDF comparison
    sorted_data = np.sort(counts_data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1].plot(sorted_data, empirical_cdf, 'b-', linewidth=2, label='Empirical CDF')

    fitted_cdf = (w_est * stats.norm.cdf(x_range, mu1_est, sigma1_est) +
                  (1 - w_est) * stats.norm.cdf(x_range, mu2_est, sigma2_est))
    axes[1].plot(x_range, fitted_cdf, 'r--', linewidth=2, label='Fitted CDF')

    axes[1].set_xlabel('Counts per second')
    axes[1].set_ylabel('Cumulative probability')
    axes[1].set_title('Comparison of cumulative distribution functions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_f_fitting.png', dpi=300, bbox_inches='tight')

    # KS test
    from scipy.stats import ks_2samp
    # Generate samples from the fitted distribution
    n_samples_fit = 10000
    u = np.random.rand(n_samples_fit)
    samples_fit = np.where(u < w_est,
                           np.random.normal(mu1_est, sigma1_est, n_samples_fit),
                           np.random.normal(mu2_est, sigma2_est, n_samples_fit))

    ks_stat, ks_p = ks_2samp(counts_data, samples_fit)

    print("Goodness-of-fit test:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_p:.4f}\n")

    print("Discussion of adequacy of the fitted distribution:")
    print("  1. The bimodal normal mixture model can capture the two main modes in the data,")
    print("     corresponding to the two groups of counters with r=0.4 and r=1.0.")
    print("  2. However, the actual data come from a mixture of Poisson distributions, not normal distributions.")
    print("  3. When parameters are large, Poisson distributions are approximately normal, so the fit can be reasonable.")
    print("  4. But note that:")
    print("     - The actual distribution is discrete (count data), while the normal distribution is continuous")
    print("     - The actual distribution may have more skewness and tail behavior")
    print("     - A better choice may be a mixture of Poisson distributions or a mixture of negative binomial distributions")
    print("  5. If the KS test p-value is relatively large, the fit is statistically acceptable,")
    print("     but it may still not be the optimal model.")
    print("Figure saved as: problem_f_fitting.png\n")

    return result.x


# ==================== Main function ====================
def main():
    """Run all problems"""
    print("\n" + "=" * 60)
    print("Geiger counter network simulation - complete solution")
    print("=" * 60 + "\n")
    # Set random seed for reproducibility
    np.random.seed(42)
    # Problem (a)
    Y_a = problem_a()
    # Problem (b)
    Y_normal, Y_poisson = problem_b(Y_a)
    # Problem (c)
    problem_c(Y_a)
    # Problem (d)
    problem_d()
    # Problem (e)
    counts_e = problem_e()
    # Problem (f)
    problem_f(counts_e)

    print("=" * 60)
    print("All problems completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
