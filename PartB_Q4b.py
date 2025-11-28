"""
Question 4 Part (b): Evaluation of Statistical Reports
Evaluate three example statistical reports
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

def example_i():
    """
    Example (i): Gamma distribution model, reciprocal(y) ~ x + z
    Issue: incorrect interpretation of reciprocal(y)
    """
    print("\n" + "=" * 60)
    print("Example (i): Gamma distribution model analysis")
    print("=" * 60)

    # Generate toy data
    n = 2000
    np.random.seed(123)

    x = np.random.uniform(0, 5, n)
    z = np.ones(n)  # Fix z = 1
    epsilon = np.random.gamma(shape=2, scale=0.1, size=n)  # Gamma error

    # According to the model: reciprocal(y) = 2 + 1.5*x + 0.6*z + 0.5*x*z + epsilon
    reciprocal_y = 2 + 1.5 * x + 0.6 * z + 0.5 * x * z + epsilon
    y = 1 / reciprocal_y  # Transform back to original y

    # Fit Gamma GLM (using reciprocal link)
    def model_func(x_data, intercept, coef_x, coef_z, coef_interaction):
        z_fixed = 1
        return intercept + coef_x * x_data + coef_z * z_fixed + coef_interaction * x_data * z_fixed

    popt, pcov = curve_fit(model_func, x, reciprocal_y)

    print("\nFitting results:")
    print(f"Intercept: {popt[0]:.4f} (reported value: 2.0)")
    print(f"Coefficient for x: {popt[1]:.4f} (reported value: 1.5)")
    print(f"Coefficient for z: {popt[2]:.4f} (reported value: 0.6)")
    print(f"Interaction coefficient: {popt[3]:.4f} (reported value: 0.5)")

    # Plot analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: x vs reciprocal(y)
    axes[0, 0].scatter(x, reciprocal_y, alpha=0.3, s=10)
    x_sorted = np.sort(x)
    y_pred_recip = model_func(x_sorted, *popt)
    axes[0, 0].plot(x_sorted, y_pred_recip, 'r-', linewidth=2, label='Fitted line')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('1/y (reciprocal)')
    axes[0, 0].set_title('x vs 1/y (model fit)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: x vs y (original scale)
    axes[0, 1].scatter(x, y, alpha=0.3, s=10)
    y_pred = 1 / y_pred_recip
    axes[0, 1].plot(x_sorted, y_pred, 'r-', linewidth=2, label='Transformed fit')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y (original scale)')
    axes[0, 1].set_title('x vs y (original scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Illustrate the misinterpretation
    # When z=1 is fixed, increasing x by 1 unit increases reciprocal(y) by 1.5
    # but the change in y is not linear
    x_test = np.array([1, 2, 3, 4, 5])
    z_test = np.ones_like(x_test)
    recip_y_at_x = 2 + 1.5 * x_test + 0.6 * z_test + 0.5 * x_test * z_test
    y_at_x = 1 / recip_y_at_x

    axes[1, 0].plot(x_test, recip_y_at_x, 'b-o', label='Change in 1/y', linewidth=2)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('1/y')
    axes[1, 0].set_title('Effect of x on 1/y when z=1 is fixed')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_test, y_at_x, 'r-o', label='Change in y', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Actual effect of x on y when z=1 is fixed (nonlinear)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_i_gamma_model.png', dpi=300, bbox_inches='tight')
    print("Figure saved as: example_i_gamma_model.png")

    # Calculate actual effect
    # When x increases from 1 to 2 (increase by 1 unit), with z=1 fixed
    recip_y_x1 = 2 + 1.5 * 1 + 0.6 * 1 + 0.5 * 1 * 1
    recip_y_x2 = 2 + 1.5 * 2 + 0.6 * 1 + 0.5 * 2 * 1
    y_x1 = 1 / recip_y_x1
    y_x2 = 1 / recip_y_x2
    actual_change = y_x2 - y_x1

    print("\n" + "=" * 60)
    print("Critical discussion:")
    print("=" * 60)
    print("""
    Issues and shortcomings:
    1. Misinterpretation: the report says "when x increases by 1 unit, y increases by 1.5 units", which is wrong
       - The model is built for 1/y, not for y
       - An increase of 1.5 in 1/y does not mean y increases by 1.5
       - The actual change in y is nonlinear and depends on the current value of y

    2. Correct interpretation:
       - When z = 1 is fixed, a 1-unit increase in x increases 1/y by 1.5
       - This implies that the change in y is approximately: Δy ≈ -1.5 * y² / (1/y)²
       - The actual effect depends on the current value of y and is not constant

    3. Suggestions for improvement:
       - Interpret the results on the original scale (y) rather than on the transformed scale (1/y)
       - Use marginal effects or average effects for interpretation
       - Provide prediction intervals, not just point estimates
       - Visualize the actual effect of x on y (nonlinear relationship)

    4. Good practice:
       - Using a Gamma distribution with a reciprocal link is reasonable (appropriate for positive responses)
       - Reporting p-values indicates that statistical tests were performed
       - Degrees-of-freedom information is complete (1996 error df)
    """)
    print("\nExample of actual effect (x from 1 to 2, z=1 fixed):")
    print(f"  Change in 1/y: {recip_y_x2 - recip_y_x1:.4f}")
    print(f"  Actual change in y: {actual_change:.4f} (not 1.5!)")


def example_ii():
    """
    Example (ii): Linear model, residual plot shows heteroscedasticity
    Issue: heteroscedasticity evident in residual plot is ignored
    """
    print("\n" + "=" * 60)
    print("Example (ii): Analysis of heteroscedasticity in linear model")
    print("=" * 60)

    # Generate toy data (simulate heteroscedasticity)
    n = 1000
    np.random.seed(456)

    x = np.random.uniform(-2, 2, n)
    # Heteroscedastic error: variance increases with |x|
    sigma = 0.05 + 0.1 * np.abs(x)
    epsilon = np.random.normal(0, sigma, n)
    y = -0.3 * x + epsilon  # Negative linear effect

    # Fit linear model
    model = LinearRegression()
    X = x.reshape(-1, 1)
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    print("\nModel fitting results:")
    print(f"Slope: {model.coef_[0]:.4f} (true value: -0.3)")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R²: {model.score(X, y):.4f}")

    # Statistical test (simplified version of Breusch–Pagan test)
    # Test whether residual variance is related to fitted values
    from scipy.stats import spearmanr
    corr_resid_pred, p_hetero = spearmanr(np.abs(residuals), np.abs(y_pred))

    print("\nHeteroscedasticity test:")
    print(f"Correlation between |residuals| and |fitted values|: {corr_resid_pred:.4f}")
    print(f"p-value: {p_hetero:.4f}")

    # Plot analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Original residual plot (recreate the plot in the report)
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20, marker='x')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs fitted values (showing heteroscedasticity)')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Standardized residual plot
    std_residuals = residuals / np.std(residuals)
    axes[0, 1].scatter(y_pred, std_residuals, alpha=0.5, s=20, marker='x')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Fitted values')
    axes[0, 1].set_ylabel('Standardized residuals')
    axes[0, 1].set_title('Standardized residuals plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: |residuals| vs fitted values (clearer for heteroscedasticity)
    axes[1, 0].scatter(y_pred, np.abs(residuals), alpha=0.5, s=20, marker='x')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('|Residuals|')
    axes[1, 0].set_title('Absolute residuals vs fitted values (heteroscedasticity diagnosis)')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Improved model (weighted least squares or transformation)
    # Use Box–Cox transformation or weighted regression
    # Here we show a sqrt transformation
    y_sqrt = np.sqrt(np.abs(y - y.min()) + 1)  # Avoid negative values
    model_improved = LinearRegression()
    model_improved.fit(X, y_sqrt)
    y_pred_improved = model_improved.predict(X)
    residuals_improved = y_sqrt - y_pred_improved

    axes[1, 1].scatter(y_pred_improved, residuals_improved, alpha=0.5, s=20, marker='x', color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Fitted values (improved model)')
    axes[1, 1].set_ylabel('Residuals (improved model)')
    axes[1, 1].set_title('Residual plot after improvement (after transformation)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_ii_heteroscedasticity.png', dpi=300, bbox_inches='tight')
    print("Figure saved as: example_ii_heteroscedasticity.png")

    print("\n" + "=" * 60)
    print("Critical discussion:")
    print("=" * 60)
    print("""
    Issues and shortcomings:
    1. Heteroscedasticity is ignored:
       - The residual plot clearly shows a "funnel" or "fan" pattern
       - The variance of the residuals increases (or decreases) with the fitted values
       - This violates the homoscedasticity assumption of linear regression

    2. Excuse of insufficient sample size:
       - The report claims "since there are few data points (1000), this is acceptable"
       - In fact, 1000 data points is not a small sample
       - Heteroscedasticity is a serious issue and cannot be ignored because of sample size

    3. Consequences:
       - Standard error estimates are biased
       - Confidence intervals and p-values are unreliable
       - This may lead to incorrect statistical inference

    4. Suggestions for improvement:
       - Use weighted least squares (WLS) regression
       - Transform the response variable (e.g., log or square-root transformation)
       - Use robust standard errors (e.g., Huber–White standard errors)
       - Consider generalized linear models (GLMs)
       - Conduct formal tests for heteroscedasticity (e.g., Breusch–Pagan test)

    5. Good practice:
       - Inspecting residual plots is good practice
       - But one must correctly interpret and address the issues revealed
       - Obvious patterns should not be ignored on the grounds of "small sample size"
    """)


def example_iii():
    """
    Example (iii): Randomization test comparing two biased coins
    Issue: unbalanced sample sizes and possibly insufficient power
    """
    print("\n" + "=" * 60)
    print("Example (iii): Randomization test analysis")
    print("=" * 60)

    # Generate toy data
    np.random.seed(789)

    nA = 20
    nB = 40
    pA = 0.8  # Probability of heads for coin A
    pB = 0.6  # Probability of heads for coin B

    # Simulate experiment
    coinA_flips = np.random.binomial(1, pA, nA)
    coinB_flips = np.random.binomial(1, pB, nB)

    propA = np.mean(coinA_flips)
    propB = np.mean(coinB_flips)

    print("\nSimulation results:")
    print(f"Coin A: {nA} flips, proportion of heads = {propA:.2f} (true value: {pA})")
    print(f"Coin B: {nB} flips, proportion of heads = {propB:.2f} (true value: {pB})")

    # Randomization test (500 permutations)
    n_permutations = 500
    observed_diff = propA - propB

    # Combine all data
    all_data = np.concatenate([coinA_flips, coinB_flips])

    # Randomization distribution
    permuted_diffs = []
    for _ in range(n_permutations):
        # Randomly shuffle
        shuffled = np.random.permutation(all_data)
        # Split into two groups
        perm_A = shuffled[:nA]
        perm_B = shuffled[nA:]
        # Compute difference
        diff = np.mean(perm_A) - np.mean(perm_B)
        permuted_diffs.append(diff)

    permuted_diffs = np.array(permuted_diffs)

    # Compute p-value (two-sided test)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

    print("\nRandomization test results:")
    print(f"Observed difference: {observed_diff:.4f}")
    print(f"p-value: {p_value:.4f} (reported value: p>0.1)")

    # Plot analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Bar plot (recreate the plot in the report)
    categories = ['A', 'B']
    proportions = [propA, propB]
    colors = ['blue', 'blue']
    axes[0, 0].bar(categories, proportions, color=colors, alpha=0.7, width=0.5)
    axes[0, 0].set_ylabel('Proportion of heads')
    axes[0, 0].set_xlabel('Coin')
    axes[0, 0].set_title(f'Proportion of heads for coins A and B\n(nA={nA}, nB={nB})')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (cat, prop) in enumerate(zip(categories, proportions)):
        axes[0, 0].text(i, prop + 0.05, f'{prop:.2f}', ha='center', fontsize=12)

    # Panel 2: Randomization distribution
    axes[0, 1].hist(permuted_diffs, bins=30, alpha=0.7, color='gray', edgecolor='black')
    axes[0, 1].axvline(observed_diff, color='r', linestyle='--', linewidth=2, label=f'Observed = {observed_diff:.3f}')
    axes[0, 1].axvline(-observed_diff, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Difference in proportions (A - B)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Randomization distribution (500 permutations)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Effect of unbalanced sample sizes
    # Simulate test power under different sample sizes
    sample_sizes = [(10, 10), (20, 20), (20, 40), (40, 40), (50, 50)]
    powers = []
    for n1, n2 in sample_sizes:
        # Simulate many experiments and compute rejection rate
        n_sims = 1000
        rejections = 0
        for _ in range(n_sims):
            simA = np.random.binomial(1, pA, n1)
            simB = np.random.binomial(1, pB, n2)
            propA_sim = np.mean(simA)
            propB_sim = np.mean(simB)
            # Simplified z-test
            pooled_p = (np.sum(simA) + np.sum(simB)) / (n1 + n2)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
            if se > 0:
                z = (propA_sim - propB_sim) / se
                if np.abs(z) > 1.96:  # Two-sided test, α=0.05
                    rejections += 1
        power = rejections / n_sims
        powers.append(power)

    labels = [f'({n1},{n2})' for n1, n2 in sample_sizes]
    axes[1, 0].bar(range(len(labels)), powers, alpha=0.7, color='green')
    axes[1, 0].set_xticks(range(len(labels)))
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].set_xlabel('Sample sizes (nA, nB)')
    axes[1, 0].set_title('Test power under different sample sizes')
    axes[1, 0].axhline(0.8, color='r', linestyle='--', label='Power = 0.8')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Confidence intervals
    # Compute confidence intervals for proportions
    from scipy.stats import norm
    z_critical = norm.ppf(0.975)

    seA = np.sqrt(propA * (1 - propA) / nA)
    seB = np.sqrt(propB * (1 - propB) / nB)
    ciA = [propA - z_critical * seA, propA + z_critical * seA]
    ciB = [propB - z_critical * seB, propB + z_critical * seB]

    axes[1, 1].barh([0, 1], [propA, propB], xerr=[[propA - ciA[0], propB - ciB[0]],
                                                  [ciA[1] - propA, ciB[1] - propB]],
                    alpha=0.7, color=['blue', 'blue'], capsize=5)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Coin A', 'Coin B'])
    axes[1, 1].set_xlabel('Proportion of heads')
    axes[1, 1].set_title('Proportion estimates and 95% confidence intervals')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('example_iii_randomization_test.png', dpi=300, bbox_inches='tight')
    print("Figure saved as: example_iii_randomization_test.png")

    print("\n" + "=" * 60)
    print("Critical discussion:")
    print("=" * 60)
    print("""
    Issues and shortcomings:
    1. Unbalanced sample sizes:
       - Coin A is flipped only 20 times, while coin B is flipped 40 times
       - Unbalanced sample sizes reduce test power
       - Smaller sample sizes lead to greater uncertainty in estimates

    2. Insufficient power:
       - p > 0.1 may not mean there is no difference, but rather that the test has low power
       - The true difference is 0.2 (0.8 vs 0.6), which is quite large
       - But the sample sizes are too small to reliably detect this difference

    3. Incomplete reporting:
       - Only the p-value is reported, without confidence intervals
       - Test power is not discussed
       - The impact of unbalanced sample sizes is not considered

    4. Suggestions for improvement:
       - Increase sample sizes, especially for coin A
       - Use a balanced design (nA = nB)
       - Report confidence intervals as well as p-values
       - Conduct a power analysis to determine the sample size needed to detect the difference
       - Consider using an exact test (Fisher’s exact test) instead of a randomization test
       - Report effect sizes (e.g., difference in proportions) and their confidence intervals

    5. Good practice:
       - Using a randomization test is reasonable, especially for small samples
       - But it is important to ensure adequate test power
       - Effect sizes and confidence intervals should be reported, not just p-values
       - Limitations in sample size and test power should be discussed
    """)


# ==================== Main function ====================

def main():
    """Run Part (b)"""
    print("\n" + "=" * 60)
    print("Question 4 Part (b): Evaluation of Statistical Reports")
    print("=" * 60)

    example_i()
    example_ii()
    example_iii()

    print("\n" + "=" * 60)
    print("Part (b) completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
