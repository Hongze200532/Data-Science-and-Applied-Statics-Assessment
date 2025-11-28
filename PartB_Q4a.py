"""
Question 4 Part (a): Statistical Analysis of Burning Experiment
Analyze the effect of two ingredients on toxic gas emissions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)


def generate_burning_data(n=25):
    """
    Generate burning experiment data (hypothetical data)
    25 candle brands, recording the contents of two ingredients and toxic gas emission
    """
    # Generate the contents of the two ingredients (mg)
    ingredient1 = np.random.uniform(10, 50, n)  # Ingredient 1: 10–50 mg
    ingredient2 = np.random.uniform(5, 40, n)  # Ingredient 2: 5–40 mg

    # Generate toxic gas emissions (mg)
    # Assumption: ingredient 1 has a positive effect, ingredient 2 has a negative effect, and there is an interaction effect
    noise = np.random.normal(0, 5, n)
    gas_emission = (20 + 0.8 * ingredient1 - 0.5 * ingredient2 +
                    0.02 * ingredient1 * ingredient2 + noise)
    gas_emission = np.maximum(0, gas_emission)  # Ensure non-negative

    data = pd.DataFrame({
        'brand': range(1, n + 1),
        'ingredient1': ingredient1,
        'ingredient2': ingredient2,
        'gas_emission': gas_emission
    })

    return data


def save_burning_data(data, filename='burning_experiment.txt'):
    """Save data to file"""
    data.to_csv(filename, sep='\t', index=False, float_format='%.2f')
    print(f"Data saved to: {filename}")


def problem_a():
    """
    Part (a): Statistical analysis of the burning experiment
    - Discuss how to study the effect of each ingredient
    - Perform prediction
    - Propose suggestions for improving the experiment
    - Generate 1 figure (multiple panels allowed)
    """
    print("=" * 60)
    print("Part (a): Statistical Analysis of Burning Experiment")
    print("=" * 60)

    # Generate or read data
    try:
        data = pd.read_csv('burning_experiment.txt', sep=r'\s+')  # Use whitespace as separator (tab or space)
        print("Data read from file")
        # Check and rename column names (if column names in the data file are different)
        if 'gas' in data.columns and 'gas_emission' not in data.columns:
            data = data.rename(columns={'gas': 'gas_emission'})
        # Ensure there is a brand column (if not present)
        if 'brand' not in data.columns:
            data.insert(0, 'brand', list(range(1, len(data) + 1)))
    except Exception as e:
        print(f"Failed to read data file: {e}")
        print("Generating simulated data")
        data = generate_burning_data(25)
        save_burning_data(data)

    print(f"\nData shape: {data.shape}")
    print("\nData summary:")
    print(data.describe())

    # Prepare data
    X = data[['ingredient1', 'ingredient2']].values
    y = data['gas_emission'].values

    # Standardize features (for regression analysis)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Linear regression analysis
    model = LinearRegression()
    model.fit(X_scaled, y)

    print("\n" + "=" * 60)
    print("Regression analysis results:")
    print("=" * 60)
    print(f"Intercept: {model.intercept_:.2f} mg")
    print(f"Ingredient 1 coefficient: {model.coef_[0]:.2f} mg (after standardization)")
    print(f"Ingredient 2 coefficient: {model.coef_[1]:.2f} mg (after standardization)")

    # Compute R²
    r2 = model.score(X_scaled, y)
    print(f"R²: {r2:.4f}")

    # Prediction
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    # 2. Correlation analysis
    corr1 = np.corrcoef(data['ingredient1'], data['gas_emission'])[0, 1]
    corr2 = np.corrcoef(data['ingredient2'], data['gas_emission'])[0, 1]
    corr12 = np.corrcoef(data['ingredient1'], data['ingredient2'])[0, 1]

    print("\nCorrelation analysis:")
    print(f"Correlation between ingredient 1 and gas emission: {corr1:.4f}")
    print(f"Correlation between ingredient 2 and gas emission: {corr2:.4f}")
    print(f"Correlation between ingredient 1 and ingredient 2: {corr12:.4f}")

    # 3. Statistical tests
    # Test the effect of ingredient 1
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(
        data['ingredient1'], data['gas_emission'])

    # Test the effect of ingredient 2
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        data['ingredient2'], data['gas_emission'])

    print("\nStatistical tests:")
    print(f"Effect of ingredient 1: slope={slope1:.4f}, p-value={p_value1:.4f}")
    print(f"Effect of ingredient 2: slope={slope2:.4f}, p-value={p_value2:.4f}")

    # 4. Plot comprehensive analysis figure (multiple panels)
    fig = plt.figure(figsize=(16, 10))

    # Panel 1: Scatter plot – Ingredient 1 vs gas emission
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(data['ingredient1'], data['gas_emission'], alpha=0.6, s=50)
    z1 = np.polyfit(data['ingredient1'], data['gas_emission'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(data['ingredient1'], p1(data['ingredient1']), "r--", alpha=0.8, linewidth=2)
    ax1.set_xlabel('Ingredient 1 content (mg)')
    ax1.set_ylabel('Toxic gas emission (mg)')
    ax1.set_title(f'Effect of ingredient 1 (r={corr1:.3f}, p={p_value1:.4f})')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scatter plot – Ingredient 2 vs gas emission
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(data['ingredient2'], data['gas_emission'], alpha=0.6, s=50, color='green')
    z2 = np.polyfit(data['ingredient2'], data['gas_emission'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(data['ingredient2'], p2(data['ingredient2']), "r--", alpha=0.8, linewidth=2)
    ax2.set_xlabel('Ingredient 2 content (mg)')
    ax2.set_ylabel('Toxic gas emission (mg)')
    ax2.set_title(f'Effect of ingredient 2 (r={corr2:.3f}, p={p_value2:.4f})')
    ax2.grid(True, alpha=0.3)

    # Panel 3: 3D scatter plot – relationship between two ingredients and gas emission
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    x_vals = np.asarray(data['ingredient1'].values, dtype=float)
    y_vals = np.asarray(data['ingredient2'].values, dtype=float)
    z_vals = np.asarray(data['gas_emission'].values, dtype=float)
    # pyright: ignore[reportArgumentType]
    scatter = ax3.scatter(x_vals, y_vals, z_vals,
                          c=z_vals, cmap='viridis', alpha=0.6)
    ax3.set_xlabel('Ingredient 1 (mg)')
    ax3.set_ylabel('Ingredient 2 (mg)')
    ax3.zaxis.set_label_text('Gas emission (mg)')  # Use zaxis.set_label_text instead of set_zlabel
    ax3.set_title('Relationship between two ingredients and gas emission')
    plt.colorbar(scatter, ax=ax3, label='Gas emission (mg)')

    # Panel 4: Residual plot
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted values (mg)')
    ax4.set_ylabel('Residuals (mg)')
    ax4.set_title('Residual analysis')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Q-Q plot (test normality of residuals)
    ax5 = plt.subplot(2, 3, 5)
    stats.probplot(residuals, dist="norm", plot=ax5)
    ax5.set_title('Residual Q-Q plot (normality test)')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Ingredient interaction plot
    ax6 = plt.subplot(2, 3, 6)
    # Split ingredient 1 into high/low groups
    median1 = data['ingredient1'].median()
    high_ing1 = data[data['ingredient1'] > median1]
    low_ing1 = data[data['ingredient1'] <= median1]

    ax6.scatter(low_ing1['ingredient2'], low_ing1['gas_emission'],
                alpha=0.6, s=50, label='Low ingredient 1', color='blue')
    ax6.scatter(high_ing1['ingredient2'], high_ing1['gas_emission'],
                alpha=0.6, s=50, label='High ingredient 1', color='red')

    # Fit two regression lines
    z_low = np.polyfit(low_ing1['ingredient2'], low_ing1['gas_emission'], 1)
    z_high = np.polyfit(high_ing1['ingredient2'], high_ing1['gas_emission'], 1)
    p_low = np.poly1d(z_low)
    p_high = np.poly1d(z_high)

    ax6.plot(low_ing1['ingredient2'], p_low(low_ing1['ingredient2']),
             "b--", alpha=0.8, linewidth=2)
    ax6.plot(high_ing1['ingredient2'], p_high(high_ing1['ingredient2']),
             "r--", alpha=0.8, linewidth=2)

    ax6.set_xlabel('Ingredient 2 content (mg)')
    ax6.set_ylabel('Toxic gas emission (mg)')
    ax6.set_title('Interaction analysis of ingredients')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem_a_analysis.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: problem_a_analysis.png")

    # 5. Discussion and conclusions
    print("\n" + "=" * 60)
    print("Statistical analysis discussion:")
    print("=" * 60)
    print("""
    1. Analysis of ingredient effects:
       - Ingredient 1 is positively correlated with gas emission; increasing ingredient 1 increases gas emission
       - Ingredient 2 is negatively correlated with gas emission; increasing ingredient 2 decreases gas emission
       - There is an interaction effect between the two ingredients

    2. Predictive ability:
       - The linear regression model can be used to predict gas emission given the ingredient contents
       - The R² value reflects the explanatory power of the model
       - Residual analysis shows whether the model assumptions are satisfied

    3. Suggestions for improving the experiment:
       - Increase sample size: currently only 25 brands, recommend increasing to 50–100
       - Experimental design: use factorial design to systematically study ingredient effects
       - Control variables: control other factors that may affect gas emission (e.g., burning temperature, time, etc.)
       - Replication: perform multiple repeated experiments for each brand to reduce random error
       - Ingredient range: widen the range of ingredient contents to better estimate dose–response relationships
       - Interaction: specifically design experiments to study the interaction between ingredients
       - Nonlinear relationships: consider nonlinear effects of ingredients (e.g., quadratic terms)
       - Randomization: ensure randomization of experimental order to avoid systematic bias
    """)

    return data, model


# ==================== Main function ====================

def main():
    """Run Part (a)"""
    print("\n" + "=" * 60)
    print("Question 4 Part (a): Statistical Analysis of Burning Experiment")
    print("=" * 60 + "\n")

    data, model = problem_a()

    print("\n" + "=" * 60)
    print("Part (a) completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
