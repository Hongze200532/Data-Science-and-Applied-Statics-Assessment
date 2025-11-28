# Question 4 Part (a): Statistical Analysis of Burning Experiment

---

## Data Summary

**Data shape:** `(25, 4)`

**Data summary:**

|       Stat |    brand | ingredient1 | ingredient2 | gas_emission |
|----------:|---------:|------------:|------------:|-------------:|
|     count | 25.000000 |   25.00000 |  25.000000 |   25.000000 |
|      mean | 13.000000 |   12.21600 |   0.876400 |   12.528000 |
|       std |  7.359801 |    5.66581 |   0.354058 |    4.739684 |
|       min |  1.000000 |    1.00000 |   0.130000 |    1.500000 |
|      25%  |  7.000000 |    8.60000 |   0.690000 |   10.000000 |
|      50%  | 13.000000 |   12.80000 |   0.900000 |   13.000000 |
|      75%  | 19.000000 |   15.10000 |   1.020000 |   15.400000 |
|      max  | 25.000000 |   29.80000 |   2.030000 |   23.500000 |

---

## Regression Analysis Results

- **Intercept:** 12.53 mg  
- **Ingredient 1 coefficient:** 5.34 mg (after standardization)  
- **Ingredient 2 coefficient:** -0.92 mg (after standardization)  
- **R²:** 0.9186  

---

## Correlation Analysis

Correlation analysis:

- Correlation between ingredient 1 and gas emission: **0.9575**  
- Correlation between ingredient 2 and gas emission: **0.9259**  
- Correlation between ingredient 1 and ingredient 2: **0.9766**  

---

## Statistical Tests

Statistical tests:

- Effect of ingredient 1: slope = **0.8010**, p-value = **0.0000**  
- Effect of ingredient 2: slope = **12.3954**, p-value = **0.0000**  

---

## Figure Information

- **Figure saved as:** `problem_a_analysis.png`

---

## Statistical Analysis Discussion


1. **Analysis of ingredient effects:**
   - Ingredient 1 is positively correlated with gas emission; increasing ingredient 1 increases gas emission.  
   - Ingredient 2 is negatively correlated with gas emission; increasing ingredient 2 decreases gas emission.  
   - There is an interaction effect between the two ingredients.

2. **Predictive ability:**
   - The linear regression model can be used to predict gas emission given the ingredient contents.  
   - The R² value reflects the explanatory power of the model.  
   - Residual analysis shows whether the model assumptions are satisfied.

3. **Suggestions for improving the experiment:**
   - Increase sample size: currently only 25 brands, recommend increasing to 50–100.  
   - Experimental design: use factorial design to systematically study ingredient effects.  
   - Control variables: control other factors that may affect gas emission (e.g., burning temperature, time, etc.).  
   - Replication: perform multiple repeated experiments for each brand to reduce random error.  
   - Ingredient range: widen the range of ingredient contents to better estimate dose–response relationships.  
   - Interaction: specifically design experiments to study the interaction between ingredients.  
   - Nonlinear relationships: consider nonlinear effects of ingredients (e.g., quadratic terms).  
   - Randomization: ensure randomization of experimental order to avoid systematic bias.  
