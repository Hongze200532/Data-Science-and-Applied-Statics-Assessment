# Question 4 Part (b): Evaluation of Statistical Reports  

---

## Example (i): Gamma Distribution Model Analysis

**Fitting results:**

- Intercept: 1.4056 (reported value: 2.0)  
- Coefficient for x: -104.7975 (reported value: 1.5)  
- Coefficient for z: 1.4058 (reported value: 0.6)  
- Interaction coefficient: 106.7947 (reported value: 0.5)  

*Figure saved as: `example_i_gamma_model.png`*

### Critical discussion

**Issues and shortcomings:**

1. Misinterpretation: the report says *"when x increases by 1 unit, y increases by 1.5 units"*, which is wrong  
   - The model is built for `1/y`, not for `y`  
   - An increase of 1.5 in `1/y` does **not** mean `y` increases by 1.5  
   - The actual change in `y` is nonlinear and depends on the current value of `y`  

2. Correct interpretation:  
   - When `z = 1` is fixed, a 1-unit increase in `x` increases `1/y` by 1.5  
   - This implies that the change in `y` is approximately:  
     \[
     \Delta y \approx -1.5 \cdot \frac{y^2}{(1/y)^2}
     \]
   - The actual effect depends on the current value of `y` and is not constant  

3. Suggestions for improvement:  
   - Interpret the results on the original scale (`y`) rather than on the transformed scale (`1/y`)  
   - Use marginal effects or average effects for interpretation  
   - Provide prediction intervals, not just point estimates  
   - Visualize the actual effect of `x` on `y` (nonlinear relationship)  

4. Good practice:  
   - Using a Gamma distribution with a reciprocal link is reasonable (appropriate for positive responses)  
   - Reporting p-values indicates that statistical tests were performed  
   - Degrees-of-freedom information is complete (1996 error df)  

**Example of actual effect (x from 1 to 2, z = 1 fixed):**

- Change in `1/y`: 2.0000  
- Actual change in `y`: -0.0659 (**not 1.5!**)  

---

## Example (ii): Analysis of Heteroscedasticity in Linear Model

**Model fitting results:**

- Slope: -0.3090 (true value: -0.3)  
- Intercept: 0.0014  
- R²: 0.8392  

**Heteroscedasticity test:**

- Correlation between \|residuals\| and \|fitted values\|: 0.4231  
- p-value: 0.0000  

*Figure saved as: `example_ii_heteroscedasticity.png`*

### Critical discussion

**Issues and shortcomings:**

1. Heteroscedasticity is ignored:  
   - The residual plot clearly shows a *“funnel”* or *“fan”* pattern  
   - The variance of the residuals increases (or decreases) with the fitted values  
   - This violates the homoscedasticity assumption of linear regression  

2. Excuse of insufficient sample size:  
   - The report claims *"since there are few data points (1000), this is acceptable"*  
   - In fact, 1000 data points is **not** a small sample  
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
   - Obvious patterns should not be ignored on the grounds of *"small sample size"*  

---

## Example (iii): Randomization Test Analysis


**Simulation results:**

- Coin A: 20 flips, proportion of heads = 0.95 (true value: 0.8)  
- Coin B: 40 flips, proportion of heads = 0.57 (true value: 0.6)  

**Randomization test results:**

- Observed difference: 0.3750  
- p-value: 0.0020 (reported value: p > 0.1)  

*Figure saved as: `example_iii_randomization_test.png`*

### Critical discussion

**Issues and shortcomings:**

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
