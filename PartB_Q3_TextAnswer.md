# Geiger counter network simulation - complete solution

---

## Problem (a): Simulate Y (α = 1)

**Statistics of Y:**

- Mean: 400.43 (theoretical value: 400.0)  
- Standard deviation: 19.48 (theoretical value: 20.00)  
- Minimum: 332  
- Maximum: 476  

**Figure saved as:** `problem_a_distribution.png`

---

## Problem (b): Approximate simulation models

**Accuracy analysis of approximate models:**

- Normal approximation  
  - KS statistic: 0.0400  
  - p-value: 0.4006  

- Poisson approximation  
  - KS statistic: 0.0530  
  - p-value: 0.1205  

**Discussion:**

1. **Poisson approximation:**  
   Since \( Y \) is the sum of 100 independent Poisson(4) random variables,  
   theoretically \( Y \sim \text{Poisson}(400) \), so the Poisson approximation is extremely accurate.

2. **Normal approximation:**  
   By the central limit theorem, when \( n \) is large,  
   \( Y \) is approximately normal \( N(400, 20^2) \). For large \( \lambda \), the Poisson distribution is also close to normal.  
   Both approximations are accurate, but the Poisson approximation is more precise because it is the exact distribution.

**Figure saved as:** `problem_b_approximations.png`

---

## Problem (c): Hypothesis testing

- **Null hypothesis** \( H_0 \): \( \mu \le 400 \)  
- **Alternative hypothesis** \( H_1 \): \( \mu > 400 \)  
- **Significance level:** 5.0%

### Sample statistics

- Sample mean: 400.43  
- Sample standard deviation: 19.49  
- Sample size: 1000  

### t-test results

- t statistic: 0.6960  
- p-value: 0.2433  
- **Conclusion:** Do not reject \( H_0 \)

### z-test results (large-sample approximation)

- z statistic: 0.6960  
- p-value: 0.2432  
- **Conclusion:** Do not reject \( H_0 \)

### Discussion on using hypothesis tests on computer-simulated data

1. Simulated data are deterministic (given a random seed), but statistical inference is still valid.  
2. Hypothesis testing can help verify whether the simulation model matches theoretical expectations.  
3. In simulations, we can generate many samples, leading to high test power.  
4. Note: Simulated data may not satisfy certain assumptions (e.g., independence),  
   but in this problem the counters are independent, so the assumptions hold.  
5. If the p-value is very small, it may indicate an error in the simulation implementation or an inaccurate theoretical model.

---

## Problem (d): Discussion (α = 0.5, r = 0.4)

> When α = 0.5, each counter's report has only a 50% probability of being successfully transmitted.  
>
> **Impact on the approximate models in problem (b):**
>
> 1. **Change in exact distribution:**
>    - When α = 0.5, \( Y = \sum (W_i X_i) \), where \( W_i \sim \text{Bernoulli}(0.5) \)  
>    - Expectation of each \( W_i X_i \):  
>      \( E[W_i X_i] = E[W_i] E[X_i] = 0.5 \times 4 = 2 \)  
>    - Expectation of \( Y \):  
>      \( E[Y] = n \times 0.5 \times 4 = 200 \)  
>    - Variance of \( Y \):  
>      \[
>      \text{Var}(Y) = n \cdot \text{Var}(W_i X_i)
>      = n \cdot [ E[W_i]\text{Var}(X_i) + \text{Var}(W_i) E[X_i]^2 ]
>      = n \cdot [0.5 \times 4 + 0.25 \times 16] = n \cdot [2 + 4] = 600
>      \]
>
> 2. **Normal approximation:**
>    - \( Y \) is approximately \( N(200, \sqrt{600}^2) = N(200, 24.49^2) \)  
>    - Still valid, because the central limit theorem still applies  
>    - But the variance increases because transmission failures introduce extra variability  
>
> 3. **Poisson approximation:**
>    - Due to the presence of \( W_i \), \( Y \) is no longer a simple Poisson distribution  
>    - But it can be approximated as \( \text{Poisson}(200) \), since \( E[Y] = 200 \)  
>    - The accuracy of this approximation decreases because the actual variance (600) is greater than the Poisson variance (200)  
>    - The actual distribution is “zero-inflated” because some counters' reports may be completely lost  
>
> 4. **Conclusion:**
>    - The normal approximation is still applicable, but the parameters need to be adjusted  
>    - The accuracy of the Poisson approximation decreases significantly because the actual distribution deviates from Poisson  
>    - More complex approximations may be needed, such as a negative binomial mixture or other mixture distributions  

---

## Problem (e): Bimodal simulation (α = 0.5, mixed r)

**Statistics:**

- Mean: 350.88  
- Standard deviation: 42.65  
- Minimum: 211  
- Maximum: 516  
- Skewness: 0.1020  
- Kurtosis: 0.0219  

**Figure saved as:** `problem_e_bimodal.png`

---

## Problem (f): Bimodal distribution fitting (MLE)

### Initial parameter estimates

- \( \mu_1 \): 322.00  
- \( \sigma_1 \): 24.43  
- \( \mu_2 \): 380.00  
- \( \sigma_2 \): 26.72  
- Weight \( w \): 0.50  

### MLE estimation results

- \( \mu_1 \): 339.32  
- \( \sigma_1 \): 38.72  
- \( \mu_2 \): 364.63  
- \( \sigma_2 \): 43.02  
- Weight \( w \): 0.5435  
- Negative log-likelihood: 5171.11  

### Goodness-of-fit test

- KS statistic: 0.0182  
- p-value: 0.9196  

### Discussion of adequacy of the fitted distribution

1. The bimodal normal mixture model can capture the two main modes in the data,  
   corresponding to the two groups of counters with \( r = 0.4 \) and \( r = 1.0 \).  
2. However, the actual data come from a mixture of Poisson distributions, not normal distributions.  
3. When parameters are large, Poisson distributions are approximately normal, so the fit can be reasonable.  
4. But note that:  
   - The actual distribution is discrete (count data), while the normal distribution is continuous.  
   - The actual distribution may have more skewness and tail behavior.  
   - A better choice may be a mixture of Poisson distributions or a mixture of negative binomial distributions.  
5. If the KS test p-value is relatively large, the fit is statistically acceptable,  
   but it may still not be the optimal model.

**Figure saved as:** `problem_f_fitting.png`
