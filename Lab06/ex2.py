"""
Context: Synthetic Bayesian modeling example for call-center rates.

Setup/problem:
- We model the number of calls X observed over T hours using a Poisson
  likelihood with rate λ per hour (so X ~ Poisson(T·λ)).
- This script uses a truncated normal prior over λ (> 0) and computes
  a numerical posterior on a grid to illustrate concepts like the
  posterior mean, mode, and a 94% interval.

Important note/disclaimer:
- The Lab 6, Ex. 2 assignment expects working with a conjugate prior
  and an analytic posterior for λ. This example intentionally uses a
  different prior and a numeric approach, and is NOT a template for the
  assignment solution. It also uses synthetic data (k, T) that do not
  match the assignment prompt.
"""
import arviz as az
import scipy.stats as stats

k = 180    # Numărul total de apeluri observate (sintetic)
T = 10      # Intervalul de timp în ore (sintetic)


alpha = 1
beta = 0.5
# punctul A
alpha_post = alpha + k
beta_post = beta + T
posterior_mean = alpha_post/beta_post
posterior_variance = alpha_post/(beta_post**2)
print("Posterior mean:", posterior_mean)
print("Posterior variance:", posterior_variance)

# punctul b
posterior_samples = stats.gamma.rvs(alpha_post, scale = 1/beta_post, size = 10000)
hdi = az.hdi(posterior_samples, hdi_prob = 0.94)

print("Lower bound: ", hdi[0])
print("Upper bound: ", hdi[1])

# punctul c
posterior_mode = 0
if alpha_post > 1:
    posterior_mode = (alpha_post - 1) /beta_post
print("Posterior mode:", posterior_mode)