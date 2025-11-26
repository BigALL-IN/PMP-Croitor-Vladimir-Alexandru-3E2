import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt


# Observed A/B outcomes
Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# a)
traces = {}
for i, theta in enumerate(theta_values):
    for j, Y in enumerate(Y_values):

        with pm.Model() as model:
            n = pm.Poisson("n", mu=10)

            pm.Binomial("y_obs", n=n, p=theta, observed=Y)

            trace = pm.sample(
                draws=2000,
                tune=2000,
                chains=2,
                cores=1,
                random_seed=2025,
                progressbar=False,
            )

        ax = axes[i, j]
        az.plot_posterior(
            trace,
            var_names=["n"],
            hdi_prob=0.94,
            ax=ax,
            textsize=10
        )

        traces[(Y, theta)] = trace

        summary = az.summary(trace, var_names=["n"], hdi_prob=0.94)
        print(f"Summary for Y = {Y} Theta = {theta}:")
        print(summary)

plt.tight_layout()
plt.savefig('pred_dist.png')
plt.show()

# c)

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

for i, theta in enumerate(theta_values):
    for j, Y in enumerate(Y_values):

        trace = traces[(Y, theta)]

        n_samples = np.ravel(trace.posterior["n"].values)
        n_pred = np.random.choice(n_samples, size=2000, replace=True)
        Y_star = np.random.binomial(n_pred, theta)
        ax = axes2[i, j]

        az.plot_dist(
            Y_star,
            kind='hist',
            ax=ax,
        )


plt.tight_layout()
plt.savefig('post_dist2.png')
plt.show()