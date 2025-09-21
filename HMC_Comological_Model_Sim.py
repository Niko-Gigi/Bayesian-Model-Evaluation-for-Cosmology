# ---------------------------
# Imports and Device Setup
# ---------------------------
import jax
import jax.numpy as jnp
from jax import jit, vmap
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_mean, Predictive
from jax.scipy.integrate import trapezoid
from jax.scipy.linalg import solve_triangular
import arviz as az
import numpy as np
import gzip
from numpyro.infer.util import log_likelihood

# ---------------------------
# Enable 64-bit precision for numerical stability, though note that this clashes with 
# set_host_device_count so only one can be used at a time
# ---------------------------
# jax.config.update("jax_enable_x64", True)

numpyro.set_host_device_count(4)

# ---------------------------
# Constants and Redshift Grid
# ---------------------------
C = 299792.458  # Speed of light in km/s
N = 1001  # Needs to be odd for trapezoid rule to include endpoint

# ---------------------------
# Load DES-SN5YR Metadata
# ---------------------------
meta = pd.read_csv("DES-SN5YR_HD+MetaData.csv") 
z = jnp.array(meta["zHD"].values)
mu_obs = jnp.array(meta["MU"].values)
mu_err = jnp.array(meta["MUERR_FINAL"].values)
n = len(mu_obs)

# ---------------------------
# Load covariance matrix and precompute Cholesky
# ---------------------------
with gzip.open("STAT+SYS.txt.gz", "rt") as f:
    flat = np.loadtxt(f)
flat = flat[1:]
cov_sys = flat.reshape((n, n))
cov_stat = np.diag(mu_err**2)
cov_full = cov_stat + cov_sys
cov_full_jax = jnp.array(cov_full)

# Precompute Cholesky decomposition once
L_chol = jnp.linalg.cholesky(cov_full_jax) 

# ---------------------------
# Distance Functions
# ---------------------------
def compute_dL(z_array, H0, Ez_fn):
    @jit
    def d_L_single(z_i):
        z_vals = jnp.linspace(0.0, z_i, N)
        Ez = Ez_fn(z_vals)
        integral = trapezoid(1.0 / Ez, z_vals)
        return (1 + z_i) * C / H0 * integral
    return vmap(d_L_single)(z_array)

# ---------------------------
# Dimensionless Hubble parameters (Square root of the variable term in the Friedmann equation)
# for all three models
# ---------------------------
def E_LCDM(Omega_m):
    return lambda z: jnp.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

def E_wCDM(Omega_m, w):
    return lambda z: jnp.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m) * (1 + z)**(3 * (1 + w)))

def E_CPL(Omega_m, w0, wa):
    return lambda z: jnp.sqrt(
        Omega_m * (1 + z)**3 +
        (1 - Omega_m) * (1 + z)**(3 * (w0 + wa + 1)) *
        jnp.exp(-3 * wa * z / (1 + z))
    )

@jit
def dL_LCDM(z_array, H0, Omega_m):
    return compute_dL(z_array, H0, E_LCDM(Omega_m))

@jit
def dL_wCDM(z_array, H0, Omega_m, w):
    return compute_dL(z_array, H0, E_wCDM(Omega_m, w))

@jit
def dL_CPL(z_array, H0, Omega_m, w0, wa):
    return compute_dL(z_array, H0, E_CPL(Omega_m, w0, wa))

# ---------------------------
# Models implemented with model assumptions
# ---------------------------
def lcdm_model(z, mu_obs=None, L_chol=None):
    H0 = numpyro.sample("H0", dist.LogNormal(jnp.log(70), 0.5))
    Omega_m = numpyro.sample("Omega_m", dist.Beta(3, 7))

    mu_model = 5 * jnp.log10(dL_LCDM(z, H0, Omega_m) + 1e-10) + 25
    numpyro.deterministic("mu_model", mu_model)

    if mu_obs is not None:
        resid = mu_obs - mu_model
        y_whitened = solve_triangular(L_chol, resid, lower=True)
        with numpyro.plate("obs", len(z)):
            numpyro.sample("y", dist.Normal(0, 1), obs=y_whitened)
        joint_ll = dist.MultivariateNormal(loc=mu_model, scale_tril=L_chol).log_prob(mu_obs)
        numpyro.sample("joint_log_likelihood", dist.Delta(joint_ll), obs=joint_ll)
    else:
        with numpyro.plate("obs", len(z)):
            y = numpyro.sample("y", dist.Normal(0, 1))
        corr_resid = L_chol @ y
        mu_pred = mu_model + corr_resid
        numpyro.deterministic("mu_pred", mu_pred)

def wcdm_model(z, mu_obs=None, L_chol=None):
    H0 = numpyro.sample("H0", dist.LogNormal(jnp.log(70), 0.5))
    Omega_m = numpyro.sample("Omega_m", dist.Beta(3, 7))
    w = numpyro.sample("w", dist.Normal(-1, 2))

    mu_model = 5 * jnp.log10(dL_wCDM(z, H0, Omega_m, w) + 1e-10) + 25
    numpyro.deterministic("mu_model", mu_model)

    if mu_obs is not None:
        resid = mu_obs - mu_model
        y_whitened = solve_triangular(L_chol, resid, lower=True)
        with numpyro.plate("obs", len(z)):
            numpyro.sample("y", dist.Normal(0, 1), obs=y_whitened)
        joint_ll = dist.MultivariateNormal(loc=mu_model, scale_tril=L_chol).log_prob(mu_obs)
        numpyro.sample("joint_log_likelihood", dist.Delta(joint_ll), obs=joint_ll)
    else:
        with numpyro.plate("obs", len(z)):
            y = numpyro.sample("y", dist.Normal(0, 1))
        corr_resid = L_chol @ y
        mu_pred = mu_model + corr_resid
        numpyro.deterministic("mu_pred", mu_pred)

def cpl_model(z, mu_obs=None, L_chol=None):
    H0 = numpyro.sample("H0", dist.LogNormal(jnp.log(70), 0.5))
    Omega_m = numpyro.sample("Omega_m", dist.Beta(3, 7))
    w0 = numpyro.sample("w0", dist.Normal(-1, 2))
    wa = numpyro.sample("wa", dist.Normal(0, 4))

    mu_model = 5 * jnp.log10(dL_CPL(z, H0, Omega_m, w0, wa) + 1e-10) + 25
    numpyro.deterministic("mu_model", mu_model)

    if mu_obs is not None:
        resid = mu_obs - mu_model
        y_whitened = solve_triangular(L_chol, resid, lower=True)
        with numpyro.plate("obs", len(z)):
            numpyro.sample("y", dist.Normal(0, 1), obs=y_whitened)
        joint_ll = dist.MultivariateNormal(loc=mu_model, scale_tril=L_chol).log_prob(mu_obs)
        numpyro.sample("joint_log_likelihood", dist.Delta(joint_ll), obs=joint_ll)
    else:
        with numpyro.plate("obs", len(z)):
            y = numpyro.sample("y", dist.Normal(0, 1))
        corr_resid = L_chol @ y
        mu_pred = mu_model + corr_resid
        numpyro.deterministic("mu_pred", mu_pred)

# ---------------------------
# Inference function that saves a .nc file using ArviZ will relevant data and simulation outputs
# ---------------------------
def run_and_save(model_fn, name, thin=1):
    rng_key = jax.random.PRNGKey(202)

    # --- Run MCMC ---
    kernel = NUTS(model_fn, init_strategy=init_to_mean(), target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=9000, num_chains=4)
    mcmc.run(rng_key, z=z, mu_obs=mu_obs, L_chol=L_chol)

    # Thin posterior samples
    posterior_samples = {k: v[::thin] for k, v in mcmc.get_samples().items()}

    # Posterior predictive
    predictive = Predictive(model_fn, posterior_samples)
    posterior_predictive = predictive(jax.random.PRNGKey(202), z, mu_obs=None, L_chol=L_chol)

    # Log-likelihood for observed data
    ll = log_likelihood(model_fn, posterior_samples, z=z, mu_obs=mu_obs, L_chol=L_chol)
    ll_y = ll["y"]
    ll_joint = ll["joint_log_likelihood"]

    # Prior predictive
    prior_predictive = Predictive(model_fn, num_samples=1000)
    prior_pred_samples = prior_predictive(jax.random.PRNGKey(202), z, mu_obs=None, L_chol=L_chol)

    # Convert to InferenceData
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive=posterior_predictive,
        prior=prior_pred_samples,
        log_likelihood={"y": ll_y, "joint_log_likelihood": ll_joint},
        constant_data={"z": z, "mu_obs": mu_obs}
    )

    # Save to NetCDF
    az.to_netcdf(idata, f"idata_{name}.nc")
    return idata

# ---------------------------
# Example of running and saving for the LCDM model
# ---------------------------
# run_and_save(lcdm_model, "lcdm")
