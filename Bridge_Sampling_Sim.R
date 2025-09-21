############################################################
# Bayesian Model Comparison for Cosmology using Bridge Sampling
#
# Description:
#   This script performs Bayesian model comparison for cosmological
#   models (ΛCDM, wCDM, and CPL) using DES-SN5YR supernova data.
#   Bridge sampling is used to estimate marginal likelihoods and
#   compute Bayes factors between the models.
############################################################


# ---------------------------
# Load required libraries
# ---------------------------
library(bridgesampling)
library(Matrix)
library(dplyr)


# ---------------------------
# Load posterior samples
# (exported from the .nc files from NumPyro/Python as CSVs)
# ---------------------------
lcdm_samples <- read.csv("posterior_lcdm.csv")
wcdm_samples <- read.csv("posterior_wcdm.csv")
cpl_samples  <- read.csv("posterior_cpl.csv")


# ---------------------------
# Load supernova data and covariance matrix
# ---------------------------
filtered_data <- read.csv("DES-SN5YR_HD+MetaData.csv")

# Extract relevant vectors
z <- filtered_data$zHD
mu_obs <- filtered_data$MU
n <- length(mu_obs)

# Load covariance matrix (exported from Python without headers)
cov_full <- as.matrix(read.csv("cov_full.csv", header = FALSE))

# Convert to sparse Matrix and stabilise with small jitter
cov_mat <- Matrix(cov_full)
diag(cov_mat) <- diag(cov_mat) + 0.001  # add jitter

# Compute Cholesky decomposition, inverse and log determinant
L <- chol(cov_mat)
cov_inv <- chol2inv(L)
log_det_cov <- 2 * sum(log(diag(L)))

cat("Covariance matrix ready.\n")
cat("Log determinant:", log_det_cov, "\n")


# ---------------------------
# Constants and helper functions
# ---------------------------
C <- 299792.458  # speed of light in km/s

# Luminosity distance function using numerical integration
luminosity_distance <- function(z_vals, E_func) {
  sapply(z_vals, function(z_val) {
    integrate(function(x) 1 / E_func(x),
              lower = 0, upper = z_val,
              rel.tol = 1e-6)$value * (1 + z_val)
  })
}


# ---------------------------
# Log-posterior functions
# (each includes likelihood + priors)
# ---------------------------

# ΛCDM
log_posterior_lcdm <- function(params, data, cov_inv, log_det_cov) {
  H0 <- params[1]
  Omega_m <- params[2]
  
  # Valid parameter region
  if (H0 <= 0 || Omega_m < 0 || Omega_m > 1) return(-Inf)
  
  # Friedmann equation for ΛCDM
  E <- function(z) sqrt(Omega_m * (1 + z)^3 + (1 - Omega_m))
  dL <- luminosity_distance(data$z, E)
  
  # Theoretical distance modulus
  mu_th <- 5 * log10((C / H0) * dL) + 25
  residuals <- matrix(data$mu - mu_th, ncol = 1)
  
  # Gaussian log-likelihood
  log_lik <- -0.5 * as.numeric(t(residuals) %*% cov_inv %*% residuals) -
    0.5 * log_det_cov - (length(residuals) / 2) * log(2 * pi)
  
  # Priors: H0 ~ LogNormal(log 70, 0.14), Ωm ~ Beta(3,7)
  log_prior <- dlnorm(H0, meanlog = log(70), sdlog = 0.5, log = TRUE) +
    dbeta(Omega_m, shape1 = 3, shape2 = 7, log = TRUE)
  
  return(log_lik + log_prior)
}


# wCDM
log_posterior_wcdm <- function(params, data, cov_inv, log_det_cov) {
  H0 <- params[1]
  Omega_m <- params[2]
  w <- params[3]
  
  if (H0 <= 0 || Omega_m < 0 || Omega_m > 1) return(-Inf)
  
  # Friedmann equation with constant w
  E <- function(z) sqrt(Omega_m * (1 + z)^3 +
                          (1 - Omega_m) * (1 + z)^(3 * (1 + w)))
  dL <- luminosity_distance(data$z, E)
  
  mu_th <- 5 * log10((C / H0) * dL) + 25
  residuals <- matrix(data$mu - mu_th, ncol = 1)
  
  log_lik <- -0.5 * as.numeric(t(residuals) %*% cov_inv %*% residuals) -
    0.5 * log_det_cov - (length(residuals) / 2) * log(2 * pi)
  
  # Priors: H0 ~ LogNormal, Ωm ~ Beta(3,7), w ~ Normal(-1, 0.5)
  log_prior <- dlnorm(H0, meanlog = log(70), sdlog = 0.5, log = TRUE) +
    dbeta(Omega_m, shape1 = 3, shape2 = 7, log = TRUE) +
    dnorm(w, mean = -1, sd = 2, log = TRUE)
  
  return(log_lik + log_prior)
}


# CPL
log_posterior_cpl <- function(params, data, cov_inv, log_det_cov) {
  H0 <- params[1]
  Omega_m <- params[2]
  w0 <- params[3]
  wa <- params[4]
  
  if (H0 <= 0 || Omega_m < 0 || Omega_m > 1) return(-Inf)
  
  # Friedmann equation with CPL parametrisation
  E <- function(z) {
    exp_part <- exp(-3 * wa * z / (1 + z))
    wz_part <- (1 + z)^(3 * (1 + w0 + wa))
    sqrt(Omega_m * (1 + z)^3 + (1 - Omega_m) * wz_part * exp_part)
  }
  
  dL <- luminosity_distance(data$z, E)
  
  mu_th <- 5 * log10((C / H0) * dL) + 25
  residuals <- matrix(data$mu - mu_th, ncol = 1)
  
  log_lik <- -0.5 * as.numeric(t(residuals) %*% cov_inv %*% residuals) -
    0.5 * log_det_cov - (length(residuals) / 2) * log(2 * pi)
  
  # Priors: H0 ~ LogNormal, Ωm ~ Beta(3,7), w0 ~ Normal(-1, 0.5), wa ~ Normal(0,1)
  log_prior <- dlnorm(H0, meanlog = log(70), sdlog = 0.5, log = TRUE) +
    dbeta(Omega_m, shape1 = 3, shape2 = 7, log = TRUE) +
    dnorm(w0, mean = -1, sd = 2, log = TRUE) +
    dnorm(wa, mean = 0, sd = 4, log = TRUE)
  
  return(log_lik + log_prior)
}


# ---------------------------
# Filter posterior samples within plausible bounds
# ---------------------------
lcdm_matrix <- as.matrix(lcdm_samples[, c("H0", "Omega_m")])
lcdm_matrix <- lcdm_matrix[
  lcdm_matrix[, "H0"] > 40 & lcdm_matrix[, "H0"] < 100 &
    lcdm_matrix[, "Omega_m"] > 0 & lcdm_matrix[, "Omega_m"] < 1,
]

wcdm_matrix <- as.matrix(wcdm_samples[, c("H0", "Omega_m", "w")])
wcdm_matrix <- wcdm_matrix[
  wcdm_matrix[, "H0"] > 40 & wcdm_matrix[, "H0"] < 100 &
    wcdm_matrix[, "Omega_m"] > 0 & wcdm_matrix[, "Omega_m"] < 1 &
    wcdm_matrix[, "w"] > -5 & wcdm_matrix[, "w"] < 3,
]

cpl_matrix <- as.matrix(cpl_samples[, c("H0", "Omega_m", "w0", "wa")])
cpl_matrix <- cpl_matrix[
  cpl_matrix[, "H0"] > 40 & cpl_matrix[, "H0"] < 100 &
    cpl_matrix[, "Omega_m"] > 0 & cpl_matrix[, "Omega_m"] < 1 &
    cpl_matrix[, "w0"] > -5 & cpl_matrix[, "w0"] < 3 &
    cpl_matrix[, "wa"] > -10 & cpl_matrix[, "wa"] < 10,
]


# ---------------------------
# Run bridge sampling
# ---------------------------
bridge_lcdm <- bridge_sampler(
  samples = lcdm_matrix,
  log_posterior = log_posterior_lcdm,
  data = list(z = z, mu = mu_obs),
  cov_inv = cov_inv,
  log_det_cov = log_det_cov,
  lb = c(H0 = 40, Omega_m = 0),
  ub = c(H0 = 100, Omega_m = 1),
  method = "normal"
)

bridge_wcdm <- bridge_sampler(
  samples = wcdm_matrix,
  log_posterior = log_posterior_wcdm,
  data = list(z = z, mu = mu_obs),
  cov_inv = cov_inv,
  log_det_cov = log_det_cov,
  lb = c(H0 = 40, Omega_m = 0, w = -5),
  ub = c(H0 = 100, Omega_m = 1, w = 3),
  method = "normal"
)

bridge_cpl <- bridge_sampler(
  samples = cpl_matrix,
  log_posterior = log_posterior_cpl,
  data = list(z = z, mu = mu_obs),
  cov_inv = cov_inv,
  log_det_cov = log_det_cov,
  lb = c(H0 = 40, Omega_m = 0, w0 = -5, wa = -10),
  ub = c(H0 = 100, Omega_m = 1, w0 = 3, wa = 10),
  method = "normal"
)


# ---------------------------
# Summaries of marginal likelihood estimates
# ---------------------------
print(summary(bridge_lcdm))
print(summary(bridge_wcdm))
print(summary(bridge_cpl))


# ---------------------------
# Bayes factors between models
# ---------------------------
bf_lcdm_vs_wcdm <- bf(bridge_lcdm, bridge_wcdm)
bf_lcdm_vs_cpl  <- bf(bridge_lcdm, bridge_cpl)
bf_wcdm_vs_cpl  <- bf(bridge_wcdm, bridge_cpl)

cat("Bayes Factor LCDM vs wCDM:", bf_lcdm_vs_wcdm$bf, "\n")
cat("Bayes Factor LCDM vs CPL: ", bf_lcdm_vs_cpl$bf, "\n")
cat("Bayes Factor wCDM vs CPL: ", bf_wcdm_vs_cpl$bf, "\n")


# ---------------------------
# Log Bayes factors (log marginal likelihood differences)
# ---------------------------
logml_lcdm <- bridge_lcdm$logml
logml_wcdm <- bridge_wcdm$logml
logml_cpl  <- bridge_cpl$logml

b12 <- logml_lcdm - logml_wcdm   # LCDM vs wCDM
b13 <- logml_lcdm - logml_cpl    # LCDM vs CPL
b23 <- logml_wcdm - logml_cpl    # wCDM vs CPL

cat("b12 (LCDM vs wCDM):", b12, "\n")
cat("b13 (LCDM vs CPL): ", b13, "\n")
cat("b23 (wCDM vs CPL):", b23, "\n")