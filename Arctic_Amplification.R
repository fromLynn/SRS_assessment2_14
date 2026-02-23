#=============================================================================
# PROJECT OVERVIEW: Spatiotemporal Modeling of Arctic Amplification
#=============================================================================
# 
# 1. RESEARCH OBJECTIVE:
# To investigate the "Arctic Amplification" effect—testing whether higher 
# latitudes (polar regions) experience accelerated temperature increases 
# due to global warming compared to lower latitudes.
#
# 2. DATA SCOPE:
# Extracted a longitudinal transect data slice (15°N to 80°N) that passes 
# directly through Scotland, covering from North Africa to the Arctic Circle.
# 
# 3. METHODOLOGY & TRAIN/TEST SPLIT:
# - Training Set: The first 100 years of data to train the Bayesian models.
# - Test Set: The remaining 12 years of data for out-of-sample forecasting.
#
# 4. MODEL COMPARISON:
# - Model A (Benchmark): A Bayesian spatiotemporal model with a polar 
#   interaction term (Year * Latitude), assuming independent errors.
# - Model B (Advanced) : Adds an Autoregressive AR(1) error structure to 
#   capture the "thermal memory" (inertia) of the climate system.
#
# 5. EVALUATION & OUTPUTS:
# - Metrics: Compares models using DIC (in-sample complexity/fit), alongside 
#   Pseudo/Adjusted R-squared, RMSE, and MAE for out-of-sample accuracy.
# - Visualizations: Generates Goodness-of-Fit scatter plots and 12-year 
#   continuous time-series forecasting plots for the Test Set.
#=============================================================================

#=============================================================================
# MODEL SPECIFICATION: Bayesian Spatiotemporal Model with AR(1) Errors
#=============================================================================
#
# 1. CORE MATHEMATICAL EQUATIONS
# This study constructs a Bayesian spatiotemporal model with an AR(1) error 
# structure, consisting of three core equations:
#
# Observation Equation:
# Y_{i,t} = \mu_{i,t} + \epsilon_{i,t}
#
# Deterministic Mean Equation:
# \mu_{i,t} = \beta_0 + MonthEffect_{m(t)} + \beta_1 * Year_{t} + 
#             \beta_2 * Lat_{i} + \beta_{polar} * (Year_{t} * Lat_{i}) + \alpha_i
#
# Error Equation (AR(1)) & Distribution Assumptions:
# \epsilon_{i,t} = \rho * \epsilon_{i,t-1} + u_{i,t}
# u_{i,t} ~ N(0, \sigma^2_{obs})
# \alpha_i ~ N(0, \sigma^2_{\alpha})
#
#-----------------------------------------------------------------------------
# 2. PARAMETER DEFINITIONS & SCIENTIFIC INTERPRETATIONS
#-----------------------------------------------------------------------------
# [Dependent Variable & Mean Structure]
# Y_{i,t}       : Actual observed absolute temperature (°C) at location i in month t.
# \mu_{i,t}     : Baseline expected temperature (regular temperature excluding 
#                 short-term abnormal fluctuations).
# \beta_0       : Global intercept. The average temperature when continuous 
#                 variables (year & lat) are at their mean, in the baseline Jan.
# MonthEffect   : Seasonal fixed effect. Captures the inherent temperature 
#                 difference of each month relative to January (set to 0).
# \beta_1       : Baseline global warming trend. Measures the average temperature 
#                 increase per year across the study region.
# \beta_2       : Baseline latitude effect. Measures the average temperature 
#                 decrease when moving north (increasing latitude).
#
# [Core Innovation: Arctic Amplification Effect]
# \beta_{polar} : Spatiotemporal interaction coefficient. Used to test whether 
#                 the "warming rate accelerates with higher latitudes." If 
#                 significantly > 0, it statistically confirms "Arctic 
#                 Amplification" driven by phenomena like sea ice melting.
#
# [Spatial and Temporal Dynamic Adjustments]
# \alpha_i      : Spatial heterogeneity intercept (random effect). Absorbs local 
#                 microclimate differences (e.g., elevation, topography) that 
#                 cannot be explained by latitude alone.
# \epsilon_{i,t}: Total residual. The difference between actual observed 
#                 temperature and the baseline expected temperature.
# \rho (Rho)    : AR(1) coefficient. Captures the "thermal memory inertia" of 
#                 the climate system, measuring the intensity of the previous 
#                 month's temperature anomaly carrying over to the current month.
# u_{i,t}       : White noise. The unpredictable fluctuation caused by purely 
#                 random weather events in the current month, after removing 
#                 trends, seasonal cycles, and memory effects.
#=============================================================================


# load the `ncdf4` and the `CFtime` packages
install.packages(c("ncdf4", "CFtime"))
library(ncdf4)
library(CFtime)

library(lattice)
library(RColorBrewer)

# set path and filename
ncpath <- "C:/Users/Lenovo/Desktop"
ncname  <- "temp_data_SRS"
ncfname <- file.path(ncpath, paste0(ncname, ".nc"))
dname   <- "tmp"
ncin <- nc_open(ncfname)
print(ncin)

#=============================================================================

# load("Arctic_Amplification_Results.RData") # you do not need to run block 0.5,1,2,3 after loading

#=============================================================================
# BLOCK 0: Load Required Packages
#=============================================================================
install.packages(c("dplyr", "rjags", "coda", "ggplot2", "tidyr"))

library(dplyr)
library(rjags)
library(coda)
library(ggplot2)
library(tidyr)


#=============================================================================
# BLOCK 0.5: Load NetCDF and Extract Longitudinal Transect (Data Slicing)
#=============================================================================
# Load required packages
library(ncdf4)
library(dplyr)

# 1. Set path and filename (Using your exact setup)
ncpath <- "C:/Users/Lenovo/Desktop"
ncname <- "temp_data_SRS"
ncfname <- file.path(ncpath, paste0(ncname, ".nc"))
dname   <- "tmp"

# 2. Open the NetCDF file
ncin <- nc_open(ncfname)

# 3. Extract the full dimension vectors
lon <- ncvar_get(ncin, "lon")
lat <- ncvar_get(ncin, "lat")
time_var <- ncvar_get(ncin, "time") # Named time_var to avoid confusion

# 4. Find the exact indices for our "North Africa to Arctic" slice
# Longitude: -5 to 5 | Latitude: 15 to 80
lon_indices <- which(lon >= -5 & lon <= 5)
lat_indices <- which(lat >= 15 & lat <= 80)

# Check if the dataset actually covers this area to prevent errors
if(length(lon_indices) == 0 | length(lat_indices) == 0) {
  stop("ERROR: The specified longitude/latitude range is out of bounds for this .nc file.")
}

# 5. Extract ONLY the targeted slice using 'start' and 'count'
# This is a memory-efficient step: it prevents RAM overflow!
# start = c(start_lon, start_lat, start_time)
# count = c(num_lon, num_lat, num_time) | -1 means all available time steps
tmp_slice <- ncvar_get(ncin, dname, 
                       start = c(lon_indices[1], lat_indices[1], 1),
                       count = c(length(lon_indices), length(lat_indices), -1))

# Close the NetCDF connection immediately to free up system memory
nc_close(ncin)

# 6. Rebuild the coordinate grid for the extracted slice
lon_grid <- lon[lon_indices]
lat_grid <- lat[lat_indices]

# Use expand.grid to create all possible combinations of coordinates and time
grid_coords <- expand.grid(lon = lon_grid, lat = lat_grid, time = time_var)

# 7. Flatten the 3D temperature array and bind it to the coordinates
# This creates the target 'grid' dataframe required for subsequent analysis
grid <- grid_coords %>%
  mutate(temp = as.vector(tmp_slice)) %>%
  # Drop ocean points and missing values instantly to reduce dataframe size
  filter(!is.na(temp))

cat("\n--- Slice Extraction Successful ---\n")
cat("Total active land data points extracted in 'grid':", nrow(grid), "\n")
#=============================================================================
#=============================================================================
# BLOCK 1: Data Preparation, Transect Slicing, and Train/Test Split
#=============================================================================
# Assuming you have your flattened raw data in a dataframe called `grid` 
# containing: lon, lat, time (days), and temp.

# 1. Filter for the "North Africa - Scotland - Arctic" Longitudinal Transect
df_transect <- grid %>%
  filter(lon >= -5 & lon <= 5) %>%
  filter(lat >= 15 & lat <= 80) %>%
  filter(!is.na(temp)) %>% # Remove ocean/missing data
  mutate(Loc_ID_raw = as.numeric(factor(paste(lon, lat, sep="_"))))

# 2. Stratified Random Sampling (Select 50 locations to keep MCMC time manageable)
set.seed(123) # For reproducibility
sampled_locations <- df_transect %>%
  distinct(Loc_ID_raw, lat) %>%
  mutate(Lat_Zone = cut(lat, breaks = c(15, 40, 60, 80), labels = c("Low", "Mid", "High"))) %>%
  group_by(Lat_Zone) %>%
  sample_n(size = 17, replace = TRUE) %>% # Sample ~17 from each zone (total 51)
  distinct(Loc_ID_raw) %>%
  pull(Loc_ID_raw)

df_model <- df_transect %>%
  filter(Loc_ID_raw %in% sampled_locations) %>%
  # Re-index Loc_ID from 1 to N_locs cleanly
  mutate(Loc_ID = as.numeric(factor(Loc_ID_raw))) %>%
  arrange(Loc_ID, time) %>%
  group_by(Loc_ID) %>%
  mutate(
    Month_Index = rep(1:12, length.out = n()),
    Time_Step = row_number(),
    # Assuming data starts in Jan 1901
    Year_approx = 1901 + (Time_Step - 1) / 12 
  ) %>%
  ungroup()

# 3. Standardize Continuous Variables (CRITICAL for interaction terms in JAGS)
# This prevents numerical overflow and helps MCMC chains converge quickly
mean_year <- mean(df_model$Year_approx)
sd_year <- sd(df_model$Year_approx)
mean_lat <- mean(df_model$lat)
sd_lat <- sd(df_model$lat)

df_model <- df_model %>%
  mutate(
    Year_std = (Year_approx - mean_year) / sd_year,
    Lat_std = (lat - mean_lat) / sd_lat
  )

# 4. Split into Training (1901-2000) and Test (2001-2012)
df_train <- df_model %>% filter(Year_approx < 2001)
df_test <- df_model %>% filter(Year_approx >= 2001)

# 5. Prepare JAGS Data List (Using Training Data)
locations <- df_train %>% distinct(Loc_ID, Lat_std) %>% arrange(Loc_ID)
N_locs <- nrow(locations)
N_time_train <- length(unique(df_train$time))

Y_train_matrix <- matrix(df_train$temp, nrow = N_locs, ncol = N_time_train, byrow = TRUE)

jags_data <- list(
  Y = Y_train_matrix,
  Lat_std = locations$Lat_std,
  Year_std = unique(df_train$Year_std), # Vector of standardized years for each timestep
  Month = (df_train %>% filter(Loc_ID == 1))$Month_Index,
  N_locs = N_locs,
  N_time = N_time_train
)

#=============================================================================
# BLOCK 2: Define JAGS Models (Model A: Benchmark, Model B: AR1)
#=============================================================================

# MODEL A: Benchmark (Polar Interaction, NO AR1)
model_A_string <- "
model {
  for (i in 1:N_locs) {
    for (t in 1:N_time) {
      Y[i, t] ~ dnorm(mu[i, t], tau_obs)
      mu[i, t] <- beta0 + MonthEffect[Month[t]] + beta1 * Year_std[t] + 
                  beta2 * Lat_std[i] + beta_polar * (Year_std[t] * Lat_std[i]) + alpha[i]
    }
    alpha[i] ~ dnorm(0, tau_alpha)
  }
  beta0 ~ dnorm(0, 0.001)
  beta1 ~ dnorm(0, 0.001) # Global warming trend
  beta2 ~ dnorm(0, 0.001) # Baseline latitude effect
  beta_polar ~ dnorm(0, 0.001) # Arctic Amplification Interaction!
  
  MonthEffect[1] <- 0
  for (m in 2:12) { MonthEffect[m] ~ dnorm(0, 0.001) }
  
  tau_obs ~ dgamma(0.01, 0.01)
  tau_alpha ~ dgamma(0.01, 0.01)
}
"
writeLines(model_A_string, con = "model_A.jags")

# MODEL B: Advanced (Polar Interaction + AR1)
model_B_string <- "
model {
  for (i in 1:N_locs) {
    # t = 1
    Y[i, 1] ~ dnorm(mu[i, 1], tau_obs)
    mu[i, 1] <- beta0 + MonthEffect[Month[1]] + beta1 * Year_std[1] + 
                beta2 * Lat_std[i] + beta_polar * (Year_std[1] * Lat_std[i]) + alpha[i]
    epsilon[i, 1] <- Y[i, 1] - mu[i, 1]
    
    # t > 1
    for (t in 2:N_time) {
      Y[i, t] ~ dnorm(mu[i, t] + rho * epsilon[i, t-1], tau_obs)
      mu[i, t] <- beta0 + MonthEffect[Month[t]] + beta1 * Year_std[t] + 
                  beta2 * Lat_std[i] + beta_polar * (Year_std[t] * Lat_std[i]) + alpha[i]
      epsilon[i, t] <- Y[i, t] - mu[i, t]
    }
    alpha[i] ~ dnorm(0, tau_alpha)
  }
  beta0 ~ dnorm(0, 0.001)
  beta1 ~ dnorm(0, 0.001)
  beta2 ~ dnorm(0, 0.001)
  beta_polar ~ dnorm(0, 0.001)
  rho ~ dunif(-1, 1) # AR1 term
  
  MonthEffect[1] <- 0
  for (m in 2:12) { MonthEffect[m] ~ dnorm(0, 0.001) }
  tau_obs ~ dgamma(0.01, 0.01)
  tau_alpha ~ dgamma(0.01, 0.01)
}
"
writeLines(model_B_string, con = "model_B.jags")

#=============================================================================
# BLOCK 3: Run Models and Calculate DIC
#=============================================================================
params_A <- c("beta0", "beta1", "beta2", "beta_polar")
params_B <- c("beta0", "beta1", "beta2", "beta_polar", "rho")

cat("\n--- Training Model A (Benchmark) ---\n")
jags_A <- jags.model("model_A.jags", data = jags_data, n.chains = 3, n.adapt = 1000)
update(jags_A, 2000)
dic_A <- dic.samples(jags_A, 2000)
samples_A <- coda.samples(jags_A, variable.names = c(params_A, "MonthEffect", "alpha"), n.iter = 2000)

cat("\n--- Training Model B (AR1) ---\n")
jags_B <- jags.model("model_B.jags", data = jags_data, n.chains = 3, n.adapt = 1000)
update(jags_B, 2000)
dic_B <- dic.samples(jags_B, 2000)
samples_B <- coda.samples(jags_B, variable.names = c(params_B, "MonthEffect", "alpha"), n.iter = 2000)

#=============================================================================
# BLOCK 4: Save Environment!
#=============================================================================
save.image(file = "Arctic_Amplification_Results.RData")
cat("\n--- Workspace Saved ---\n")

#=============================================================================
# BLOCK 5: Generate Parameter Table with Bayesian Significance Stars
#=============================================================================
# Helper function to calculate Bayesian significance stars based on Credible Intervals
format_param_stars <- function(samples, param_name) {
  # Extract the MCMC chain for the specific parameter
  chain <- as.matrix(samples)[, param_name]
  mean_val <- mean(chain)
  
  # Calculate 90%, 95%, and 99% Credible Intervals
  q90 <- quantile(chain, probs = c(0.05, 0.95))
  q95 <- quantile(chain, probs = c(0.025, 0.975))
  q99 <- quantile(chain, probs = c(0.005, 0.995))
  
  # Determine significance stars (Check if interval excludes zero)
  sig_star <- ""
  if (sign(q99[1]) == sign(q99[2])) {
    sig_star <- "***" # Excludes 0 at 99% CI (~ 1% significance)
  } else if (sign(q95[1]) == sign(q95[2])) {
    sig_star <- "**"  # Excludes 0 at 95% CI (~ 5% significance)
  } else if (sign(q90[1]) == sign(q90[2])) {
    sig_star <- "*"   # Excludes 0 at 90% CI (~ 10% significance)
  }
  
  # Output format: Mean Star [95% Lower, 95% Upper]
  sprintf("%8.4f %-3s [%.4f, %.4f]", mean_val, sig_star, q95[1], q95[2])
}

cat("\n==================== PARAMETER ESTIMATION ====================\n")
cat("Significance levels: * (10%), ** (5%), *** (1%)\n")
cat("Note: Significance is determined by 90%, 95%, and 99% Bayesian Credible Intervals excluding zero.\n\n")

cat("MODEL A (Benchmark - No AR1):\n")
for(p in params_A) cat(sprintf("%-12s: %s\n", p, format_param_stars(samples_A, p)))
cat("DIC Score   :", sum(dic_A$deviance) + sum(dic_A$penalty), "\n\n")

cat("MODEL B (Advanced - With AR1):\n")
for(p in params_B) cat(sprintf("%-12s: %s\n", p, format_param_stars(samples_B, p)))
cat("DIC Score   :", sum(dic_B$deviance) + sum(dic_B$penalty), "\n")
cat("==============================================================\n")


#=============================================================================
# BLOCK 6: Full Predictions and Out-of-Sample Metrics (R2, RMSE, MAE)
#=============================================================================
# Extract means for Model B
post_B <- summary(samples_B)$statistics[, "Mean"]
b0 <- post_B["beta0"]
b1 <- post_B["beta1"]
b2 <- post_B["beta2"]
bp <- post_B["beta_polar"]
rho <- post_B["rho"]
month_eff <- c(0, sapply(2:12, function(m) post_B[paste0("MonthEffect[", m, "]")]))
alpha_eff <- sapply(1:N_locs, function(i) post_B[paste0("alpha[", i, "]")])

# Predict continuously over the FULL dataset
df_pred_all <- df_model %>%
  mutate(
    mu_hat = b0 + month_eff[Month_Index] + b1*Year_std + b2*Lat_std + bp*(Year_std*Lat_std) + alpha_eff[Loc_ID],
    epsilon_hat = temp - mu_hat
  ) %>%
  group_by(Loc_ID) %>%
  mutate(
    Y_fitted_B = mu_hat + rho * lag(epsilon_hat, default = 0)
  ) %>%
  ungroup()

# Separate results
train_res <- df_pred_all %>% filter(Year_approx < 2001)
test_res <- df_pred_all %>% filter(Year_approx >= 2001)

# --- Calculate Metrics (R2, RMSE, MAE) ---
# 1. Pseudo R-squared
r2_train_B <- cor(train_res$temp, train_res$Y_fitted_B, use = "complete.obs")^2
r2_test_B <- cor(test_res$temp, test_res$Y_fitted_B, use = "complete.obs")^2

# 2. Root Mean Squared Error (RMSE)
rmse_train_B <- sqrt(mean((train_res$temp - train_res$Y_fitted_B)^2, na.rm = TRUE))
rmse_test_B <- sqrt(mean((test_res$temp - test_res$Y_fitted_B)^2, na.rm = TRUE))

# 3. Mean Absolute Error (MAE)
mae_train_B <- mean(abs(train_res$temp - train_res$Y_fitted_B), na.rm = TRUE)
mae_test_B <- mean(abs(test_res$temp - test_res$Y_fitted_B), na.rm = TRUE)

cat("\n--- Out-of-Sample Performance Evaluation (Model B) ---\n")
cat(sprintf("%-15s | %-12s | %-12s\n", "Metric", "Train (100y)", "Test (12y)"))
cat("----------------|--------------|--------------\n")
cat(sprintf("%-15s | %-12.4f | %-12.4f\n", "Pseudo R-squared", r2_train_B, r2_test_B))
cat(sprintf("%-15s | %-12.4f | %-12.4f\n", "RMSE (°C)", rmse_train_B, rmse_test_B))
cat(sprintf("%-15s | %-12.4f | %-12.4f\n", "MAE (°C)", mae_train_B, mae_test_B))
cat("----------------------------------------------\n")

#=============================================================================
# BLOCK 7: Visualizations
#=============================================================================
# 1. Test Set Scatter Plot (Goodness-of-Fit)
p_scatter <- ggplot(test_res, aes(x = Y_fitted_B, y = temp)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Test Set (2001-2012): Actual vs Predicted Absolute Temp", 
    subtitle = paste("Model B (AR1 + Polar Interaction) | Pseudo R-squared =", round(r2_test_B, 4)), 
    x = "Predicted Temperature (°C)", 
    y = "Actual Temperature (°C)"
  ) + theme_minimal()

print(p_scatter)

# 2. 12-Year Full Time Series Plot for the Test Set (Select High Latitude Location)
# Find a location deep in the Arctic/Scotland region for best visualization
high_lat_loc <- (locations %>% arrange(desc(Lat_std)) %>% pull(Loc_ID))[1]
test_ts_single <- test_res %>% filter(Loc_ID == high_lat_loc)

p_ts <- ggplot(test_ts_single, aes(x = Year_approx)) +
  geom_line(aes(y = temp, color = "Actual Temp"), linewidth = 0.8, alpha = 0.7) +
  geom_line(aes(y = Y_fitted_B, color = "Predicted Temp"), linewidth = 0.8, alpha = 0.9, linetype = "twodash") +
  scale_color_manual(values = c("Actual Temp" = "black", "Predicted Temp" = "red")) +
  scale_x_continuous(breaks = seq(2001, 2012, by = 1)) +
  labs(
    title = "12-Year Forecasting (Test Set): High Latitude Location", 
    subtitle = "One-Step-Ahead Prediction tracking extreme seasonal fluctuations", 
    x = "Year", y = "Temperature (°C)", color = "Legend"
  ) +
  theme_minimal() + theme(legend.position = "bottom")

print(p_ts)
#=============================================================================