
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  array[N] int<lower=1> J_1;  // grouping indicator per observation
}

parameters {
  vector[K] b;  // regression coefficients
  real<lower=0> sigma;  // dispersion parameter
  real<lower=0> sd_1;  // group-level standard deviations
  vector[N_1] z_1;  // standardized group-level effects
}

model {
  // Prior
  b ~ normal(0, 10);
  sigma ~ student_t(3, 0, 2.5);
  sd_1 ~ student_t( 3, 0, 2.5);
  
  // Random Effects
  target += normal_lpdf(z_1 | 0, sd_1);
  
  // likelihood
  target += normal_lpdf(Y | X*b + z_1[J_1], sigma);
}

