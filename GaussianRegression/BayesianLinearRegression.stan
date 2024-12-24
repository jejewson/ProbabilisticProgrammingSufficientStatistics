data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
}

parameters {
  vector[K] b;  // regression coefficients
  real<lower=0> sigma;  // dispersion parameter
}

model {
  b ~ normal(0, 10);
  sigma ~ student_t(3, 0, 3.7);
  Y ~ normal(X*b, sigma);
}
