data {
  int<lower=1> N;  // total number of observations
  int Y[N];  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
}

parameters {
  vector[K] b;  // regression coefficients
}

model {
  b ~ normal(0, 2);
  Y ~ poisson(exp(X*b));
}
