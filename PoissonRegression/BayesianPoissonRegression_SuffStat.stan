data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  //int Y[N];  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
}

transformed data {
  //vector[K] means_X;  // column means of X before centering
  real Syy;
  row_vector[K] Syx;

  Syy = sum(lgamma(Y + 1));
  Syx = Y'*X;
}

parameters {
  vector[K] b;  // regression coefficients
}

transformed parameters {   
}

model {
  // Priors:
  target += normal_lpdf(b | 0, 2);
  // Likelihood:
  target += Syx*b - Syy - sum(exp(X*b));
}



