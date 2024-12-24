data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
}

transformed data {
  //vector[K] means_X;  // column means of X before centering
  real Syy;
  row_vector[K] Syx;
  matrix[K,K] Sxx;

  Syy = Y'*Y;
  Syx = Y'*X;
  Sxx = crossprod(X);
}

parameters {
  vector[K] b;  // regression coefficients
  real<lower=0> sigma;  // dispersion parameter
}

transformed parameters {   
}

model {
  // Priors:
  target += normal_lpdf(b | 0, 10);
  target += student_t_lpdf(sigma | 3, 0, 3.7)
    - 1 * student_t_lccdf(0 | 3, 0, 3.7);
  // Likelihood:
  target += -N*log(sigma)-(Syy-2*Syx*b+b'*Sxx*b)/(2*sigma^2);
}



