
functions {
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  array[N] int<lower=1> J_1;  // grouping indicator per observation
}
transformed data {
  real Syy;                  // Y' * Y
  row_vector[K] Syx;         // Y' * X
  matrix[K, K] Sxx;          // X' * X
  vector[N_1] u_count;       // Number of observations in each group
  vector[N_1] u_sumY;        // Sum of Y for each group
  matrix[N_1, K] u_sumX;     // Sum of X for each group

  
  Syy = dot_self(Y);         // Equivalent to Y' * Y
  Syx = Y' * X;              // Equivalent to Y' * X
  Sxx = crossprod(X);        // Equivalent to X' * X
  
  u_count = rep_vector(0.0, N_1);
  u_sumY = rep_vector(0.0, N_1);
  u_sumX = rep_matrix(0.0, N_1, K);
  
  for (n in 1:N) {
    u_count[J_1[n]] += 1;
    u_sumY[J_1[n]] += Y[n];
    u_sumX[J_1[n], ] += X[n, ];
  }


}
parameters {
  vector[K] b;  // regression coefficients
  real<lower=0> sigma;  // dispersion parameter
  real<lower=0> sd_1;  // group-level standard deviations
  vector[N_1] z_1;  // standardized group-level effects
}
transformed parameters {
  vector[N_1] r_1_1;  // actual group-level effects
  r_1_1 = (sd_1 * (z_1));
  
}

model {

  // Adjust sufficient statistics for the group-level effects
  real Syy_adjusted = Syy - 2 * r_1_1' * u_sumY + (r_1_1^2)' * u_count;
  row_vector[K] Syx_adjusted = Syx - r_1_1' * u_sumX;


  // Likelihood using sufficient statistics
  target += -N*log(sigma) 
            - (Syy_adjusted - 2 * Syx_adjusted * b + b' * Sxx * b) / (2 * sigma^2);

  // priors including constants
  target += normal_lpdf(b | 0, 10);
  target += student_t_lpdf(sigma | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  target += student_t_lpdf(sd_1 | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  target += std_normal_lpdf(z_1);
}
generated quantities {
}
