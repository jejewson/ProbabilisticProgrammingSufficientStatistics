// Bayesian Factor Analysis
// https://rfarouni.github.io/assets/projects/BayesianFactorAnalysis/BayesianFactorAnalysis.html

data {
  int<lower=1> N;                // number of observations
  int<lower=1> P;                // dimension of observations
  vector[P] Y[N];                 // data matrix of order [N,P]
  int<lower=1> D;              // number of latent dimensions 
}

transformed data {
  int<lower=1> M;
  vector[P] mu;
  M  = D*(P-D)+ D*(D-1)/2;  // number of non-zero loadings
  mu = rep_vector(0.0,P);
}

parameters {    
  vector[M] L_t;   // lower diagonal elements of L
  vector<lower=0>[D] L_d;   // lower diagonal elements of L
  vector<lower=0>[P] psi;         // vector of variances
  real<lower=0>   mu_psi;
  real<lower=0>  sigma_psi;
  real   mu_lt;
  real<lower=0>  sigma_lt;
}
transformed parameters{
  cholesky_factor_cov[P,D] L;  //lower triangular factor loadings Matrix 
  cov_matrix[P] Q;   //Covariance mat
{
  int idx2 = 0;
  for(i in 1:P){
    for(j in (i+1):D){
      L[i,j] = 0; //constrain the upper triangular elements to zero 
    }
  }
  for (j in 1:D) {
      L[j,j] = L_d[j];
    for (i in (j+1):P) {
      idx2 += 1;
      L[i,j] = L_t[idx2];
    } 
  }
} 
  Q = L*L' + diag_matrix(psi); 

}
model {
   // the hyperpriors 
   target +=  cauchy_lpdf(mu_psi | 0, 1);
   target +=  cauchy_lpdf(sigma_psi | 0,1);
   target +=  cauchy_lpdf(mu_lt | 0, 1);
   target +=  cauchy_lpdf(sigma_lt | 0,1);
   // the priors 
   target +=  cauchy_lpdf(L_d | 0,3);
   target +=  cauchy_lpdf(L_t | mu_lt,sigma_lt);
   target +=  cauchy_lpdf(psi | mu_psi,sigma_psi);
   //The likelihood
   
   target += multi_normal_lpdf(Y | mu, Q); 
}
