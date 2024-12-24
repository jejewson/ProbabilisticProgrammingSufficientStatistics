// Bayesian Factor Analysis
// https://rfarouni.github.io/assets/projects/BayesianFactorAnalysis/BayesianFactorAnalysis.html
// with added
// Sufficient Statistics Likelihood
// Woodbury Decomposition Omega - https://gregorygundersen.com/blog/2018/11/30/woodbury/

functions {
   // Woodbury Identity
   matrix woodbury_inverse_broadcast(vector Psi_diag, matrix U){
      // Psi_diag is a p x 1 vector of the diagonal elements of Psi
      // U is a p x k matrix 
      // V is a k times p matrix
      // V = U'
      
      int dimensions[2] = dims(U);
      int p = dimensions[1];
      int k = dimensions[2];
      matrix[k, p] V = U';
      matrix[p, k] Psi_inv_broadcast = rep_matrix((1 ./ Psi_diag), k);
      //matrix[k, k] B = diag_matrix(rep_vector(1, k)) + V*(Psi_inv_broadcast .* U);
      matrix[k, k] B_inv = inverse(diag_matrix(rep_vector(1, k)) + V*(Psi_inv_broadcast .* U));
      
      //return (Psi_inv - Psi_inv * U * inverse(B) * V * Psi_inv);
      return (diag_matrix(Psi_inv_broadcast[,1]) - (Psi_inv_broadcast .* U) * (Psi_inv_broadcast .* (B_inv * V)')');
   }
   
}

data {
  int<lower=1> N;                // number of 
  int<lower=1> P;                // number of 
  matrix[N,P] Y;                 // data matrix of order [N,P]
  int<lower=1> D;              // number of latent dimensions 
}

transformed data {
  int<lower=1> M;
  vector[P] mu;
  // Sufficient Statistics
  matrix[P, P] S_bar; 
  M  = D*(P-D)+ D*(D-1)/2;  // number of non-zero loadings
  mu = rep_vector(0.0,P);
  S_bar = Y'*Y/N;
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
  cov_matrix[P] Omega;   //precision mat
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
  Omega = woodbury_inverse_broadcast(psi, L);

}
model {
   // the hyperpriors 
   target += cauchy_lpdf(mu_psi | 0, 1);
   target += cauchy_lpdf(sigma_psi |0, 1);
   target += cauchy_lpdf(mu_lt | 0, 1);
   target += cauchy_lpdf(sigma_lt | 0, 1);
   // the priors 
   target += cauchy_lpdf(L_d | 0,3);
   target += cauchy_lpdf(L_t | mu_lt, sigma_lt);
   target += cauchy_lpdf(psi | mu_psi, sigma_psi);
   //The likelihood
   // non-zero mean
   //target += N*(-0.5*P*log(2*pi()) + 0.5*log_determinant(Omega)  - 0.5*(x_bar - mu)'*Omega*(x_bar - mu) - 0.5*trace(S_bar*Omega));
   // zero-mean
   target += 0.5*N*(-P*log(2*pi()) + log_determinant(Omega) - trace(S_bar*Omega));
}
