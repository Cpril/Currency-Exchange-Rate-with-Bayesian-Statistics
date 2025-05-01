data {
  int<lower=1> n;     // number of observations
  vector[n] delta_exch_rate;    // response
  vector[n] Dif_Inflation_Rate;     // predictor 2
  vector[n] Dif_GDP;     // predictor 3
}
parameters {
  real<lower=0> sigma; // std of response, single continuous value
  real b0;
  real b1;
  real b2;
}
model {
  vector[n] mu; // vector of n values: expected 
  for (i in 1:n) { // loop over the n cases in the dataset to estimate mu_i values
    mu[i] = b0 + b1 * Dif_Inflation_Rate[i] + b2 * Dif_GDP[i]; 
  }
  b0 ~ normal(0, 1); // prior for intercept
  b1 ~ normal(-0.5, 1); // prior for both slope 2 (Inflation)
  b2 ~ normal(0.5, 1); // prior for both slope 3 (GDP)
  sigma ~ exponential(1); // prior for sigma
  delta_exch_rate ~ normal(mu, sigma); // defining likelihood in terms of mus and sigma
}
generated quantities {
vector[n] log_lik;
  for (i in 1:n) {
    log_lik[i] = normal_lpdf(delta_exch_rate[i] | b0 + b1 * Dif_Inflation_Rate[i] + b2 * Dif_GDP[i],
    sigma); 
  } 
}
