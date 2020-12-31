#ifndef INCLUDE_UTIL_TYPES_H_
#define INCLUDE_UTIL_TYPES_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

// Multivariate gaussian struct which stores a state vector mu and covariance matrix sigma
struct Gaussian {
  Eigen::VectorXf mu;       // (state_dim_, 1) State vector
  Eigen::MatrixXf sigma;  // (state_dim_, state_dim_) Covariance Matrix
};

// Particle struct which stores a state vector and weight
struct Particle {
  Eigen::VectorXf state;
  double weight;
};

#endif // INCLUDE_UTIL_TYPES_H_
