/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// std
#include <random>
#include <cmath>
#include <vector>

/// Third Party
#include <Eigen/Core>

/// Bayesian
#include "gaussian_utils.h"

float GaussianSample1D(float mu, float sigma) {
  std::random_device rand;
  std::default_random_engine generator(rand());
  std::normal_distribution<float> distribution(mu, sigma);
  return distribution(generator);
}

Eigen::VectorXf NormalVector1D (float mu, float sigma, uint32_t dims) {
  Eigen::VectorXf out(dims);
  for (uint32_t i = 0; i < dims; i++) {
      out(i) = GaussianSample1D(mu, sigma);
  }
  return out;
}

Eigen::VectorXf MultivariateNormalVector(const Eigen::VectorXf& mu, const Eigen::VectorXf& sigmas_vector) {
  Eigen::VectorXf out (mu.size());
  for (int i = 0; i < mu.size(); i++) {
      out(i) = GaussianSample1D(mu(i), sigmas_vector(i));
  }
  return out;
}

Eigen::MatrixXf MultivariateNormalMatrix(const Eigen::VectorXf& mu, const Eigen::VectorXf &sigmas_vector, int num_samples) {
  const int dimension = mu.size();
  Eigen::MatrixXf out(dimension, num_samples);

  for (int sample_index = 0; sample_index < num_samples; sample_index++) {
      out.row(sample_index) = MultivariateNormalVector(mu, sigmas_vector);
  }
  return out;
}

Eigen::MatrixXf MultivariateNormalMatrix(const Gaussian& gaussian, int num_samples) {
  return MultivariateNormalMatrix(gaussian.mu, GetStdDeviationsVector(gaussian.sigma), num_samples);
}

Eigen::VectorXf GetStdDeviationsVector(const Eigen::MatrixXf &sigma) {
  Eigen::VectorXf variances_sq = sigma.diagonal();
  return variances_sq.cwiseSqrt();
}
