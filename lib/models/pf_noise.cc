/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// Bayesian
#include "pf_noise.h"
#include "../util/gaussian_utils.h"

void GaussianNoise::AddNoise(Eigen::VectorXf* state) {
  Eigen::VectorXf zeros_vector{Eigen::VectorXf::Zero(sigmas_vector_.size())};
  *state+= MultivariateNormalVector(zeros_vector, sigmas_vector_);
}
