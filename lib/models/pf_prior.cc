/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// std
#include <vector>

/// Third Party
#include <Eigen/Core>

// Bayesian
#include "../util/gaussian_utils.h"
#include "../util/types.h"
#include "pf_prior.h"

void GaussianPrior::SamplePrior(std::vector<Particle>* prior, const int num_samples) {
  if (prior == nullptr) { return; }
  Eigen::MatrixXf init_particle_data = MultivariateNormalMatrix(mu_, sigma_, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    prior->at(i).state = init_particle_data.row(i);
    prior->at(i).weight = 1.0 / num_samples;
  }
}
