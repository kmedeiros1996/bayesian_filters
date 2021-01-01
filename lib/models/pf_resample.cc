/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// std
#include <random>

// Bayesian
#include "pf_resample.h"

void DiscreteResampler::Resample(const std::vector<Particle> *prior, std::vector<Particle>* new_particles) {
  if (new_particles == nullptr) { return; }

  std::vector<float> weights(prior->size());
  for (size_t i = 0; i < prior->size(); i++) {
    weights[i] = prior->at(i).weight;
  }
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  std::random_device rand;
  std::default_random_engine generator(rand());

  for (size_t i = 0; i < prior->size(); i++) {
    int sample_index = distribution(generator);
    new_particles->at(i) = prior->at(sample_index);
  }
}

void LowVarianceResampler::Resample(const std::vector<Particle> *prior, std::vector<Particle>* new_particles) {
  if (new_particles == nullptr) { return; }
  const int sample_size = prior->size();
  const double kUpperSampleBound = 1.0 / sample_size;
  std::random_device rand;
  std::default_random_engine generator(rand());
  std::uniform_real_distribution<double> distribution (0.0, kUpperSampleBound);

  double r = distribution(generator);
  double w = 0.0;
  size_t i = 1;
  size_t j = 1;

  for (size_t m = 1; m < sample_size; m++) {
    double U = r + (m - 1) / sample_size;
    while (U > w) {
      i++;
      w += prior->at(i).weight;
    }
    new_particles->at(j) = prior->at(i);
    j++;
  }
}
