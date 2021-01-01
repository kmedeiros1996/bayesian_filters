#ifndef INCLUDE_MODELS_PF_RESAMPLE_H_
#define INCLUDE_MODELS_PF_RESAMPLE_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// Bayesian
#include "../util/types.h"

/*
* @brief abstract resampler called by the particle filter.
* Samples from prior distribution and stores samples in new_particles.
*/
class Resampler {
public:
  virtual void Resample(const std::vector<Particle> *prior, std::vector<Particle>* new_particles) = 0;
};

/*
* Particle sampler which implements a discrete resampling from a std::discrete_distribution.
*/
class DiscreteResampler : public Resampler {
public:
  void Resample(const std::vector<Particle> *prior, std::vector<Particle>* new_particles);
};

/*
* Particle sampler which implements the Low Variance Resampling algorithm.
*/
class LowVarianceResampler : public Resampler {
public:
  void Resample(const std::vector<Particle> *prior, std::vector<Particle>* new_particles);
};

#endif // INCLUDE_MODELS_PF_RESAMPLE_H_
