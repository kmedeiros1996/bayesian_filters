#ifndef INCLUDE_FILTERS_PARTICLE_FILTER_H_
#define INCLUDE_FILTERS_PARTICLE_FILTER_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// std
#include <memory.h>

// bayesian
#include "bayesian_filter.h"
#include "../models/pf_models.h"
#include "../models/pf_models.h"
#include "../models/pf_noise.h"
#include "../models/pf_prior.h"
#include "../models/pf_resample.h"
#include "../models/pf_weight.h"
/*
* The particle filter makes use Markov Chain Monte-Carlo methods to estimate an arbitrary distribution from random sampling.
*/
class ParticleFilter : public BayesianFilter {
public:

  /*
  * @brief Calls the base Bayesian filter constructor and dynamically allocates a state space vector of size num_particles_
  */
  ParticleFilter(PFDynamicsModel *dynamics_model,
                 PFMeasurementModel *measurement_model,
                 size_t num_particles,
                 PriorFunction *prior_fn,
                 NoiseFunction *noise_fn,
                 WeightFunction *weight_fn,
                 Resampler *resample_fn_);

  ~ParticleFilter();

  /*
  * @brief Calls BayesianFilter::Init along with sampling the state space via the prior_fn_
  */
  void Init(const Eigen::VectorXf& init_state, const Eigen::MatrixXf &init_covariance);

  /*
  * @brief predict new state for each particle via dynamics_model_.
  * If a NoiseFunction was passed to the filter, uses noise_fn_ to add noise to the posterior.
  */
  void Predict(const Eigen::VectorXf &control_u);

  /*
  * @brief Update state space and weight each particle given a new measurement.
  * Computes a hypothesis measurement for each particle given measurement_model_
  * and assigns weights via weight_fn_ based on how well each proposal compares with the real measurement.
  */
  void Update(const Eigen::VectorXf &measurement_z);

  /*
  * @brief Samples a new state space from the current state space via resample_fn_
  * and updates the particles.
  */
  void Resample();

  /*
  * @brief computes mean state vector and covariance matrix given particles and weights.
  * Called after the predict and update steps.
  */
  void ComputeGaussianState();

  /*
  * @brief change the number of particles in the state space.
  * Deletes current state space and initializes new state space from the prior_fn_.
  * any data used in prior_fn_ should be updated first to reflect the current state.
  */
  void ChangeNumParticles(size_t new_num_particles);

  /*
  * @brief get pointer to state space
  */
  const std::vector<Particle>* GetPointerToParticles() const { return particles_; }


  /*
  * @brief get pointer to dynamics model
  */
  const PFDynamicsModel* GetPtrToDynamicsModel() const { return dynamics_model_; }

  /*
  * @brief get pointer to measurement model
  */
  const PFMeasurementModel* GetPtrToMeasurementModel() const { return measurement_model_; }

private:
  PFDynamicsModel *dynamics_model_;       // Pointer to particle filter dynamics model
  PFMeasurementModel *measurement_model_; // Pointer to particle filter hypothesis model
  std::vector<Particle> *particles_;      // Pointer to particles vector
  size_t num_particles_;                  // Number of particles in the state space
  PriorFunction *prior_fn_;               // prior function to sample init state space from
  NoiseFunction *noise_fn_;               // optional noise adder for particles post-prediction
  WeightFunction *weight_fn_;             // Weighting function to compare hypothesis and measurement
  Resampler *resample_fn_;                // Resampler for state space given weighted particles
};

#endif // INCLUDE_FILTERS_PARTICLE_FILTER_H_
