/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// Bayesian
#include "particle_filter.h"
#include "../models/pf_models.h"
#include "../models/pf_noise.h"
#include "../models/pf_prior.h"
#include "../models/pf_resample.h"
#include "../models/pf_weight.h"

ParticleFilter::ParticleFilter(PFDynamicsModel *dynamics_model,
                               PFMeasurementModel *measurement_model,
                               size_t num_particles,
                               PriorFunction *prior_fn,
                               NoiseFunction *noise_fn,
                               WeightFunction *weight_fn,
                               Resampler *resample_fn_) :
                               BayesianFilter(dynamics_model, measurement_model),
                               dynamics_model_(dynamics_model),
                               measurement_model_(measurement_model),
                               num_particles_(num_particles),
                               prior_fn_(prior_fn),
                               noise_fn_(noise_fn),
                               weight_fn_(weight_fn),
                               resample_fn_(resample_fn_){
  particles_ = new std::vector<Particle>(num_particles_);
}

ParticleFilter::~ParticleFilter() {
  delete particles_;
}

void ParticleFilter::Init(const Eigen::VectorXf& init_state, const Eigen::MatrixXf& init_covariance) {
  BayesianFilter::Init(init_state, init_covariance);
  prior_fn_->SamplePrior(particles_, num_particles_);
}

void ParticleFilter::Predict(const Eigen::VectorXf &control_u) {
  assert(is_init_); // The particle filter must be initialized first!
  assert(control_u.size() == ctrl_dim_); // Controls input size must match control dim!

  for (Particle &particle : *particles_) {
    Eigen::VectorXf new_state = dynamics_model_->Predict(particle.state, control_u);
    if (noise_fn_ != nullptr) {
      noise_fn_->AddNoise(&new_state);
    }
    particle.state = new_state;
  }
  ComputeGaussianState();
}

void ParticleFilter::Update(const Eigen::VectorXf &measurement_z) {
  assert(is_init_); // The particle filter must be initialized first!
  assert(measurement_z.size() == meas_dim_); // Measurement input size must match measurement dim!

  double weight_sum = 0.0f;
  for (Particle &particle : *particles_) {
    Eigen::VectorXf hypothesis = measurement_model_->GenerateHypothesis(particle.state);
    particle.weight = weight_fn_->ComputeWeight(hypothesis, measurement_z);
    weight_sum+=particle.weight;
  }

  // Normalize the particle weights
  for (Particle &particle : *particles_) {
    particle.weight/=weight_sum;
  }
  ComputeGaussianState();
}

void ParticleFilter::Resample() {
  std::vector <Particle> *new_particles = new std::vector<Particle>(num_particles_);
  resample_fn_->Resample(particles_, new_particles);
  delete particles_;
  particles_ = new_particles;
}

void ParticleFilter::ComputeGaussianState() {
  state_.mu.setZero();
  state_.sigma.setZero();

  for (size_t i = 0; i < num_particles_; i++) {
      state_.mu+=(particles_->at(i).weight * particles_->at(i).state);
  }

  for (uint8_t j = 0; j < state_.mu.size(); j++) {
      for (uint8_t k = 0; k < state_.mu.size(); k++) {
          float cov {0.0};
          float weight_sum_sq {0.0};

          for (size_t i = 0; i < num_particles_; i++) {
              weight_sum_sq += particles_->at(i).weight * particles_->at(i).weight;
              cov+= particles_->at(i).weight * (particles_->at(i).state[j] - state_.mu[j])*(particles_->at(i).state[k] - state_.mu[k]);
          }
          cov/=(1.0 - weight_sum_sq);
          state_.sigma(j, k) = cov;
          if (j != k) {
              state_.sigma(k, j) = cov;
          }
      }
  }
}

void ParticleFilter::ChangeNumParticles(size_t new_num_particles) {
  delete particles_;
  num_particles_ = new_num_particles;
  particles_ = new std::vector<Particle>(num_particles_);
  prior_fn_->SamplePrior(particles_, num_particles_);
}
