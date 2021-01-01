#ifndef INCLUDE_MODELS_PF_NOISE_H_
#define INCLUDE_MODELS_PF_NOISE_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

/*
* @brief abstract noise function which adds noise to a state vector
*/
class NoiseFunction {
public:
  virtual void AddNoise(Eigen::VectorXf* state) = 0;
};

/*
* @brief Adds gaussian noise to a state vector given sigmas_vector_, a vector of standard deviations
*/
class GaussianNoise : public NoiseFunction {
public:
  GaussianNoise(const Eigen::VectorXf& sigmas_vector) : sigmas_vector_(sigmas_vector) {}
  void AddNoise(Eigen::VectorXf* state);

private:
  const Eigen::VectorXf& sigmas_vector_;  // Vector of standard deviations, or
};


#endif // INCLUDE_MODELS_PF_NOISE_H_
