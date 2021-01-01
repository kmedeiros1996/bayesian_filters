#ifndef INCLUDE_MODELS_PF_PRIOR_H_
#define INCLUDE_MODELS_PF_PRIOR_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// std
#include <vector>

/// Third Party
#include <Eigen/Core>

// Bayesian
#include "../util/types.h"

/*
* @brief abstract prior function which initializes prior particle distribution
*/
class PriorFunction {
public:
  virtual void SamplePrior(std::vector<Particle>* prior, const int num_samples) = 0;
};

/*
* @brief prior function which samples particles from a multivariate gaussian distribution given mu and sigma.
*/
class GaussianPrior : public PriorFunction {
public:
  GaussianPrior(const Eigen::VectorXf& mu, const Eigen::MatrixXf& sigma) : mu_(mu), sigma_(sigma) {}
  void SamplePrior(std::vector<Particle>* prior, const int num_samples);

private:
  Eigen::VectorXf mu_;
  Eigen::VectorXf sigma_;
};

#endif // INCLUDE_MODELS_PF_PRIOR_H_
