#ifndef INCLUDE_MODELS_PF_MODELS_H_
#define INCLUDE_MODELS_PF_MODELS_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

// Bayesian
#include "base_models.h"
#include "../util/types.h"
#include "../util/gaussian_utils.h"

// Base abstract particle filter models to inherit from

class PFDynamicsModel : public DynamicsModel {
public:
  virtual Eigen::VectorXf Predict(const Eigen::VectorXf& state, const Eigen::VectorXf& control) = 0;
};

class PFMeasurementModel : public MeasurementModel {
public:
  virtual Eigen::VectorXf GenerateHypothesis(const Eigen::VectorXf& state) = 0;
};


// Example implementations

/*
* @brief additive dynamics model which returns the state vector + the control vector.
* Assumes state dim == ctrl dim
*/

class AdditiveDynamicsModel : public PFDynamicsModel {
  Eigen::VectorXf Predict(const Eigen::VectorXf& state, const Eigen::VectorXf& control);

};

/*
* @brief passthrough model which simply returns the state.
*/
class PassthroughModel : public PFMeasurementModel {
  Eigen::VectorXf GenerateHypothesis(const Eigen::VectorXf& state) { return state; }
};

#endif // INCLUDE_MODELS_PF_MODELS_H_
