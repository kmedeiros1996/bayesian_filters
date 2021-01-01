/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// Bayesian
#include "pf_models.h"

Eigen::VectorXf AdditiveDynamicsModel::Predict(const Eigen::VectorXf& state, const Eigen::VectorXf& control) {
  assert (state.size() = control.size());
  return state + control;
}
