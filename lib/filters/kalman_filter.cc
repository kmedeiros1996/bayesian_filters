/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

#include "kalman_filter.h"

void KalmanFilter::Predict(const Eigen::VectorXf &control_u) {
  assert(is_init);
  state_ = dynamics_model_->Predict(state_, control_u);
}

void KalmanFilter::Update(const Eigen::VectorXf &measurement_z) {
  assert(is_init);
  state_ = measurement_model_->Update(state_, measurement_z);
}
