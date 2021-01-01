/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// std
#include <assert.h>

/// Third Party
#include <Eigen/Core>

// Bayesian
#include "kf_models.h"

KalmanDynamicsModel::KalmanDynamicsModel(uint8_t state_dims,
                                         uint8_t ctrl_dims,
                                         const Eigen::MatrixXf& state_transition,
                                         const Eigen::MatrixXf& control_transition,
                                         const Eigen::MatrixXf& process_noise) :
                                         DynamicsModel(state_dims, ctrl_dims),
                                         state_transition_F_ (state_transition),
                                         control_transition_B_ (control_transition),
                                         process_noise_Q_ (process_noise) {
    assert(state_transition_F_.rows() == state_dim_);
    assert(state_transition_F_.cols() == state_dim_);
    assert(control_transition_B_.rows() == state_dim_);
    assert(control_transition_B_.cols() == ctrl_dim_);
    assert(process_noise_Q_.rows() == state_dim_);
    assert(process_noise_Q_.cols() == state_dim_);
}

void KalmanDynamicsModel::SetStateTransition(const Eigen::MatrixXf& new_state_transition) {
  assert(new_state_transition.rows() == state_dim_);
  assert(new_state_transition.cols() == state_dim_);
  state_transition_F_ = new_state_transition;
}

void KalmanDynamicsModel::SetControlTransition(const Eigen::MatrixXf& new_control_transition) {
  assert(new_control_transition.rows() == state_dim_);
  assert(new_control_transition.cols() == ctrl_dim_);
  control_transition_B_ = new_control_transition;
}

void KalmanDynamicsModel::SetProcessNoise(const Eigen::MatrixXf& new_proc_noise) {
  assert(new_proc_noise.rows() == state_dim_);
  assert(new_proc_noise.cols() == state_dim_);
  process_noise_Q_ = new_proc_noise;
}

Gaussian KalmanDynamicsModel::Predict(const Gaussian& prev_state, const Eigen::VectorXf& control) const {
  assert(prev_state.mu.size() = state_dim_);
  assert(prev_state.sigma.rows() = state_dim_);
  assert(prev_state.sigma.cols() = state_dim_);
  assert(control.size() = ctrl_dim_);

  Gaussian est_state;

  est_state.mu = state_transition_F_ * prev_state.mu + control_transition_B_ * control;
  est_state.sigma = state_transition_F_ * prev_state.sigma * state_transition_F_.transpose() + process_noise_Q_;

  return est_state;
}


KalmanMeasurementModel::KalmanMeasurementModel(uint8_t state_dims,
                                               uint8_t meas_dims,
                                               const Eigen::MatrixXf& measurement_matrix_,
                                               const Eigen::MatrixXf& measurement_noise_matrix) :
                                               MeasurementModel(state_dims, meas_dims),
                                               measurement_matrix_H_ (measurement_matrix_),
                                               measurement_noise_matrix_R_ (measurement_noise_matrix),
                                               kalman_gain_K_ (Eigen::MatrixXf::Zero(state_dim_, meas_dim_)) {
    assert(measurement_matrix_H_.rows() == meas_dim_);
    assert(measurement_matrix_H_.cols() == state_dim_);
    assert(measurement_noise_matrix_R_.rows() == meas_dim_);
    assert(measurement_noise_matrix_R_.cols() == meas_dim_);
}

void KalmanMeasurementModel::SetMeasurementMatrix(const Eigen::MatrixXf& new_meas_mat) {
  assert(new_meas_mat.rows() == meas_dim_);
  assert(new_meas_mat.cols() == state_dim_);
  measurement_matrix_H_ = new_meas_mat;
}

void KalmanMeasurementModel::SetMeasurementNoiseMatrix(const Eigen::MatrixXf& new_meas_noise_mat) {
  assert(new_meas_mat.rows() == meas_dim_);
  assert(new_meas_mat.cols() == meas_dim_);
  measurement_noise_matrix_R_ = new_meas_noise_mat;
}

Gaussian KalmanMeasurementModel::Update(const Gaussian& pred_state, const Eigen::VectorXf& meas) {
  assert(pred_state.mu.size() = state_dim_);
  assert(pred_state.sigma.rows() = state_dim_);
  assert(pred_state.sigma.cols() = state_dim_);
  assert(meas.size() = ctrl_dim_);

  Eigen::MatrixXf k_lhs = pred_state.sigma * measurement_matrix_H_.transpose();
  Eigen::MatrixXf k_rhs =
  (measurement_matrix_H_ * pred_state.sigma * measurement_matrix_H_.transpose())
  + measurement_noise_matrix_R_;

  kalman_gain_K_ = k_lhs * k_rhs;


  Eigen::VectorXf residual = meas - measurement_matrix_H_ * pred_state.mu;

  Eigen::MatrixXf identity = Eigen::MatrixXf::Identity(state_dim_, state_dim_);
  Eigen::MatrixXf sigma_lhs = identity - kalman_gain_K_ * measurement_matrix_H_;

  Gaussian new_state;
  new_state.mu = pred_state.mu + kalman_gain_K_*residual;
  new_state.sigma = sigma_lhs * pred_state.sigma;
  return new_state;
}
