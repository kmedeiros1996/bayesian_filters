#ifndef INCLUDE_MODELS_KF_MODELS_H_
#define INCLUDE_MODELS_KF_MODELS_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

/// Bayesian
#include "base_models.h"
#include "../util/types.h"

class KalmanDynamicsModel : public DynamicsModel {
public:
  KalmanDynamicsModel(uint8_t state_dims,
                      uint8_t ctrl_dims,
                      const Eigen::MatrixXf& state_transition,
                      const Eigen::MatrixXf& control_transition,
                      const Eigen::MatrixXf& process_noise);

  /*
  * @brief computes the predicted state vector and covariance matrix based on system dynamics.
  * Given
  * - state at time t-1 mu[t-1]
  * - covariance matrix sigma at time t-1 sigma[t-1]
  * - Control command u
  * - state transition matrix F
  * - control matrix B
  *
  * The predicted state vector mu_hat and covariance matrix sigma_hat at time t is as follows:
  *
  * mu_hat[t] = F*mu[t-1] + B*u[t]
  * sigma_hat[t] = F*sigma[t-1] *F.transpose() + Q
  *
  * @return gaussian struct which contains the predicted state vector and covariance matrix
  */
  Gaussian Predict(const Gaussian& prev_state, const Eigen::VectorXf &control) const;


  // Setters and getters for state/control transition matrices and process noise
  Eigen::MatrixXf GetStateTransition() const { return state_transition_F_; }
  void SetStateTransition(const Eigen::MatrixXf& new_state_transition);

  Eigen::MatrixXf GetControlTransition() const { return control_transition_B_; }
  void SetControlTransition(const Eigen::MatrixXf& new_control_transition);

  Eigen::MatrixXf GetProcessNoise() const { return process_noise_Q_; }
  void SetProcessNoise(const Eigen::MatrixXf& new_proc_noise);

private:
  Eigen::MatrixXf state_transition_F_;      // (state_dim x state_dim) matrix which describes dynamics of the system
  Eigen::MatrixXf control_transition_B_;    // (state_dim x ctrl_dim) matrix which converts control command into an effect on the system
  Eigen::MatrixXf process_noise_Q_;         // (state_dim x state_dim) matrix which models the noise in the process

};

class KalmanMeasurementModel : public MeasurementModel {
public:
  KalmanMeasurementModel(uint8_t state_dims,
                         uint8_t meas_dims,
                         const Eigen::MatrixXf& measurement_matrix_,
                         const Eigen::MatrixXf& measurement_noise_matrix);

  /*
  * @brief Updates the state vector and covariance matrix given a predicted state and measurement.
  * Given
  * - predicted state at time t mu_hat[t]
  * - predicted covariance matrix sigma at time t sigma_hat[t]
  * - Measurement matrix H
  * - Measurement noise matrix R
  * - Measurement z
  *
  * The error/residual value Y is computed as
  * Y = z - H * mu_hat[t]
  *
  * The Kalman Gain K, which scales the residual by observation certainty, is computed as follows:
  * K = (sigma_hat[t] * H.transpose()) * (H * sigma_hat[t] *  H.transpose() + R)^-1
  *
  * The updated state vector mu and covariance matrix sigma at time t is as follows:
  * mu[t] = mu_hat[t] + K*Y
  * sigma[t] = (Identity() - K*H) * sigma_hat[t]
  *
  * @return gaussian struct which contains the updated state vector and covariance matrix
  */
  Gaussian Update(const Gaussian& pred_state, const Eigen::VectorXf& meas);

  /*
  * @brief getters for measurement mapping matrix and measurement noise matrix
  */
  Eigen::MatrixXf GetMeasurementMatrix() const { return measurement_matrix_H_; }
  void SetMeasurementMatrix(const Eigen::MatrixXf& new_meas_mat);
  Eigen::MatrixXf GetMeasurementNoiseMatrix() const { return measurement_noise_matrix_R_; }
  void SetMeasurementNoiseMatrix(const Eigen::MatrixXf& new_meas_noise_mat);
  Eigen::MatrixXf GetKalmanGain() const { return kalman_gain_K_; }

private:
  Eigen::MatrixXf measurement_matrix_H_;            // (meas_dim x state_dim) Matrix which describes how to map state vector to observation vector
  Eigen::MatrixXf measurement_noise_matrix_R_;      // (meas_dim x meas_dim) Matrix describing measurement noise
  Eigen::MatrixXf kalman_gain_K_;                   // Kalman gain, represents certainty of observations
};


#endif // INCLUDE_MODELS_KF_MODELS_H_
