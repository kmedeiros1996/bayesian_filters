#ifndef INCLUDE_FILTERS_BAYESIAN_FILTER_H_
#define INCLUDE_FILTERS_BAYESIAN_FILTER_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// std
#include <cstdint>
#include <assert.h>

/// Third Party
#include <Eigen/Core>

/// Bayesian
#include "../models/base_models.h"
#include "../util/types.h"

/*
* The Bayesian filter is an abstract class which all filters in this library inherit from.
* Will always have a dynamics model and measurement model for doing the predict and update steps.
*/

class BayesianFilter {

public:

  /*
  * @brief constructor which initializes the filter with dimensions from dynamics / observation models.
  * @param dynamics_model pointer to base DynamicsModel class with dimension info
  * @param observation_model pointer to base MeasurementModel class with dimension info
  */
  BayesianFilter(DynamicsModel *dynamics_model, MeasurementModel *measurement_model) :
  state_dim_(dynamics_model->StateDim()), ctrl_dim_(dynamics_model->CtrlDim()), meas_dim_(measurement_model->MeasDim()) {}

  /*
  * @brief init method which sets initial state and covariance for the filter.
  * @param init_state initial filter state. size must match state_dim.
  * @param init_covariance initial filter covariance. rows and columns must match state_dim.
  */
  virtual void Init(const Eigen::VectorXf& init_state, const Eigen::MatrixXf& init_covariance) {
    assert(state_dim_ == init_state.size()); // Init state size must match state dims!
    assert(state_dim_ == init_covariance.rows() && state_dim_ == init_covariance.cols()); // Init covariance dims must match state dims!
    is_init_ = true;
    state_.mu = init_state;
    state_.sigma = init_covariance;
  }

  /*
  * @brief Abstract bayesian filter prediction step. Must be implemented by subclasses.
  */
  virtual void Predict(const Eigen::VectorXf &control_u) = 0;

  /*
  * @brief Abstract bayesian filter update step. Must be implemented by subclasses.
  */
  virtual void Update(const Eigen::VectorXf &measurement_z) = 0;

  /*
  * Getters for state and covariance.
  */
  Eigen::VectorXf GetState() const { return state_.mu; }
  Eigen::VectorXf GetCovariance() const { return state_.sigma; }

protected:
  uint8_t state_dim_;           // Dimensionality of state vector
  uint8_t ctrl_dim_;            // Dimensionality of controls vector
  uint8_t meas_dim_;            // Dimensionality of measurements vector
  bool is_init_{false};         // Flag indicating initialization
  Gaussian state_;              // Gaussian which stores state vector and covariance matrix
};

#endif // INCLUDE_FILTERS_BAYESIAN_FILTER_H_
