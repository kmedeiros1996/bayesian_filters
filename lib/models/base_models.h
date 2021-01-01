#ifndef INCLUDE_MODELS_BASE_MODELS_H_
#define INCLUDE_MODELS_BASE_MODELS_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

/// std
#include <cstdint>

/*
* Base class to provide an interface for dynamics models used during the Bayesian filter predict step.
*/
class DynamicsModel {
public:

  /*
  * @brief base constructor which initializes state and control dimensions.
  */
  DynamicsModel(uint8_t state_dims, uint8_t ctrl_dims) : state_dim_(state_dims), ctrl_dim_(ctrl_dims) {}

  // Dimension Getters
  uint8_t StateDim() const {return state_dim_; }
  uint8_t CtrlDim() const {return ctrl_dim_; }

protected:
  uint8_t state_dim_; // Dimensionality of state vector
  uint8_t ctrl_dim_;  // Dimensionality of controls vector
};

/*
* Base class to provide an interface for measurement models used during the Bayesian filter update step.
*/
class MeasurementModel {
public:

  /*
  * @brief base constructor which initializes measurement dimensions.
  */
  MeasurementModel(uint8_t state_dims, uint8_t meas_dims) : state_dim_(state_dims), meas_dim_(meas_dims){}

  // Dimension Getters
  uint8_t MeasDim() const {return meas_dim_; }

protected:
  uint8_t state_dim_; // Dimensionality of state vector
  uint8_t meas_dim_; // Dimensionality of measurements vector
};

#endif // INCLUDE_MODELS_BASE_MODELS_H_
