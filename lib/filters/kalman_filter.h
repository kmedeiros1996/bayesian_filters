#ifndef INCLUDE_FILTERS_KALMAN_FILTER_H_
#define INCLUDE_FILTERS_KALMAN_FILTER_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// Bayesian
#include "bayesian_filter.h"
#include "../models/kf_models.h"

/*
* The Kalman Filter is a recursive state estimator which makes two key assumptions:
* 1: The distribution is gaussian
* 2: The dynamics / measurement models are linear
*
*/
class KalmanFilter : public BayesianFilter {
public:
  /*
  * @brief constructor for the kalman filter. Initializes dynamics / measurement model
  * and sets dimensions in the base bayesian filter class.
  */
  KalmanFilter(KalmanDynamicsModel* dynamics_model,
               KalmanMeasurementModel* measurement_model) :
               BayesianFilter(dynamics_model, measurement_model),
               dynamics_model_(dynamics_model),
               measurement_model_(measurement_model) {}


  /*
  * @brief Calls KalmanDynamicsModel::Predict which implements the kalman filter prediction equations.
  */
  void Predict(const Eigen::VectorXf &control_u);

  /*
  * @brief Calls KalmanDynamicsModel::Update which implements the kalman filter update equations.
  */
  void Update(const Eigen::VectorXf &measurement_z);

  /*
  * @brief get pointer to dynamics model
  */
  const KalmanDynamicsModel* GetPtrToDynamicsModel() const { return dynamics_model_; }

  /*
  * @brief get pointer to measurement model
  */
  const KalmanMeasurementModel* GetPtrToMeasurementModel() const { return measurement_model_; }

private:
  KalmanDynamicsModel *dynamics_model_;           // Dynamics model of the system
  KalmanMeasurementModel *measurement_model_;     // Measurement model of the system
};

#endif // INCLUDE_FILTERS_KALMAN_FILTER_H_
