#ifndef INCLUDE_MODELS_PF_WEIGHT_H_
#define INCLUDE_MODELS_PF_WEIGHT_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

/*
* @brief abstract weighting function used to compare how well a proposal sample matches with a given measurement.
*/
class WeightFunction {
public:
  virtual double ComputeWeight(const Eigen::VectorXf& hypothesis, const Eigen::VectorXf& measurement) = 0;
};

/*
* @brief simple weighting function which returns the euclidean/L2 norm of two vectors.
*/
class EuclideanWeight : public WeightFunction {
  double ComputeWeight(const Eigen::VectorXf& hypothesis, const Eigen::VectorXf& measurement);
};

/*
* @brief weighting function which returns the mahalanobis distance between a vector
* and a distribution formed by a mean vector and covariance matrix.
*/
class MahalanobisWeight : public WeightFunction {
public:
  MahalanobisWeight(const Eigen::MatrixXf& covariance) : inv_covariance_(covariance.inverse()) {}
  void SetCovariance(const Eigen::MatrixXf& covariance) { inv_covariance_ = covariance.inverse(); }
  double ComputeWeight(const Eigen::VectorXf& hypothesis, const Eigen::VectorXf& measurement);
private:
  Eigen::MatrixXf inv_covariance_;
};



#endif // INCLUDE_MODELS_PF_WEIGHT_H_
