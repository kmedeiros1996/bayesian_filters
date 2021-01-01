#include "pf_weight.h"

double EuclideanWeight::ComputeWeight(const Eigen::VectorXf& hypothesis, const Eigen::VectorXf& measurement) {
  assert(hypothesis.rows() == measurement.rows());
  Eigen::VectorXf delta = hypothesis - measurement;
  return delta.norm();
}

double MahalanobisWeight::ComputeWeight(const Eigen::VectorXf& hypothesis, const Eigen::VectorXf& measurement) {
  assert(hypothesis.rows() == measurement.rows());
  assert(measurement.rows() == inv_covariance_.rows());
  assert(measurement.rows() == inv_covariance_.cols());

  Eigen::VectorXf delta = hypothesis - measurement;
  Eigen::VectorXf lhs = delta.adjoint() * inv_covariance_;
  double mahal = lhs.dot(delta);
  return sqrt(mahal);
}
