/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

// Bayesian
#include "matrix_utils.h"

//Third Party
#include <Eigen/Core>

//C++
#include <vector>

std::vector<float> EigenToStd(const Eigen::VectorXf& eigen) {
  std::vector<float> out(static_cast<size_t>(eigen.size()));
  for (int i = 0; i < out.size(); i++) {
    out[i] = eigen(i);
  }
  return out;
}

Eigen::VectorXf StdToEigen(const std::vector<float>& std_vec) {
  Eigen::VectorXf out (static_cast<int>(std_vec.size()));
  for (int i = 0; i < out.size(); i++) {
    out[i] = std_vec[i];
  }
  return out;
}
