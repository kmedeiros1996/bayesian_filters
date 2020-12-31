#ifndef INCLUDE_UTIL_MATRIX_UTILS_H_
#define INCLUDE_UTIL_MATRIX_UTILS_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

//Third Party
#include <Eigen/Core>

//C++
#include <vector>

/*
* @brief convert a Eigen::VectorXf to a std::vector<float>
*/
std::vector<float> EigenToStd(const Eigen::VectorXf& eigen);

/*
* @brief convert a std::vector<float> to a Eigen::VectorXf
*/
Eigen::VectorXf StdToEigen(const std::vector<float>& std_vec);

#endif // INCLUDE_UTIL_MATRIX_UTILS_H_
