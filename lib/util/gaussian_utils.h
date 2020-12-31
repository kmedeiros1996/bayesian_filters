#ifndef INCLUDE_UTIL_GAUSSIAN_UTILS_H_
#define INCLUDE_UTIL_GAUSSIAN_UTILS_H_

/*
* Bayesian filters library.
* Author: Kyle M. Medeiros <kyle.medeiros@kylemail.net>
*/

/// Third Party
#include <Eigen/Core>

/// C++
#include <vector>

/// Bayesian
#include "types.h"

/*
* @brief Extract a random sample from a 1D gaussian distribution centered around mu w/ standard deviation sigma.
*/
float GaussianSample1D(float mu, float sigma);

/*
* @brief generates vector of floats sampled from a 1D gaussian distribution of mu and sigma
* @return Eigen type vector of length dims
*/
Eigen::VectorXf NormalVector1D (float mu, float sigma, uint32_t dims);

/*
* @brief generate a multivariate gaussian vector given a mean vector and vector of standard deviations.
*/
Eigen::VectorXf MultivariateNormalVector(const Eigen::VectorXf& mu, const Eigen::VectorXf& sigmas_vector);

/*
* @brief generate a matrix of multivariate gaussian vectors given a Gaussian struct.
*/
Eigen::MatrixXf MultivariateNormalMatrix(const Gaussian& gaussian, int num_samples);

/*
* @brief generate a matrix of multivariate gaussian vectors, given a mean vector and vector of standard deviations.
*/
Eigen::MatrixXf MultivariateNormalMatrix(const Eigen::VectorXf& mu, const Eigen::VectorXf& sigmas_vector, int num_samples);

/*
* @brief Add gaussian noise to a vector centered around zero, given a vector of standard deviations.
*/
void AddGaussianNoise(std::vector<Eigen::VectorXf> &x, const Eigen::VectorXf &sigmas_vector);

/*
* @brief returns the standard deviations vector via the square root of the diagonal of a covariance matrix.
*/
Eigen::VectorXf GetStdDeviationsVector(const Eigen::MatrixXf &sigma);


#endif // INCLUDE_UTIL_GAUSSIAN_UTILS_H_
