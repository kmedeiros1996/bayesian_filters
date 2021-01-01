
include(${PROJECT_SOURCE_DIR}/cmake/path_defs.cmake)

# Set HEADERS_PUBLIC variable (public headers, included in the library)
set(HEADERS_PUBLIC
  ${ROOT_MODELS}/base_models.h
  ${ROOT_MODELS}/kf_models.h
  ${ROOT_MODELS}/pf_models.h
  ${ROOT_MODELS}/pf_prior.h
  ${ROOT_MODELS}/pf_noise.h
  ${ROOT_MODELS}/pf_weight.h
  ${ROOT_MODELS}/pf_resample.h
  ${ROOT_FILTERS}/bayesian_filter.h
  ${ROOT_FILTERS}/kalman_filter.h
  ${ROOT_FILTERS}/particle_filter.h
  ${ROOT_UTIL}/types.h
  ${ROOT_UTIL}/gaussian_utils.h
  ${ROOT_UTIL}/matrix_utils.h
)

# Set SOURCES variable
set(SOURCES
  ${ROOT_MODELS}/kf_models.cc
  ${ROOT_MODELS}/pf_models.cc
  ${ROOT_MODELS}/pf_prior.cc
  ${ROOT_MODELS}/pf_noise.cc
  ${ROOT_MODELS}/pf_weight.cc
  ${ROOT_MODELS}/pf_resample.cc
  ${ROOT_FILTERS}/kalman_filter.cc
  ${ROOT_FILTERS}/particle_filter.cc
  ${ROOT_UTIL}/matrix_utils.cc
  ${ROOT_UTIL}/gaussian_utils.cc
)

include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)
