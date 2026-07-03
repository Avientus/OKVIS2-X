/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file ceres/RadarErrorSynchronous.hpp
 * @brief Header file for the RadarErrorSynchronous class.
 * @author Daniel Spinn 
 */

#ifndef INCLUDE_OKVIS_CERES_RADARERRORSYNCHRONOUS_HPP_
#define INCLUDE_OKVIS_CERES_RADARERRORSYNCHRONOUS_HPP_

#include <vector>
#include <memory>
#include <ceres/sized_cost_function.h>
#include <ceres/covariance.h>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/Parameters.hpp>

namespace okvis {
namespace ceres {


/// \brief Implements a nonlinear radar velocity factor.
class RadarErrorSynchronous :
    public ::ceres::SizedCostFunction<3 /* number of residuals */,
        7 /* size of first parameter (RobotPoseParameterBlock T_WS) */,
        9 /* size of second parameter (SpeedAndBiasParameterBlock) */>,
    public ErrorInterface {

 public:

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<3, 7, 9> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 3;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 3, 3> covariance_t;

  /// \brief The measurement type.
  typedef Eigen::Vector3d measurement_t;

  /// \brief The type of the information (inverse of covariance, same dimension).
  typedef covariance_t information_t;

  /// \brief The type of Jacobian w.r.t. robot pose in world frame
  /// \warning This is w.r.t. minimal tangential space coordinates (lifted from 3x6 to 3x7)
  typedef Eigen::Matrix<double, 3, 7> jacobian0_t;

  /// \brief The type of the Jacobian w.r.t. SpeedAndBias
/// \note SpeedAndBias uses Euclidean parameterization (already minimal)
  typedef Eigen::Matrix<double, 3, 9> jacobian1_t;


  //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Default constructor.
  RadarErrorSynchronous();

  /// \brief Construct with measurement and information matrix
  /// @param[in] radarId The id of the radar.
  /// @param[in] measurement The velocity measurement in radar frame.
  /// @param[in] information The information (weight) matrix.
  /// @param[in] radarParameters The radar parameters struct
  /// @param[in] omega_S The angular velocity measurement in IMU frame
  RadarErrorSynchronous(uint64_t radarId, const measurement_t & measurement,
                    const covariance_t & information, const RadarParameters & radarParameters,
                    const Eigen::Vector3d & omega_S);

  /// \brief Trivial destructor.
  virtual ~RadarErrorSynchronous()
  {
  }

  // setters
  /// \brief Set radar parameters
  /// @param[in] radarParameters the parameters of radar sensor
  virtual void setRadarParameters(const RadarParameters & radarParameters){
      radarParameters_ = radarParameters;
  }

  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  virtual void setInformation(const information_t& information);

  /// \brief Set radar ID.
  /// @param[in] radarId ID of the radar.
  virtual void setRadarId(uint64_t radarId) {
    radarId_ = radarId;
  }

  /// \brief Set angular velocity measurement.
  /// @param[in] omega_S Angular velocity in IMU frame.
  virtual void setOmegaS(const Eigen::Vector3d & omega_S) {
    omega_S_ = omega_S;
  }

  // getters
  /// \brief Radar ID.
  uint64_t radarId() const {
    return radarId_;
  }

  /// \brief Get the measurement.
  /// \return The measurement vector.
  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  virtual const covariance_t& information() const
  {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  virtual const covariance_t& covariance() const
  {
    return covariance_;
  }

  /// \brief Get the radar parameters.
  virtual const okvis::RadarParameters& radarParameters() const {
    return radarParameters_;
  }

  /// \brief Get the angular velocity measurement.
  virtual const Eigen::Vector3d& omegaS() const {
    return omega_S_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of the evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // added convenient check
  virtual bool VerifyJacobianNumDiff(double const* const * parameters, double** jacobian) const;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  int parameterBlockDim(int parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "RadarErrorSynchronous";
  }

 protected:

  // the measurement
  measurement_t measurement_; ///< The (3D) velocity measurement in radar frame.

  // Radar parameters
  RadarParameters radarParameters_; ///< The radar parameters

  // Angular velocity measurement
  Eigen::Vector3d omega_S_; ///< Angular velocity measurement in IMU frame

  // weighting related
  covariance_t information_; ///< The 3x3 information matrix.
  covariance_t squareRootInformation_; ///< The 3x3 square root information matrix.
  covariance_t covariance_; ///< The 3x3 covariance matrix.

  //radar id
  uint64_t radarId_; ///< ID of the radar.

};

}  // namespace ceres
}  // namespace okvis

#endif // /* INCLUDE_OKVIS_CERES_RADARERRORSYNCHRONOUS_HPP_ */

