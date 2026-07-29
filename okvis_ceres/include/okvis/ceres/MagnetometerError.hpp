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
 * @file ceres/MagnetometerError.hpp
 * @brief Synchronous magnetometer heading factor.
 *
 * Residual:  e = C_MI * C_WS^T * b_ref_W - b_measured_M
 *
 * The measurement b_measured_M is the raw 3D field vector in the magnetometer
 * frame [T].  Switching to a fused-attitude source (e.g. PX4 vehicle_attitude)
 * requires only a change in the subscriber: synthesise
 *   b_measured_M = C_MI * C_WS_att^T * b_ref_W
 * and pass it through the same interface — the cost function is unchanged.
 */

#ifndef INCLUDE_OKVIS_CERES_MAGNETOMETERERROR_HPP_
#define INCLUDE_OKVIS_CERES_MAGNETOMETERERROR_HPP_

#include <ceres/sized_cost_function.h>
#include <Eigen/Core>

#include <okvis/Parameters.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
namespace ceres {

/// @brief Synchronous magnetometer factor.
/// Parameters: T_WS pose (7D).  Residuals: 3D field vector error.
class MagnetometerError
    : public ::ceres::SizedCostFunction<3 /* residuals */, 7 /* T_WS */>,
      public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef ::ceres::SizedCostFunction<3, 7> base_t;
  static const int kNumResiduals = 3;
  typedef Eigen::Matrix3d   covariance_t;
  typedef Eigen::Vector3d   measurement_t;
  typedef covariance_t      information_t;
  typedef Eigen::Matrix<double, 3, 7> jacobian0_t;

  MagnetometerError() = default;

  /// @param measurement   Measured field in magnetometer frame [T].
  /// @param information   Inverse-covariance (3×3) [T^-2].
  /// @param params        Magnetometer sensor parameters.
  MagnetometerError(const measurement_t& measurement,
                    const information_t& information,
                    const MagnetometerParameters& params);

  virtual ~MagnetometerError() = default;

  // --- setters ---
  virtual void setMeasurement(const measurement_t& m) { measurement_ = m; }
  virtual void setInformation(const information_t& info);
  virtual void setMagnetometerParameters(const MagnetometerParameters& p) { params_ = p; }

  // --- getters ---
  virtual const measurement_t& measurement() const { return measurement_; }
  virtual const information_t& information() const { return information_; }
  virtual const covariance_t&  covariance()  const { return covariance_; }

  // --- Ceres interface ---
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override;

  virtual bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const override;

  // --- ErrorInterface ---
  int residualDim()                  const override { return kNumResiduals; }
  int parameterBlocks()              const override { return parameter_block_sizes().size(); }
  int parameterBlockDim(int id)      const override { return base_t::parameter_block_sizes().at(id); }
  std::string typeInfo()             const override { return "MagnetometerError"; }

 protected:
  measurement_t measurement_;
  MagnetometerParameters params_;

  information_t information_;
  covariance_t  covariance_;
  mutable information_t squareRootInformation_;
};

}  // namespace ceres
}  // namespace okvis

#endif  // INCLUDE_OKVIS_CERES_MAGNETOMETERERROR_HPP_
