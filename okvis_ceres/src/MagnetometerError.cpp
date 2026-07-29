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

#include <okvis/ceres/MagnetometerError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
namespace ceres {

MagnetometerError::MagnetometerError(const measurement_t& measurement,
                                     const information_t& information,
                                     const MagnetometerParameters& params) {
  setMeasurement(measurement);
  setInformation(information);
  setMagnetometerParameters(params);
}

void MagnetometerError::setInformation(const information_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  Eigen::LLT<information_t> llt(information);
  squareRootInformation_ = llt.matrixL().transpose();
}

bool MagnetometerError::Evaluate(double const* const* parameters,
                                 double* residuals,
                                 double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool MagnetometerError::EvaluateWithMinimalJacobians(
    double const* const* parameters,
    double* residuals,
    double** jacobians,
    double** jacobiansMinimal) const {

  // Unpack T_WS from parameter block [r_x r_y r_z q_x q_y q_z q_w]
  Eigen::Map<const Eigen::Vector3d> r_WS(&parameters[0][0]);
  const Eigen::Quaterniond q_WS(
      parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  const Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();
  const Eigen::Matrix3d C_SW = C_WS.transpose();

  // Extrinsics: rotation from magnetometer frame to IMU (sensor) frame
  const Eigen::Matrix3d C_IM = params_.T_IM.C();
  const Eigen::Matrix3d C_MI = C_IM.transpose();

  // Predicted field in the magnetometer frame:
  //   b_pred_M = C_MI * C_WS^T * b_ref_W
  const Eigen::Vector3d b_in_S = C_SW * params_.b_ref_W;  // field in IMU frame
  const Eigen::Vector3d b_pred_M = C_MI * b_in_S;

  // Residual and weighting
  const Eigen::Vector3d error = b_pred_M - measurement_;
  const Eigen::Vector3d weighted_error = squareRootInformation_ * error;

  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];
  residuals[2] = weighted_error[2];

  if (jacobians == nullptr || jacobians[0] == nullptr) {
    return true;
  }

  // Jacobian of e w.r.t. the rotation part (minimal 3D right-perturbation δφ):
  //
  //   C_WS_pert = C_WS * Exp(δφ)  =>  C_SW_pert = Exp(-δφ) * C_SW
  //   b_pred_pert ≈ C_MI * (I - [δφ]×) * b_in_S
  //              = b_pred_M + C_MI * [b_in_S]× * δφ
  //
  //   ∂e/∂δφ = C_MI * crossMx(b_in_S)
  //
  // Minimal Jacobian [translation | rotation] = [0_{3×3}, C_MI * crossMx(b_in_S)]
  Eigen::Matrix<double, 3, 6> J_min;
  J_min.leftCols<3>().setZero();
  J_min.rightCols<3>() = C_MI * okvis::kinematics::crossMx(b_in_S);

  // Lift 6D → 7D via the manifold minus-Jacobian
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
  PoseManifold::minusJacobian(parameters[0], J_lift.data());

  Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
  J = squareRootInformation_ * J_min * J_lift;

  if (jacobiansMinimal != nullptr && jacobiansMinimal[0] != nullptr) {
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J_min_map(jacobiansMinimal[0]);
    J_min_map = squareRootInformation_ * J_min;
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis
