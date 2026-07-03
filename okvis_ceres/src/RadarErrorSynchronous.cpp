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
 * @file RadarErrorSynchronous.cpp
 * @brief Source File for the RadarErrorSynchronous class
 * @author Daniel Spinn 
 */

#include <eigen3/Eigen/Core>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/RadarErrorSynchronous.hpp>
#include <okvis/Parameters.hpp>
//#include <gtest/gtest.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// // Default constructor - not in GpsErrorSynchronous.cpp, so commented out
// RadarErrorSynchronous::RadarErrorSynchronous() {
//   radarId_ = 0;
//   measurement_.setZero();
//   information_.setIdentity();
//   covariance_.setIdentity();
//   squareRootInformation_.setIdentity();
//   omega_S_.setZero();
// }

// Construct with measurement and information matrix.
RadarErrorSynchronous::RadarErrorSynchronous(
    uint64_t radarId, const measurement_t & measurement, const covariance_t & information, 
    const RadarParameters & radarParameters, const Eigen::Vector3d & omega_S) {
  setRadarId(radarId);
  setMeasurement(measurement);
  setInformation(information);
  setRadarParameters(radarParameters);
  setOmegaS(omega_S);
}

// Set the information.
void RadarErrorSynchronous::setInformation(
    const information_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<Eigen::Matrix3d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool RadarErrorSynchronous::Evaluate(double const* const * parameters,
                                             double* residuals,
                                             double** jacobians) const {

  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);  
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool RadarErrorSynchronous::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {


  // robot pose: world to sensor
  Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
  const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();
  Eigen::Matrix3d C_SW = C_WS.transpose(); // R_IW (rotation from World to IMU)

  // SpeedAndBias parameter block
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> speedAndBias(parameters[1]);
  Eigen::Vector3d v_W = speedAndBias.head<3>(); // v_I_W (velocity in world frame)
  Eigen::Vector3d b_g = speedAndBias.segment<3>(3); // b_g_I gyro bias

  // Transformation from IMU to Radar frame
  okvis::kinematics::Transformation T_IR = radarParameters_.T_IR;
  Eigen::Matrix3d C_IR = T_IR.C(); // R_IR (rotation from Radar to IMU)
  Eigen::Matrix3d C_RI = C_IR.transpose(); // R_RI (rotation from IMU to Radar)
  Eigen::Vector3d p_IR_I = T_IR.r(); // p_IR_I (position of radar in IMU frame)

  // Angular velocity in IMU frame (measurement - bias)
  Eigen::Vector3d omega_S_corrected = omega_S_ - b_g; // ω_I - b_g_I

  // Calculate the radar error: e_R = R_RI (R_IW * v_I_W + (ω_I - b_g_I) × p_IR_I) - ṽ_r_R
  Eigen::Vector3d v_I = C_SW * v_W; // R_IW * v_I_W (velocity in IMU frame)
  Eigen::Vector3d omega_cross_p = omega_S_corrected.cross(p_IR_I); // (ω_I - b_g_I) × p_IR_I
  Eigen::Vector3d v_R_expected = C_RI * (v_I + omega_cross_p); // R_RI * (R_IW * v_I_W + (ω_I - b_g) × I p_R)
  
  measurement_t error = v_R_expected - measurement_; // v_R_expected - v_R_measured

  // weight error by square root of information matrix:
  measurement_t weighted_error = squareRootInformation_ * error;

  // assign:
  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];
  residuals[2] = weighted_error[2];


  // calculate jacobians, if required
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {

        // Jacobians w.r.t robot pose
        Eigen::Matrix<double,3,6> J0_minimal;
        J0_minimal.setZero();
        
        // Jacobian w.r.t. translation: zero (velocity measurement doesn't depend on position)
        // J0_minimal.topLeftCorner<3,3>() is already zero
        
        // Jacobian w.r.t. rotation: C_RI * crossMx(v_I)
        // The rotation affects v_I = C_SW * v_W, but not the cross product term
        J0_minimal.topRightCorner<3,3>() = C_RI * C_SW * okvis::kinematics::crossMx(v_W);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J0_lift;
        PoseManifold::minusJacobian(parameters[0], J0_lift.data());

        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J0(jacobians[0]);
        J0 = squareRootInformation_ * J0_minimal * J0_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
            J0_minimal_mapped = squareRootInformation_ * J0_minimal;  // this is for Euclidean-style perturbation only.
          }
        }

    }
    if (jacobians[1] != NULL) {

        // Jacobians w.r.t SpeedAndBias
        Eigen::Matrix<double,3,9> J1_minimal;
        J1_minimal.setZero();
        
        // Jacobian w.r.t. velocity: R_RI * R_IW
        J1_minimal.topLeftCorner<3,3>() = C_RI * C_SW;
        
        // Jacobian w.r.t. gyro bias: R_RI * crossMx(p_IR)
        J1_minimal.block<3,3>(0,3) = C_RI * okvis::kinematics::crossMx(p_IR_I);
        
        // Jacobian w.r.t. accel bias: zero (not used in radar error)

        // SpeedAndBias uses Euclidean parameterization, so lift is identity
        Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > J1(jacobians[1]);
        J1 = squareRootInformation_ * J1_minimal;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
            J1_minimal_mapped = squareRootInformation_ * J1_minimal;
          }
        }
    }
  }

  return true;
}

bool RadarErrorSynchronous::VerifyJacobianNumDiff(double const* const * parameters,
                                     double** jacobian) const{

  bool success = true;
  const double threshold = 1e-6;
  // Only execute when Jacobians are provided
  if(jacobian != NULL){

      // T_WS
      Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
      const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
      okvis::kinematics::Transformation T_WS(t_WS_W, q_WS);
      Eigen::Matrix3d C_WS = T_WS.C();
      Eigen::Matrix3d C_SW = C_WS.transpose();

      // SpeedAndBias
      Eigen::Map<const Eigen::Matrix<double, 9, 1>> speedAndBias(parameters[1]);
      Eigen::Vector3d v_W = speedAndBias.head<3>();
      Eigen::Vector3d b_g = speedAndBias.segment<3>(3);

      // T_IR
      okvis::kinematics::Transformation T_IR = radarParameters_.T_IR;
      Eigen::Matrix3d C_IR = T_IR.C();
      Eigen::Matrix3d C_RI = C_IR.transpose();
      Eigen::Vector3d p_IR_I = T_IR.r();

      // Angular velocity
      Eigen::Vector3d omega_S_corrected = omega_S_ - b_g;

      // x, dx : in minimal representation
      double dx = 1e-7;     // Perturbation size
      Eigen::Matrix<double, 6, 1> delta;     // Perturbation vector for robot pose
      Eigen::Matrix<double, 9, 1> delta_sb;     // Perturbation vector for SpeedAndBias
      Eigen::Matrix<double, 6, 1> xp; // x + dx
      Eigen::Matrix<double, 6, 1> xm; // x - dx
      Eigen::Matrix<double, 9, 1> sbp; // speedAndBias + dx
      Eigen::Matrix<double, 9, 1> sbm; // speedAndBias - dx
      // e(x+dx), e(x-dx)
      Eigen::Matrix<double,3,1> ep; // e(x+dx)
      Eigen::Matrix<double,3,1> em; // e(x-dx)

      // Helper function to compute error
      auto computeError = [&](const okvis::kinematics::Transformation& T_WS_val, 
                              const Eigen::Vector3d& v_W_val, 
                              const Eigen::Vector3d& b_g_val) -> Eigen::Vector3d {
        Eigen::Matrix3d C_WS_val = T_WS_val.C();
        Eigen::Matrix3d C_SW_val = C_WS_val.transpose();
        Eigen::Vector3d v_I_val = C_SW_val * v_W_val;
        Eigen::Vector3d omega_S_corrected_val = omega_S_ - b_g_val;
        Eigen::Vector3d omega_cross_p_val = omega_S_corrected_val.cross(p_IR_I);
        Eigen::Vector3d v_R_expected_val = C_RI * (v_I_val + omega_cross_p_val);
        return squareRootInformation_ * (v_R_expected_val - measurement_);
      };

      // if jacobians for robot pose are provided
      if(jacobian[0]!=NULL){

          // Jacobian w.r.t robot pose
          Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > Jr(jacobian[0]);
          Eigen::Matrix<double,3,6> Jrn_minimal;
          Jrn_minimal.setZero();

          // Numerical Jacobian w.r.t robot pose
          for (size_t i = 0; i < 6; ++i) {

            delta.setZero();

            // x+dx
            delta[i] = dx;
            okvis::kinematics::Transformation Tp = T_WS;
            Tp.oplus(delta);
            ep = computeError(Tp, v_W, b_g);

            // x-dx
            delta[i] = -dx;
            okvis::kinematics::Transformation Tm = T_WS;
            Tm.oplus(delta);
            em = computeError(Tm, v_W, b_g);

            // difference quotient (e(x+dx) - e(x-dx)) / (2*dx)
            Jrn_minimal.col(i) = (ep - em) / (2 * dx);
          }

          // liftJacobian to switch to non-minimal representation
          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> Jrn_lift;
          PoseManifold::minusJacobian(parameters[0], Jrn_lift.data());

          // hallucinate Jacobian w.r.t. state
          Eigen::Matrix<double, 3, 7> Jrn;
          Jrn = Jrn_minimal * Jrn_lift;

          // Test if analytically and numerically computed Jacobians are close enough
          //EXPECT_TRUE((Jr - Jrn).norm() < 1e-6);

          // Actual check
          double norm_diff = (Jr - Jrn).norm();
          if (norm_diff >= threshold) {
            success = false;
            std::cout << "RadarErrorSynchronous: Jacobian w.r.t. robot pose differs! Norm of difference: " << norm_diff << std::endl;
            // Temporary debug print
            std::cout << "--- ANALYTICAL Jr (3x7) ---\n" << Jr << std::endl;
            std::cout << "--- NUMERICAL Jrn (3x7) ---\n" << Jrn << std::endl;
            std::cout << "T_WS for debugging:\n" << T_WS.T() << std::endl;
            std::cout << "v_I: " << (C_SW * v_W).transpose() << std::endl;
            std::cout << "C_RI:\n" << C_RI << std::endl;
          }

      }

      // if jacobians for SpeedAndBias are provided
      if(jacobian[1]!=NULL){

          Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > Jsb(jacobian[1]);
          Eigen::Matrix<double,3,9> Jsbn_minimal;
          Jsbn_minimal.setZero();

          // Numerical Jacobian w.r.t SpeedAndBias
          for (size_t i = 0; i < 9; ++i) {

            delta_sb.setZero();

            // x+dx
            delta_sb[i] = dx;
            Eigen::Vector3d v_W_p = v_W;
            Eigen::Vector3d b_g_p = b_g;
            Eigen::Vector3d b_a_p = speedAndBias.tail<3>();
            if (i < 3) {
              v_W_p[i] += dx;
            } else if (i < 6) {
              b_g_p[i-3] += dx;
            } else {
              b_a_p[i-6] += dx;
            }
            ep = computeError(T_WS, v_W_p, b_g_p);

            // x-dx
            delta_sb[i] = -dx;
            Eigen::Vector3d v_W_m = v_W;
            Eigen::Vector3d b_g_m = b_g;
            Eigen::Vector3d b_a_m = speedAndBias.tail<3>();
            if (i < 3) {
              v_W_m[i] -= dx;
            } else if (i < 6) {
              b_g_m[i-3] -= dx;
            } else {
              b_a_m[i-6] -= dx;
            }
            em = computeError(T_WS, v_W_m, b_g_m);

            // difference quotient (e(x+dx) - e(x-dx)) / (2*dx)
            Jsbn_minimal.col(i) = (ep - em) / (2 * dx);
          }

          // SpeedAndBias uses Euclidean parameterization, so lift is identity
          Eigen::Matrix<double, 3, 9> Jsbn;
          Jsbn = Jsbn_minimal;

          // Test if analytically and numerically computed Jacobians are close enough
          //EXPECT_TRUE((Jsb - Jsbn).norm() < 1e-6);

          // Actual check
          double norm_diff = (Jsb - Jsbn).norm();
          if (norm_diff >= threshold) {
            success = false;
            std::cout << "RadarErrorSynchronous: Jacobian w.r.t. SpeedAndBias differs! Norm of difference: " << norm_diff << std::endl;
            // Check specifically for bias sign error if you suspect it:
            // LOG(INFO) << "Analytical J_bias:\n" << Jsb.block<3,3>(0,3);
            // LOG(INFO) << "Numerical J_bias:\n" << Jsbn.block<3,3>(0,3);
          }

      }
      return success;

  }

  else
      return true;

} /* verifyJacobianNumDiff */

} /* namespace ceres */
} /* namespace okvis */

