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
 * @file RadarErrorAsynchronous.cpp
 * @brief Source File for the RadarErrorAsynchronous class
 * @author Daniel Spinn
 */

#include <eigen3/Eigen/Core>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/RadarErrorAsynchronous.hpp>
#include <okvis/ceres/ImuError.hpp> // required for propagation
#include <okvis/ceres/ode/ode.hpp>
#include <okvis/Parameters.hpp>
//#include <gtest/gtest.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

std::atomic_bool RadarErrorAsynchronous::redoPropagationAlways(false);
std::atomic_bool RadarErrorAsynchronous::useImuCovariance(true);

// // Default constructor - not in GpsErrorAsynchronous.cpp, so commented out
// RadarErrorAsynchronous::RadarErrorAsynchronous() {
//   radarId_ = 0;
//   measurement_.setZero();
//   radarInformation_.setIdentity();
//   radarCovariance_.setIdentity();
//   squareRootInformation_.setIdentity();
//   omega_S_tr_.setZero();
// }

// Constructor with provided information matrix
RadarErrorAsynchronous::RadarErrorAsynchronous(uint64_t radarId, const measurement_t & measurement, const covariance_t & information,
                                           const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                                           const okvis::Time& tk, const okvis::Time& tr,
                                           const Eigen::Vector3d & omega_S_tr,
                                           const okvis::RadarParameters & radarParameters){
    setRadarId(radarId);
    setMeasurement(measurement);
    setInformation(information);
    setImuMeasurements(imuMeasurements);
    setImuParameters(imuParameters);
    setTk(tk);
    setTr(tr);
    setOmegaSTr(omega_S_tr);
    setRadarParameters(radarParameters);

    // derivatives of covariance matrix w.r.t. sigmas
    dPdsigma_.resize(4);
    for(size_t j=0; j<4; ++j) {
      dPdsigma_.at(j).setZero();
    }
}

// Set the information.
void RadarErrorAsynchronous::setInformation(const covariance_t& information) {
    radarInformation_ = information;
    radarCovariance_ = information.inverse();
}

// This evaluates the error term and additionally computes the Jacobians.
bool RadarErrorAsynchronous::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const {

    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool RadarErrorAsynchronous::EvaluateWithMinimalJacobians(double const* const * parameters,
                                                        double* residuals, double** jacobians,
                                                        double** jacobiansMinimal) const {

    // Obtain Transformations from parameters

    // robot pose at t=k
    Eigen::Map<const Eigen::Vector3d> r_WS(&parameters[0][0]);
    const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3],parameters[0][4], parameters[0][5]);
    okvis::kinematics::Transformation T_WS_tk(r_WS, q_WS);

    // speed and biases at t=k
    okvis::SpeedAndBias speedAndBiases;
    for (size_t i = 0; i < 9; ++i) {
      speedAndBiases[i] = parameters[1][i];
    }

    // Transformation from IMU to Radar frame
    okvis::kinematics::Transformation T_IR = radarParameters_.T_IR;
    Eigen::Matrix3d C_IR = T_IR.C(); // R_IR (rotation from Radar to IMU)
    Eigen::Matrix3d C_RI = C_IR.transpose(); // R_RI (rotation from IMU to Radar)
    Eigen::Vector3d p_IR_I = T_IR.r(); // p_IR_I (position of radar in IMU frame)

    // ----- PRE-INTEGRATION PROPAGATION - BEGIN -----

    // this will NOT be changed:
    const Eigen::Matrix3d C_WS_tk = T_WS_tk.C();

    // call the propagation
    const double Delta_t = (tr_ - tk_).toSec();
    Eigen::Matrix<double, 6, 1> Delta_b;
    // ensure unique access
    {
      std::lock_guard<std::mutex> lock(preintegrationMutex_);
      Delta_b = speedAndBiases.tail<6>()
            - speedAndBiases_ref_.tail<6>();
      redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
      if (redoPropagationAlways || (redo_ && ((imuMeasurements_.size() < 50) )) || redoCounter_==0) {     //TODO: Question about ((imuMeasurements_.size() < 50) ) condition
        redoPreintegration(T_WS_tk, speedAndBiases);
        redoCounter_++;
        Delta_b.setZero();
        redo_ = false;
      }
    } // lock releaed

    // actual propagation output:
    std::lock_guard<std::mutex> lock(preintegrationMutex_); // this is a bit stupid, but shared read-locks only come in C++14
    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    Eigen::Vector3d b_g = speedAndBiases.segment<3>(3);
    Eigen::Vector3d v_W_tk = speedAndBiases.head<3>();

    // estimate robot pose at t=r from preintegration results
    okvis::kinematics::Transformation T_WS_tr;
    T_WS_tr.set(r_WS + v_W_tk*Delta_t - 0.5*g_W*Delta_t*Delta_t
                + C_WS_tk * (acc_doubleintegral_ + dp_db_g_ * Delta_b.head<3>() - C_doubleintegral_ * Delta_b.tail<3>()),
                T_WS_tk.q()*Delta_q_*okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b.head<3>()));

    // Rotation from world to IMU at time tr
    Eigen::Matrix3d C_WS_tr = T_WS_tr.C();          // R_WI at time tr
    Eigen::Matrix3d C_SW_tr = C_WS_tr.transpose();  // R_IW at time tr (rotation from World to IMU)
    
    // Velocity in world frame at time tr 
    Eigen::Vector3d v_W_tr = v_W_tk - g_W * Delta_t +
                             C_WS_tk * (acc_integral_ + dv_db_g_ * Delta_b.head<3>() - C_integral_ * Delta_b.tail<3>());
    
    // Velocity in IMU frame at time tr
    Eigen::Vector3d v_I_tr = C_SW_tr * v_W_tr; // R_IW_tr * v_W_tr

    // Angular velocity in IMU frame at time tr (corrected for bias)
    Eigen::Vector3d omega_S_corrected_tr = omega_S_tr_ - b_g; // ω_I_tr - b_g_I

    // Calculate expected radar velocity: v_R_expected = R_RI * (v_I_tr + (ω_I_tr - b_g) × p_IR_I)
    Eigen::Vector3d omega_cross_p_tr = omega_S_corrected_tr.cross(p_IR_I);
    Eigen::Vector3d v_R_expected = C_RI * (v_I_tr + omega_cross_p_tr);



    // Jprop: How propagated pose and speedAndBias at tr changes w.r.t. pose and speedAndBias at tk
    Eigen::Matrix<double,15,15> Jprop =
      Eigen::Matrix<double,15,15>::Identity(); // holds for d/db_g, d/db_a

    Jprop.block<3,3>(0,3) = -okvis::kinematics::crossMx(C_WS_tk*acc_doubleintegral_);
    Jprop.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*Delta_t;
    Jprop.block<3,3>(0,9) = C_WS_tk*dp_db_g_;
    Jprop.block<3,3>(0,12) = -C_WS_tk*C_doubleintegral_;
    Jprop.block<3,3>(3,9) = -C_WS_tk*dalpha_db_g_;
    Jprop.block<3,3>(6,3) = -okvis::kinematics::crossMx(C_WS_tk*acc_integral_);
    Jprop.block<3,3>(6,9) = C_WS_tk*dv_db_g_;
    Jprop.block<3,3>(6,12) = -C_WS_tk*C_integral_;


    // ----- PRE-INTEGRATION PROPAGATION - END -----

    if(useImuCovariance){

        // We need to first transform the Covariance into world frame!
        Eigen::Matrix<double,15,15> T = Eigen::Matrix<double,15,15>::Identity();
        T.topLeftCorner<3,3>() = C_WS_tk;
        T.block<3,3>(3,3) = C_WS_tk;
        T.block<3,3>(6,6) = C_WS_tk;
        Eigen::Matrix<double,15,15> P;
        P = T * P_delta_ * T.transpose();

        // Jvr: Jacobian of radar velocity w.r.t. position, orientation, velocity, at time tr
        // w.r.t. to translation and accel bias is zero, so those columns stay 0
        // v_R_expected = R_RI * (R_IW_tr * v_W_tr + (ω_tr - b_g) × p_IR)
        Eigen::Matrix<double,3,15> Jvr;
        // translation and accel bias: zero
        Jvr.setZero();
        // Rotation: C_RI * C_SW_tr * crossMx(v_W_tr) (rotation affects R_IW_tr, which affects v_I_tr)
        Jvr.block<3,3>(0,3) = C_RI * C_SW_tr * okvis::kinematics::crossMx(v_W_tr);
        // Velocity: C_RI * C_SW_tr
        Jvr.block<3,3>(0,6) = C_RI * C_SW_tr;
        // Gyro bias: C_RI * crossMx(p_IR_I)
        Jvr.block<3,3>(0,9) = C_RI * okvis::kinematics::crossMx(p_IR_I);

        // compute square root information matrix
        Eigen::Matrix<double,3,3> covOverall = radarCovariance_ + Jvr * P * Jvr.transpose(); // consider both sources of covariances
        Eigen::LLT<information_t> lltOfInformation(covOverall.inverse());
        squareRootInformation_ = lltOfInformation.matrixL().transpose();

    }

    else{
        Eigen::LLT<information_t> lltOfInformation(radarCovariance_.inverse());
        squareRootInformation_ = lltOfInformation.matrixL().transpose();
    }


    // calculate the radar error
    measurement_t error = v_R_expected - measurement_;
    error_ = error;
    // weight error by square root of information matrix:
    measurement_t weighted_error = squareRootInformation_ * error;

    // assign residuals
    residuals[0] = weighted_error[0];
    residuals[1] = weighted_error[1];
    residuals[2] = weighted_error[2];

    // Compute jacobians if required
    if(jacobians!=NULL){

        if(jacobians[0]!=NULL){ // w.r.t. robot pose at tk

            // Jacobian of radar velocity error w.r.t. robot pose at tk
            // v_R = R_RI * (R_IW_tr * v_W_tr + (ω_tr - b_g) × p_IR)
            // Chain rule: d/dpose_tk = (d/dpose_tr) * (dpose_tr/dpose_tk) + (d/dv_W_tr) * (dv_W_tr/dpose_tk)

            // Direct effect through propagated pose at tr d/dpose_tr
            Eigen::Matrix<double,3,6> J0_minimal_tr; 
            J0_minimal_tr.setZero();
            // Translation: zero (velocity doesn't depend on position)
            // Rotation affects R_RI * (R_IW_tr * v_W_tr) through R_WI_tr
            J0_minimal_tr.topRightCorner<3,3>() = C_RI * C_SW_tr * okvis::kinematics::crossMx(v_W_tr);

            // Indirect effect through velocity at tr d/dv_W_tr
            Eigen::Matrix<double,3,3> J1_minimal_tr;
            J1_minimal_tr.setZero();
            // Velocity affects v_R through R_RI * (R_IW_tr * v_W_tr)
            J1_minimal_tr.topLeftCorner<3,3>() = C_RI * C_SW_tr;

            // Propagation to xk in minimal representation, p/pose = [translation, rotation]
            Eigen::Matrix<double,3,6> J_minimal_tk;
            Eigen::Matrix<double,6,6> J_dpr_dpk;
            Eigen::Matrix<double,3,6> J_dvr_dpk;
            J_dpr_dpk = Jprop.topLeftCorner<6,6>(); // Jacobian of propagated pose w.r.t. pose at tk
            J_dvr_dpk = Jprop.block<3,6>(6,0); // Jacobian of velocity at tr w.r.t. pose at tk
            J_minimal_tk = J0_minimal_tr * J_dpr_dpk + J1_minimal_tr * J_dvr_dpk;

            // Lift to non-minimal representation
            Eigen::Matrix<double,6,7, Eigen::RowMajor> J0_tk_lift;
            PoseManifold::minusJacobian(parameters[0], J0_tk_lift.data());

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J0(jacobians[0]);
            J0 = squareRootInformation_ * J_minimal_tk * J0_tk_lift;

            // if requested, provide minimal Jacobians
            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
                J0_minimal_mapped = squareRootInformation_ * J_minimal_tk;
              }
            }

        }

        if(jacobians[1]!=NULL){

            // Jacobian of radar velocity error w.r.t. speedAndBias at tk
            // v_R = R_RI * (R_IW_tr * v_W_tr + (ω_tr - b_g) × p_IR)
            // Chain rule: d/dsb_tk = d/dsb_tr * dsb_tr/dsb_tk + d/dpose_tr * dpose_tr/dsb_tk
            
            // Direct effect through velocity at tk
            // dv_R/dv_W_tk = C_RI * d/dv_W_tr (R_IW_tr * v_W_tr) = C_RI * R_IW_tr = C_RI * C_SW_tr
            Eigen::Matrix<double,3,3> J0_v = C_RI * C_SW_tr;
            
            // Direct effect through gyro bias at tk
            // dv_R/db_g = C_RI * d/db_g ((ω_tr - b_g) × p_IR) = C_RI * crossMx(p_IR_I)
            Eigen::Matrix<double,3,3> J0_bg = C_RI * okvis::kinematics::crossMx(p_IR_I);

            Eigen::Matrix<double,3,9> J0_minimal_tr;
            J0_minimal_tr.setZero();
            J0_minimal_tr.topLeftCorner<3,3>() = J0_v;    // velocity part
            J0_minimal_tr.block<3,3>(0,3) = J0_bg; // gyro bias part
            // accel bias: zero (not used in radar error)
            
            // Indirect effect through pose propagation: bias and velocity at tk affect propagated pose at tr
            // dv_R/dpose_tr
            Eigen::Matrix<double,3,6> J1_minimal_tr;
            J1_minimal_tr.setZero();
            J1_minimal_tr.topRightCorner<3,3>() = C_RI * C_SW_tr * okvis::kinematics::crossMx(v_W_tr);
            // Rotation part only, as translation doesn't affect velocity
            
            Eigen::Matrix<double,9,9> J_sbr_dsbk;
            J_sbr_dsbk = Jprop.block<9,9>(6,6); // Jacobian of propagated speedAndBias at tr w.r.t. speedAndBias at tk
            Eigen::Matrix<double,6,9> J_dpr_dsbk;
            J_dpr_dsbk = Jprop.block<6,9>(0,6); // Jacobian of propagated pose at tr w.r.t. speedAndBias at tk
            
            // Combined Jacobian w.r.t. speedAndBias at tk
            Eigen::Matrix<double,3,9> J_minimal_tk;
            J_minimal_tk.setZero();
            J_minimal_tk = J0_minimal_tr * J_sbr_dsbk + J1_minimal_tr * J_dpr_dsbk;

            // No Lift needed (SpeedAndBias uses Euclidean parameterization)
            Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > J1(jacobians[1]);
            J1 = squareRootInformation_ * J_minimal_tk;

            // if requested, provide minimal Jacobians
            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
                J1_minimal_mapped = squareRootInformation_ * J_minimal_tk;
              }
            }

        }

    }

    return true;

}

// Propagates pose, speeds and biases with given IMU measurements.
int RadarErrorAsynchronous::redoPreintegration(const okvis::kinematics::Transformation& /*T_WS*/,
                                 const okvis::SpeedAndBias & speedAndBiases) const {

  // now the propagation
  okvis::Time time = tk_;
  okvis::Time end = tr_;

  // sanity check:
  assert(imuMeasurements_.front().timeStamp<=time);
  if (!(imuMeasurements_.back().timeStamp >= end))
    return -1;  // nothing to do...

  // increments (initialise with identity)
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

  // derivatives of covariance matrix w.r.t. sigmas
  dPdsigma_.resize(4);
  for(size_t j=0; j<4; ++j) {
    dPdsigma_.at(j).setZero();
  }

  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
      it != imuMeasurements_.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = tr_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = tr_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double gyr_sat_mult = 1.0;
    double acc_sat_mult = 1.0;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      gyr_sat_mult *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      acc_sat_mult *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
        - speedAndBiases.segment < 3 > (3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
        - speedAndBiases.segment < 3 > (6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
        + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral_ += acc_integral_ * dt
        + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // Jacobian parts
    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
        + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ += dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    if(useImuCovariance){

        // covariance propagation
        Eigen::Matrix<double, 15, 15> F_delta =
            Eigen::Matrix<double, 15, 15>::Identity();
        // transform
        F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
            acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
        F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
        F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
            + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
        F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
            + 0.25 * (C + C_1) * dt * dt;
        F_delta.block<3, 3>(3, 9) = -dt * C_1;
        F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
            0.5 * (C + C_1) * acc_S_true * dt);
        F_delta.block<3, 3>(6, 9) = 0.5 * dt
            * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
        F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;

        // Q = K * sigma_sq
        Eigen::Matrix<double,15,15> K0 = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,15> K1 = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,15> K2 = Eigen::Matrix<double,15,15>::Zero();
        Eigen::Matrix<double,15,15> K3 = Eigen::Matrix<double,15,15>::Zero();
        K0.block<3,3>(3,3) = gyr_sat_mult*dt * Eigen::Matrix3d::Identity();
        K1.block<3,3>(0,0) = 0.5 * dt*dt*dt * acc_sat_mult*acc_sat_mult*acc_sat_mult * Eigen::Matrix3d::Identity();
        K1.block<3,3>(6,6) = acc_sat_mult*dt* Eigen::Matrix3d::Identity();
        K2.block<3,3>(9,9) = dt * Eigen::Matrix3d::Identity();
        K3.block<3,3>(12,12) = dt * Eigen::Matrix3d::Identity();
        dPdsigma_.at(0) = F_delta*dPdsigma_.at(0)*F_delta.transpose() + K0;
        dPdsigma_.at(1) = F_delta*dPdsigma_.at(1)*F_delta.transpose() + K1;
        dPdsigma_.at(2) = F_delta*dPdsigma_.at(2)*F_delta.transpose() + K2;
        dPdsigma_.at(3) = F_delta*dPdsigma_.at(3)*F_delta.transpose() + K3;

    }


    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == tr_)
      break;

  }

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  if(useImuCovariance){

      // get the weighting:
      // enforce symmetric
      for(int j=0; j<4; ++j) {
        dPdsigma_.at(j) = 0.5 * dPdsigma_.at(j) + 0.5 * dPdsigma_.at(j).transpose().eval();
      }
      P_delta_ = dPdsigma_.at(0)*imuParameters_.sigma_g_c*imuParameters_.sigma_g_c;
      P_delta_ += dPdsigma_.at(1)*imuParameters_.sigma_a_c*imuParameters_.sigma_a_c;
      P_delta_ += dPdsigma_.at(2)*imuParameters_.sigma_gw_c*imuParameters_.sigma_gw_c;
      P_delta_ += dPdsigma_.at(3)*imuParameters_.sigma_aw_c*imuParameters_.sigma_aw_c;

  }



  return i;
}

bool RadarErrorAsynchronous::VerifyJacobianNumDiff(double const* const * parameters,
                                     double** jacobian) const{

  bool success = true;
  const double threshold = 1e-6;
  // Only execute when Jacobians are provided
  if(jacobian != NULL){

      // T_WS at tk
      Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
      const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
      okvis::kinematics::Transformation T_WS_tk(t_WS_W, q_WS);

      // SpeedAndBias at tk
      Eigen::Map<const Eigen::Matrix<double, 9, 1>> speedAndBias(parameters[1]);

      // T_IR
      okvis::kinematics::Transformation T_IR = radarParameters_.T_IR;
      Eigen::Matrix3d C_IR = T_IR.C();
      Eigen::Matrix3d C_RI = C_IR.transpose();
      Eigen::Vector3d p_IR_I = T_IR.r();

      // Compute square root information for unperturbed state (needed for numerical differentiation)
      // We need to evaluate once to get the square root information
      // TODO: Try later without this step by storing it from normal evaluation
      double dummy_residuals[3];
      double* dummy_jacobians[2] = {nullptr, nullptr};
      EvaluateWithMinimalJacobians(parameters, dummy_residuals, dummy_jacobians, nullptr);
      // Now squareRootInformation_ is set for the unperturbed state
      Eigen::Matrix3d squareRootInformation_fixed = squareRootInformation_;

      // x, dx : in minimal representation
      double dx = 1e-7;     // Perturbation size
      Eigen::Matrix<double, 6, 1> delta;     // Perturbation vector for robot pose
      Eigen::Matrix<double, 9, 1> delta_sb;     // Perturbation vector for SpeedAndBias
      // e(x+dx), e(x-dx)
      Eigen::Matrix<double,3,1> ep; // e(x+dx)
      Eigen::Matrix<double,3,1> em; // e(x-dx)

      
      // Helper function to compute error
      // This propagates from tk to tr and computes radar velocity error
      auto computeError = [&](const okvis::kinematics::Transformation& T_WS_tk_val, 
                              const okvis::SpeedAndBias& speedAndBias_val) -> Eigen::Vector3d {
        // Ensure preintegration is done for this state
        // TODO: Check if always preingrating would work as well
        {
          std::lock_guard<std::mutex> lock(preintegrationMutex_);
          Eigen::Matrix<double, 6, 1> Delta_b_val = speedAndBias_val.tail<6>()
                - speedAndBiases_ref_.tail<6>();
          redo_ = redo_ || (Delta_b_val.head<3>().norm() > 0.0003);
          if (redoPropagationAlways || (redo_ && ((imuMeasurements_.size() < 50) )) || redoCounter_==0) {
            redoPreintegration(T_WS_tk_val, speedAndBias_val);
            Delta_b_val.setZero();
            redo_ = false;
            redoCounter_++;
          }
        }
        
        // Propagate to tr
        const Eigen::Vector3d r_WS_tk_val = T_WS_tk_val.r();
        const Eigen::Matrix3d C_WS_tk_val = T_WS_tk_val.C();
        const double Delta_t = (tr_ - tk_).toSec();
        const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();
        
        std::lock_guard<std::mutex> lock(preintegrationMutex_);
        Eigen::Matrix<double, 6, 1> Delta_b_val = speedAndBias_val.tail<6>()
              - speedAndBiases_ref_.tail<6>();

        Eigen::Vector3d b_g_val = speedAndBias_val.segment<3>(3);
        Eigen::Vector3d v_W_tk_val = speedAndBias_val.head<3>();
        
        // Propagate pose to tr3
        okvis::kinematics::Transformation T_WS_tr;
        T_WS_tr.set(r_WS_tk_val + v_W_tk_val*Delta_t - 0.5*g_W*Delta_t*Delta_t
                    + C_WS_tk_val * (acc_doubleintegral_ + dp_db_g_ * Delta_b_val.head<3>() - C_doubleintegral_ * Delta_b_val.tail<3>()),
                    T_WS_tk_val.q()*Delta_q_*okvis::kinematics::deltaQ(-dalpha_db_g_*Delta_b_val.head<3>()));
        
        // Velocity at tr (using full IMU preintegration)
        Eigen::Vector3d v_W_tr = v_W_tk_val - g_W * Delta_t +
                                 C_WS_tk_val * (acc_integral_ + dv_db_g_ * Delta_b_val.head<3>() - C_integral_ * Delta_b_val.tail<3>());
        
        // Compute radar velocity
        Eigen::Matrix3d C_WS_tr = T_WS_tr.C();
        Eigen::Matrix3d C_SW_tr = C_WS_tr.transpose();
        Eigen::Vector3d v_I_tr = C_SW_tr * v_W_tr;
        Eigen::Vector3d omega_S_corrected_tr_val = omega_S_tr_ - b_g_val;
        Eigen::Vector3d omega_cross_p_tr = omega_S_corrected_tr_val.cross(p_IR_I);
        Eigen::Vector3d v_R_expected = C_RI * (v_I_tr + omega_cross_p_tr);
        
        // Use the fixed square root information from unperturbed state
        return squareRootInformation_fixed * (v_R_expected - measurement_);
      };

      // if jacobians for robot pose are provided
      if(jacobian[0]!=NULL){

          // Jacobian w.r.t robot pose
          Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > Jp(jacobian[0]);
          Eigen::Matrix<double,3,6> Jpn_minimal;
          Jpn_minimal.setZero();

          // Numerical Jacobian w.r.t robot pose
          for (size_t i = 0; i < 6; ++i) {

            delta.setZero();

            // x+dx
            delta[i] = dx;
            okvis::kinematics::Transformation Tp_tk = T_WS_tk;
            Tp_tk.oplus(delta);
            ep = computeError(Tp_tk, speedAndBias);

            // x-dx
            delta[i] = -dx;
            okvis::kinematics::Transformation Tm_tk = T_WS_tk;
            Tm_tk.oplus(delta);
            em = computeError(Tm_tk, speedAndBias);

            // difference quotient (e(x+dx) - e(x-dx)) / (2*dx)
            Jpn_minimal.col(i) = (ep - em) / (2 * dx);
          }

          // liftJacobian to switch to non-minimal representation
          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> Jpn_lift;
          PoseManifold::minusJacobian(parameters[0], Jpn_lift.data());

          // hallucinate Jacobian w.r.t. state
          Eigen::Matrix<double, 3, 7> Jpn;
          Jpn = Jpn_minimal * Jpn_lift;

          // Test if analytically and numerically computed Jacobians are close enough
          double norm_diff = (Jp - Jpn).norm();
          if (norm_diff >= threshold) {
            success = false;
            std::cout << "RadarErrorAsynchronous: Jacobian w.r.t. robot pose differs! Norm of difference: " << norm_diff << std::endl;
            // Temporary debug print
            std::cout << "--- ANALYTICAL Jp (3x7) ---\n" << Jp << std::endl;
            std::cout << "--- NUMERICAL Jpn (3x7) ---\n" << Jpn << std::endl;
            std::cout << "T_WS_tk for debugging:\n" << T_WS_tk.T() << std::endl;
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
            okvis::SpeedAndBias speedAndBias_p = speedAndBias;
            speedAndBias_p[i] += dx;
            ep = computeError(T_WS_tk, speedAndBias_p);

            // x-dx
            delta_sb[i] = -dx;
            okvis::SpeedAndBias speedAndBias_m = speedAndBias;
            speedAndBias_m[i] -= dx;
            em = computeError(T_WS_tk, speedAndBias_m);

            // difference quotient (e(x+dx) - e(x-dx)) / (2*dx)
            Jsbn_minimal.col(i) = (ep - em) / (2 * dx);
          }

          // SpeedAndBias uses Euclidean parameterization, so lift is identity
          Eigen::Matrix<double, 3, 9> Jsbn;
          Jsbn = Jsbn_minimal;

          // Test if analytically and numerically computed Jacobians are close enough
          double norm_diff = (Jsb - Jsbn).norm();
          if (norm_diff >= threshold) {
            success = false;
            std::cout << "RadarErrorAsynchronous: Jacobian w.r.t. SpeedAndBias differs! Norm of difference: " << norm_diff << std::endl;
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
