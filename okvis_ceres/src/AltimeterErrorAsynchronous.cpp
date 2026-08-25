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
 * @file AltimeterErrorAsynchronous.cpp
 * @brief Source File for the AltimeterErrorAsynchronous class
 */

#include <eigen3/Eigen/Core>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/AltimeterErrorAsynchronous.hpp>
#include <okvis/ceres/ImuError.hpp> // required for propagation
#include <okvis/ceres/ode/ode.hpp>
#include <okvis/Parameters.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

std::atomic_bool AltimeterErrorAsynchronous::redoPropagationAlways(false);
std::atomic_bool AltimeterErrorAsynchronous::useImuCovariance(true);

AltimeterErrorAsynchronous::AltimeterErrorAsynchronous() {
  measurement_ = 0.0;
  altimeterInformation_ = 0.0;
  altimeterCovariance_ = 0.0;
}

// Constructor with provided information.
AltimeterErrorAsynchronous::AltimeterErrorAsynchronous(const measurement_t & measurement, const information_t & information,
                                           const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                                           const okvis::Time& tk, const okvis::Time& tr){
    setMeasurement(measurement);
    setInformation(information);
    setImuMeasurements(imuMeasurements);
    setImuParameters(imuParameters);
    setTk(tk);
    setTr(tr);

    // derivatives of covariance matrix w.r.t. sigmas
    dPdsigma_.resize(4);
    for(size_t j=0; j<4; ++j) {
      dPdsigma_.at(j).setZero();
    }
}

// Set the information.
void AltimeterErrorAsynchronous::setInformation(const information_t& information) {
    altimeterInformation_ = information;
    altimeterCovariance_ = information > 0.0 ? 1.0 / information : 0.0;
}

// This evaluates the error term and additionally computes the Jacobians.
bool AltimeterErrorAsynchronous::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const {

    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

bool AltimeterErrorAsynchronous::EvaluateWithMinimalJacobians(double const* const * parameters,
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
      if (redoPropagationAlways || (redo_ && ((imuMeasurements_.size() < 50) )) || redoCounter_==0) {
        redoPreintegration(T_WS_tk, speedAndBiases);
        redoCounter_++;
        Delta_b.setZero();
        redo_ = false;
      }
    } // lock released

    // actual propagation output:
    std::lock_guard<std::mutex> lock(preintegrationMutex_); // this is a bit stupid, but shared read-locks only come in C++14
    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    Eigen::Vector3d v_W_tk = speedAndBiases.head<3>();

    // Velocity in world frame at time tr (this is directly our measurement model --
    // no sensor-frame rotation or lever arm needed, since the altimeter-derived
    // vertical rate is already expressed in the world/gravity-aligned frame).
    Eigen::Vector3d v_W_tr = v_W_tk - g_W * Delta_t +
                             C_WS_tk * (acc_integral_ + dv_db_g_ * Delta_b.head<3>() - C_integral_ * Delta_b.tail<3>());

    // Jprop: How propagated pose and speedAndBias at tr changes w.r.t. pose and speedAndBias at tk
    // (state ordering: position(0:3), rotation(3:6), velocity(6:9), gyro bias(9:12), accel bias(12:15))
    Eigen::Matrix<double,15,15> Jprop =
      Eigen::Matrix<double,15,15>::Identity();

    Jprop.block<3,3>(0,3) = -okvis::kinematics::crossMx(C_WS_tk*acc_doubleintegral_);
    Jprop.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*Delta_t;
    Jprop.block<3,3>(0,9) = C_WS_tk*dp_db_g_;
    Jprop.block<3,3>(0,12) = -C_WS_tk*C_doubleintegral_;
    Jprop.block<3,3>(3,9) = -C_WS_tk*dalpha_db_g_;
    Jprop.block<3,3>(6,3) = -okvis::kinematics::crossMx(C_WS_tk*acc_integral_);
    Jprop.block<3,3>(6,9) = C_WS_tk*dv_db_g_;
    Jprop.block<3,3>(6,12) = -C_WS_tk*C_integral_;

    // ----- PRE-INTEGRATION PROPAGATION - END -----

    // row of Jprop corresponding to the z-component of velocity (row 6+2=8)
    Eigen::Matrix<double,1,15> Jvz = Jprop.row(8);

    if(useImuCovariance){

        // We need to first transform the Covariance into world frame!
        Eigen::Matrix<double,15,15> T = Eigen::Matrix<double,15,15>::Identity();
        T.topLeftCorner<3,3>() = C_WS_tk;
        T.block<3,3>(3,3) = C_WS_tk;
        T.block<3,3>(6,6) = C_WS_tk;
        Eigen::Matrix<double,15,15> P;
        P = T * P_delta_ * T.transpose();

        // Uncertainty of v_W_tr.z() induced by IMU propagation: just the
        // (velocity_z, velocity_z) entry of the (already world-frame) covariance --
        // no bilinear form needed since our residual IS a component of that state.
        const double imuInducedVariance = P(8, 8);
        const double covOverall = altimeterCovariance_ + imuInducedVariance;
        squareRootInformation_ = covOverall > 0.0 ? 1.0 / std::sqrt(covOverall) : 0.0;

    }
    else{
        squareRootInformation_ = altimeterCovariance_ > 0.0 ? 1.0 / std::sqrt(altimeterCovariance_) : 0.0;
    }

    // calculate the altimeter error
    const double error = v_W_tr.z() - measurement_;
    error_ = error;
    // weight error by square root of information:
    const double weighted_error = squareRootInformation_ * error;

    // assign residual
    residuals[0] = weighted_error;

    // Compute jacobians if required
    if(jacobians!=NULL){

        if(jacobians[0]!=NULL){ // w.r.t. robot pose at tk

            // minimal Jacobian: [position(3), rotation(3)] columns of Jvz
            Eigen::Matrix<double,1,6> J0_minimal = Jvz.head<6>();

            // lift to non-minimal representation
            Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J0_lift;
            PoseManifold::minusJacobian(parameters[0], J0_lift.data());

            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J0(jacobians[0]);
            J0 = squareRootInformation_ * J0_minimal * J0_lift;

            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
                J0_minimal_mapped = squareRootInformation_ * J0_minimal;
              }
            }
        }

        if(jacobians[1]!=NULL){ // w.r.t. speedAndBias at tk

            // minimal Jacobian: [velocity(3), gyro bias(3), accel bias(3)] columns of Jvz
            Eigen::Matrix<double,1,9> J1_minimal = Jvz.tail<9>();

            // SpeedAndBias uses Euclidean parameterization, so no lift needed
            Eigen::Map<Eigen::Matrix<double, 1, 9, Eigen::RowMajor> > J1(jacobians[1]);
            J1 = squareRootInformation_ * J1_minimal;

            if (jacobiansMinimal != NULL) {
              if (jacobiansMinimal[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 1, 9, Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
                J1_minimal_mapped = squareRootInformation_ * J1_minimal;
              }
            }
        }
    }

    return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
// Identical boilerplate to RadarErrorAsynchronous/GpsErrorAsynchronous -- no shared
// base class exists in this codebase for this preintegration scheme (copy-adapt is
// the established convention for each new asynchronous sensor factor).
int AltimeterErrorAsynchronous::redoPreintegration(const okvis::kinematics::Transformation& /*T_WS*/,
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

} /* namespace ceres */
} /* namespace okvis */
