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

/*
 * TestAltimeterErrorAsynchronous.cpp
 *
 * Verifies AltimeterErrorAsynchronous: a single-state, IMU-propagated vertical-
 * velocity factor (see okvis/ceres/AltimeterErrorAsynchronous.hpp). The scenario
 * is a constant vertical-velocity trajectory -- the altimeter factor should pull
 * the estimated velocity towards the true vertical rate, and (via the IMU link)
 * the second pose's height towards the true climb.
 */

#include "glog/logging.h"
#include "ceres/ceres.h"
#include <gtest/gtest.h>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/AltimeterErrorAsynchronous.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

TEST(okvisTestSuite, AltimeterErrorAsynchronous){

  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  okvis::ceres::Map map;

  // IMU parameters
  okvis::ImuParameters imuParameters;
  imuParameters.a0.setZero();
  imuParameters.g = 9.81;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.rate = 1000; // 1 kHz
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
  imuParameters.sigma_aw_c = 2.0e-5;
  imuParameters.tau = 3600.0;

  // ground truth: constant vertical (climb) velocity, no rotation
  const double vZTrue = 1.5; // [m/s]
  Eigen::Vector3d v_W(0.0, 0.0, vZTrue);

  Eigen::Vector3d b_g = Eigen::Vector3d(0.0001,-0.0002,0.0003);
  Eigen::Vector3d b_a = Eigen::Vector3d(-0.001,0.002,0.003);

  const double Dt = 1.02; // [sec]
  const double dt = 0.005; // 200 Hz
  const okvis::kinematics::Transformation T_WS_0;
  const okvis::kinematics::Transformation T_WS_1(v_W*(Dt-0.02), Eigen::Quaterniond::Identity());
  okvis::SpeedAndBias zeroSpeedAndBias = okvis::SpeedAndBias::Zero();
  zeroSpeedAndBias.head<3>() = v_W;

  const double altimeterVariance = 1.0e-4; // [(m/s)^2]

  // parameter blocks
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock0(
      new okvis::ceres::PoseParameterBlock(T_WS_0, 1, okvis::Time(0.01)));
  std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock0(
      new okvis::ceres::SpeedAndBiasParameterBlock(zeroSpeedAndBias, 2, okvis::Time(0.01)));
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock1(
      new okvis::ceres::PoseParameterBlock(T_WS_1, 3, okvis::Time(Dt-0.01)));
  std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock1(
      new okvis::ceres::SpeedAndBiasParameterBlock(zeroSpeedAndBias, 4, okvis::Time(Dt-0.01)));

  map.addParameterBlock(poseParameterBlock0, okvis::ceres::Map::Pose4d);
  map.setParameterBlockConstant(poseParameterBlock0);
  map.addParameterBlock(speedAndBiasParameterBlock0);
  std::shared_ptr<::ceres::CostFunction> speedAndBiasError(
      new okvis::ceres::SpeedAndBiasError(zeroSpeedAndBias, 0.01, 0.0001, 0.0001));
  map.addResidualBlock(speedAndBiasError, NULL, speedAndBiasParameterBlock0);
  map.addParameterBlock(poseParameterBlock1, okvis::ceres::Map::Pose6d);
  map.addParameterBlock(speedAndBiasParameterBlock1);

  // generate IMU measurements and altimeter measurements (20 Hz) along the way
  okvis::ImuMeasurementDeque imuMeasurements;
  int ctr = 0;
  ceres::ResidualBlockId altimeterMeasurementId = 0;
  for(double t = 0; t < Dt; t += dt) {
    Eigen::Vector3d gyr = b_g + imuParameters.sigma_g_c/sqrt(dt) * Eigen::Vector3d::Random();
    // constant velocity, no rotation -> zero net world acceleration, gravity-compensated
    Eigen::Vector3d acc = Eigen::Vector3d(0, 0, imuParameters.g) + b_a
        + imuParameters.sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random();
    imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(t), okvis::ImuSensorReadings(gyr, acc)));

    if(ctr % 10 == 0 && t > 0.01 && t < Dt-0.01){
      std::shared_ptr<::ceres::CostFunction> altimeterError(new okvis::ceres::AltimeterErrorAsynchronous(
          vZTrue, 1.0/altimeterVariance, imuMeasurements, imuParameters, okvis::Time(0.01), okvis::Time(t)));
      altimeterMeasurementId = map.addResidualBlock(altimeterError, NULL,
                             poseParameterBlock0, speedAndBiasParameterBlock0);
    }
    ctr++;
  }

  // IMU error term linking the two states
  std::shared_ptr<::ceres::CostFunction> imuError(new okvis::ceres::ImuError(
      imuMeasurements, imuParameters, okvis::Time(0.01), okvis::Time(Dt-0.01)));
  ceres::ResidualBlockId imuId = map.addResidualBlock(imuError, NULL,
                       poseParameterBlock0, speedAndBiasParameterBlock0,
                       poseParameterBlock1, speedAndBiasParameterBlock1);

  // solve
  map.options.trust_region_strategy_type = ::ceres::DOGLEG;
  map.options.minimizer_progress_to_stdout = false;
  map.options.max_num_iterations = 100;
  FLAGS_stderrthreshold = google::WARNING;
  map.solve();

  // check Jacobians
  OKVIS_ASSERT_TRUE(Exception, map.isJacobianCorrect(imuId, 1.0e-3),
                    "IMU error Jacobian did not verify");
  OKVIS_ASSERT_TRUE(Exception, map.isJacobianCorrect(altimeterMeasurementId, 1.0e-3),
                    "Altimeter error Jacobian did not verify");

  // check convergence: estimated vertical velocity at state0 should match truth,
  // and (via the IMU link) state1's height should match the true climb.
  const double vZEstimated = speedAndBiasParameterBlock0->estimate().head<3>().z();
  OKVIS_ASSERT_TRUE(Exception, std::fabs(vZEstimated - vZTrue) < 0.1,
                    "estimated vertical velocity not close enough: " << vZEstimated << " vs " << vZTrue);

  const double zEstimated = poseParameterBlock1->estimate().r().z();
  const double zTrue = T_WS_1.r().z();
  OKVIS_ASSERT_TRUE(Exception, std::fabs(zEstimated - zTrue) < 0.1,
                    "estimated height at state1 not close enough: " << zEstimated << " vs " << zTrue);
}
