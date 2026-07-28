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

#include <okvis/assert_macros.hpp>
#include <okvis/LidarMotionUndistortion.hpp>
#include <okvis/SubmappingUtils.hpp>

namespace okvis{
LidarMotionUndistortion::LidarMotionUndistortion(const State& initialState, const kinematics::Transformation& T_WS_live,
                                                 const kinematics::Transformation& T_SL,
                                                 const LidarMeasurementDeque& lidarScan,
                                                 const ImuMeasurementDeque& imuMeasurementDeque)
                                                 : initialState_(initialState),
                                                 T_WS_live_(T_WS_live), T_SW_live_(T_WS_live.inverse()), T_SL_(T_SL),
                                                 lidarScan_(lidarScan), imuMeasurementDeque_(imuMeasurementDeque){};

bool LidarMotionUndistortion::deskew() {
  if(initialState_.id.value() == 0){
    return false;
  }
  if(lidarScan_.empty()){
    return false;
  }
  OKVIS_ASSERT_TRUE(std::runtime_error, imuMeasurementDeque_.front().timeStamp <= initialState_.timestamp, "IMU Measurement Mismatch Motion Undistortion with State");
  OKVIS_ASSERT_TRUE(std::runtime_error, imuMeasurementDeque_.back().timeStamp >= lidarScan_.back().timeStamp, "IMU Measurement Mismatch Motion Undistortion with Lidar");

  // do the propagation...
  kinematics::Transformation T_WS_0 = initialState_.T_WS;
  SpeedAndBias speedAndBias_0;
  speedAndBias_0.head<3>() = initialState_.v_W;
  speedAndBias_0.segment<3>(3) = initialState_.b_g;
  speedAndBias_0.tail<3>() = initialState_.b_a;
  kinematics::Transformation T_WS_1;
  SpeedAndBias speedAndBias_1;

  State state;
  BatchedLidarPropagator propagator(initialState_.timestamp, imuMeasurementDeque_);
  lidarScanPostProcessed_.reserve(lidarScan_.size());

  size_t i = 0;
  kinematics::Transformation T_SliveL = T_SW_live_ * initialState_.T_WS * T_SL_;
  //Check if the first lidar measurement timestamp is already the first state
  auto lidarMeasurementIter = lidarScan_.begin();
  if(lidarMeasurementIter->timeStamp == initialState_.timestamp){
    state = initialState_;
    lidarScanPostProcessed_.push_back((T_SliveL.C() * lidarMeasurementIter->measurement.rayMeasurement + T_SliveL.r()).cast<float>());
    lidarMeasurementIter++;
  }

  // Iterate measurements
  okvis::Time previousTime(0);
  while(lidarMeasurementIter != lidarScan_.end()) {
    if(lidarMeasurementIter->timeStamp < initialState_.timestamp ){
      lidarMeasurementIter++;
      continue;
    }

    if(previousTime != lidarMeasurementIter->timeStamp ) {
      propagator.appendTo(imuMeasurementDeque_,
                          T_WS_0, speedAndBias_0, lidarMeasurementIter->timeStamp );
      propagator.getState(T_WS_0, speedAndBias_0,
                          T_WS_1, speedAndBias_1, state.omega_S);
      previousTime = lidarMeasurementIter->timeStamp ;
      T_SliveL = T_SW_live_ * T_WS_1 * T_SL_;
    }

    // Write into return vector
    lidarScanPostProcessed_.push_back((T_SliveL.C() * lidarMeasurementIter->measurement.rayMeasurement + T_SliveL.r()).cast<float>());
    ++lidarMeasurementIter;
  }
  return true;

}

bool LidarMotionUndistortion::downsample(size_t num_output_points, double voxel_grid_resolution)
{
  if(lidarScanPostProcessed_.empty()){
    return false;
  }
  okvis::VoxelDownSampleInplace(lidarScanPostProcessed_, voxel_grid_resolution);
  randomDownsample(num_output_points);
  return true;
}

void LidarMotionUndistortion::randomDownsample(size_t num_output_points) {
  if(lidarScanPostProcessed_.size() > num_output_points) {
    std::shuffle(lidarScanPostProcessed_.begin(), lidarScanPostProcessed_.end(), std::mt19937{std::random_device{}()});
    // then take first n
    lidarScanPostProcessed_.resize(num_output_points);
  }

  return;
}

size_t LidarMotionUndistortion::filterObserved(const SupereightMapType* map, const okvis::kinematics::Transformation& T_WM)
{
  if(lidarScanPostProcessed_.size() == 0)
  {
    return 0;
  }

  Eigen::Matrix<float, 3, 4> T_MS_live = T_WM.inverse().T3x4().cast<float>() * T_WS_live_.T().cast<float>();
  const Eigen::Matrix3f R_MS = T_MS_live.leftCols<3>();
  const Eigen::Vector3f t_MS = T_MS_live.col(3);

  size_t write_idx = 0;
  
  for (size_t i = 0; i < lidarScanPostProcessed_.size(); ++i) {
    auto& lidarMeasurement = lidarScanPostProcessed_[i];
    const Eigen::Vector3f pt = R_MS * lidarMeasurement + t_MS;
    if (map->interpField<se::Safe::On>(pt)) {
      lidarScanPostProcessed_[write_idx++] = lidarMeasurement;
    }
  }

  lidarScanPostProcessed_.resize(write_idx);
  return write_idx;
}

}
