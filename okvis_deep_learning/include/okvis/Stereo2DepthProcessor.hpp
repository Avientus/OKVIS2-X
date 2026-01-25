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

#ifndef STEREO_DEPTH_STEREO2DEPTH_PROCESSOR_HPP
#define STEREO_DEPTH_STEREO2DEPTH_PROCESSOR_HPP

#include <map>
#include <thread>
#include <atomic>
#include <iostream>

#include <opencv2/calib3d.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <okvis/Measurements.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/QueuedTrajectory.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViSensorBase.hpp>
#include <okvis/DeepLearningProcessor.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

namespace okvis {

class Stereo2DepthProcessor : public okvis::DeepLearningProcessor{

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)


  Stereo2DepthProcessor(okvis::ViParameters &parameters, std::string modelDir);
  Stereo2DepthProcessor() = default;
  virtual ~Stereo2DepthProcessor();


  /// @brief Display image and sigma
  /// @param images Output map of images
  /// @param timestamp Output timestamp of the stereo pair
  virtual void display(std::map<std::string, cv::Mat> &images, okvis::Time &timestamp) override;
  
  /// @brief Legacy display function (backward compatibility)
  virtual void display(std::map<std::string, cv::Mat> &images) override;

  /// \name Add measurements to the algorithm.
  /**
   * \brief              Add a set of new image.
   * \param stamp        The image timestamp.
   * \param images       The images.
   * \param depthImages  For consistency with the other processors, WILL NOT BE USED
   * \return             Returns true normally. False, if the previous one has not been processed yet.
   */
  virtual bool addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images, 
                         const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImages) override final;

  /// @brief Check whether the processor is finished.
  bool finishedProcessing() override;

protected:

  struct StereoCameraData {
    cv::Mat leftImage;
    cv::Mat rightImage;
  };

  typedef okvis::Measurement<StereoCameraData> StereoMeasurement;

  struct VisualizationData {
    StereoMeasurement frame;
    cv::Mat depthImage;  // Visualization (colorized)
    cv::Mat rawDepthImage;  // Raw depth in meters (CV_32FC1)
    cv::Mat sigmaImage;
  };

  /// @brief Processing loops and according threads.
  void processing();
  
  /// @brief Function where the actual neural network predicts the depth from the stereo images
  /// @param frame The stereo images measurement which will then be processed by the neural network
  void processStereoNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frame);

  torch::jit::script::Module depthModel_;
  okvis::threadsafe::Queue<VisualizationData> visualisationsQueue_;

  std::set<size_t> greyScaleCameras_;

  double focalLength_;
  double baseline_;

  cv::Mat rectMapLeft0_, rectMapLeft1_, rectMapRight0_, rectMapRight1_;
  bool needRectify_; // If true, stereo image is rectified before network
  int imgWidth_; // Image width
  int imgHeight_; // Image height

  size_t idLeft_; // Left camera id
  size_t idRight_; // Right camera id
};

} // namespace srl

#endif //STEREO_DEPTH_STEREO2DEPTH_PROCESSOR_HPP
