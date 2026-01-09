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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <okvis/timing/Timer.hpp>
#include <okvis/Stereo2DepthProcessor.hpp>
#include <okvis/utils.hpp>

namespace okvis {

Stereo2DepthProcessor::Stereo2DepthProcessor(
  okvis::ViParameters &parameters, std::string modelDir) { 

  for(size_t i = 0; i < parameters.camera.stereo_indices.size(); i++) {
    size_t index_i = parameters.camera.stereo_indices[i];
    if(!parameters.nCameraSystem.cameraType(index_i).isColour){
      greyScaleCameras_.insert(index_i);
    }
  }

  if(greyScaleCameras_.size() != 2) {
    throw std::runtime_error("Can not run the stereo network with more than two cameras which are greyscale");
  }

  idLeft_ = parameters.camera.stereo_indices[0];
  idRight_ = parameters.camera.stereo_indices[1];

  Eigen::VectorXd intrinsics;
  kinematics::Transformation T_lr;
  if(!parameters.nCameraSystem.cameraType(idLeft_).depthType.needRectify) {
    needRectify_ = false;
    imgWidth_ = parameters.nCameraSystem.cameraGeometry(idLeft_)->imageWidth();
    imgHeight_ = parameters.nCameraSystem.cameraGeometry(idLeft_)->imageHeight();
    parameters.nCameraSystem.cameraGeometry(idLeft_)->getIntrinsics(intrinsics);
    T_lr = parameters.nCameraSystem.T_SC(idLeft_)->inverse() *
      (*parameters.nCameraSystem.T_SC(idRight_));
  } else {
    needRectify_ = true;
    imgWidth_ = parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->imageWidth();
    imgHeight_ = parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->imageHeight();
    parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)->getIntrinsics(intrinsics);
    T_lr = parameters.nCameraSystem.rectifyT_SC(idLeft_)->inverse() *
      (*parameters.nCameraSystem.rectifyT_SC(idRight_));
  }
  
  
  focalLength_ = intrinsics(0);  // left camera focal length
  baseline_ = T_lr.r().norm();
  LOG(INFO) << "Stereo network image width, height, focal length [pixel], and baseline [m] are " 
    << imgWidth_ << ", " << imgHeight_ << ", " << focalLength_ << ", " << baseline_;

  // Load depth model from traced model file.
  try {
    depthModel_ = torch::jit::load(modelDir + "/depth-model.pt", torch::kCUDA);

    // Warm up forward pass for GPU.
    LOG(INFO) << "Warming up GPU ...";
    torch::Tensor leftTensor = torch::zeros({480, 640}).to(torch::kU8).to(torch::kCUDA);
    torch::Tensor rightTensor = torch::zeros({480, 640}).to(torch::kU8).to(torch::kCUDA);
    for(int i = 0; i < 10; i++) {
      LOG(INFO) << "GPU warmup iteration " << i << std::endl;
      depthModel_.forward({leftTensor, rightTensor});
    }
    LOG(INFO) << "Done warming up GPU.";
  }
  catch (const c10::Error& e) {
    LOG(ERROR) << "Error loading the depth model from " << modelDir + "/depth-model.pt" << e.what();
    return;
  }

  // Compute rectification map
  if (needRectify_) {
    parameters.nCameraSystem.rectifyCameraGeometry(idLeft_)
      ->getRectifyMap(rectMapLeft0_, rectMapLeft1_);
    parameters.nCameraSystem.rectifyCameraGeometry(idRight_)
      ->getRectifyMap(rectMapRight0_, rectMapRight1_);
  }

  // Starting depth processing thread.
  shutdown_ = false;
  processingThread_ = std::thread(&Stereo2DepthProcessor::processing, this);
}

Stereo2DepthProcessor::~Stereo2DepthProcessor() {
  cameraMeasurementsQueue_.Shutdown();
  visualisationsQueue_.Shutdown();
  shutdown_ = true;
  if(processingThread_.joinable()) {
    processingThread_.join();
  }
}

void Stereo2DepthProcessor::display(std::map<std::string, cv::Mat> &images) {
  if(visualisationsQueue_.Empty()) {
    return;
  }
  VisualizationData vis_data;
  if(visualisationsQueue_.PopNonBlocking(&vis_data)) {
    images["leftImage"] = vis_data.frame.measurement.leftImage;
    images["rightImage"] = vis_data.frame.measurement.rightImage;
    images["stereoDepth"] = vis_data.depthImage;  // Visualization
    if(!vis_data.rawDepthImage.empty()) {
      images["stereoDepthRaw"] = vis_data.rawDepthImage;  // Raw depth in meters
    }
    if(!vis_data.sigmaImage.empty()) {
      images["stereoSigma"] = vis_data.sigmaImage;
    }
  }
}

bool Stereo2DepthProcessor::addImages(const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & images,
                                      const std::map<size_t, std::pair<okvis::Time, cv::Mat>> & depthImage) {

  std::map<size_t, std::vector<okvis::CameraMeasurement>> camera_measurements;
  for(auto& it: images){
    okvis::CameraMeasurement camMeasurement;
    camMeasurement.measurement.image = it.second.second;
    camMeasurement.timeStamp = it.second.first;
    camMeasurement.sensorId = it.first;
    camera_measurements[it.first] = {camMeasurement};
  }

  if(blocking_){
    cameraMeasurementsQueue_.PushBlockingIfFull(camera_measurements, 1);
    return true;
  }
  else{
    const int queue_size = 10;
    if(cameraMeasurementsQueue_.PushNonBlockingDroppingIfFull(camera_measurements, queue_size)) {
      DLOG(WARNING) <<  "stereo depth frame drop";
      return false;
    }
    else{
      return true;
    }
  }
}

void Stereo2DepthProcessor::processStereoNetwork(std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
  okvis::TimerSwitchable tAll("DNN 1 Stereo depth");

  // Set the processing flag true.
  isProcessing_ = true;

  auto& frame0 = frames.at(idLeft_).front();
  auto& frame1 = frames.at(idRight_).front();
  // Stereo rectification
  if (needRectify_) {
      cv::Mat rectLeftImage, rectRightImage;
      cv::remap(frame0.measurement.image, rectLeftImage, rectMapLeft0_, rectMapLeft1_, cv::INTER_LINEAR);
      cv::remap(frame1.measurement.image, rectRightImage, rectMapRight0_, rectMapRight1_, cv::INTER_LINEAR);
      frame0.measurement.image = rectLeftImage;
      frame1.measurement.image = rectRightImage;
  }

  okvis::TimerSwitchable t0("DNN 1.1 cv::Mat to Tensor");
  auto leftTensor = cvMatToTensor(frame0.measurement.image).to(torch::kCUDA);
  auto rightTensor = cvMatToTensor(frame1.measurement.image).to(torch::kCUDA);
  t0.stop();

  okvis::TimerSwitchable t1("DNN 1.2 Actual model inference");
  torch::Tensor networkOutput = depthModel_.forward({leftTensor, rightTensor}).toTensor();
  t1.stop();
  torch::Tensor outputDisparity, disparitySigma;
  if (networkOutput.sizes()[0] == 2) {
    outputDisparity = networkOutput.index({0,torch::indexing::Slice(),torch::indexing::Slice()});
    disparitySigma = networkOutput.index({1,torch::indexing::Slice(),torch::indexing::Slice()});
  }
  else { // if sigma is not available output 0 values (will not be used)
    outputDisparity = networkOutput;
    disparitySigma = torch::zeros({networkOutput.sizes()[0], networkOutput.sizes()[1]}).to(torch::kU8).to(torch::kCUDA);
  }
  outputDisparity = outputDisparity.squeeze();

  torch::Tensor outputDepth = (focalLength_ * baseline_) / outputDisparity;
  torch::Tensor outputSigma = 2.0*(outputDepth / outputDisparity) * disparitySigma;

  // Put some large value for invalid uncertainties
  outputSigma.index_put_({torch::isinf(outputSigma)}, 100.0f);
  outputSigma.index_put_({torch::isnan(outputSigma)}, 100.0f);

  frame0.measurement.depthImage = tensorToCvMatFloat(outputDepth.to(torch::kFloat).detach().cpu()).clone();
  frame0.measurement.sigmaImage = tensorToCvMatFloat(outputSigma.to(torch::kFloat).detach().cpu()).clone();

  if(imageCallback_) {
    imageCallback_(frames);
  }

  torch::Tensor vis = ((outputDisparity / 60.0f).clamp(0.0, 1.0f) * 255.0f).to(torch::kU8);
  cv::Mat visMat = tensorToCvMatByte(vis.detach().cpu());
  cv::Mat visMatColored;
  cv::applyColorMap(visMat, visMatColored, cv::COLORMAP_INFERNO);

  torch::Tensor visSigma = ((disparitySigma / 10.0f).clamp(0.0, 1.0f) * 255.0f).to(torch::kU8);
  cv::Mat visSigmaMat = tensorToCvMatByte(visSigma.detach().cpu());
  cv::Mat visSigmaMatColored;
  cv::applyColorMap(visSigmaMat, visSigmaMatColored, cv::COLORMAP_JET);
  
  StereoMeasurement frame;
  frame.measurement.leftImage = frame0.measurement.image;
  frame.measurement.rightImage = frame1.measurement.image;
  frame.timeStamp = frame0.timeStamp;
  VisualizationData visData;
  visData.frame = frame;
  visData.depthImage = visMatColored;  // Visualization (colorized)
  visData.rawDepthImage = frame0.measurement.depthImage.clone();  // Raw depth in meters
  visData.sigmaImage = visSigmaMatColored;
  visualisationsQueue_.PushNonBlockingDroppingIfFull(visData, 1);

  // Set the processing flag false.
  isProcessing_ = false;

  tAll.stop();
}

void Stereo2DepthProcessor::processing() {
  while(!shutdown_) {
    std::map<size_t, std::vector<okvis::CameraMeasurement>> frame;
    if(blocking_){
      while(cameraMeasurementsQueue_.PopBlocking(&frame)) {
        processStereoNetwork(frame);
      }
    }
    else{
      while(cameraMeasurementsQueue_.PopNonBlocking(&frame)) {
        processStereoNetwork(frame);
      }
    }
  }
}

bool Stereo2DepthProcessor::finishedProcessing() {
  if (cameraMeasurementsQueue_.Size() == 0 && !isProcessing_) {
    return true;
  }
  else {
    return false;
  }
}

}
