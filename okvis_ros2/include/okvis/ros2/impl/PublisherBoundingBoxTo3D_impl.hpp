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

#ifndef INCLUDE_OKVIS_ROS2_PUBLISHER_BOUNDINGBOXTO3D_IMPL_HPP_
#define INCLUDE_OKVIS_ROS2_PUBLISHER_BOUNDINGBOXTO3D_IMPL_HPP_

#include <okvis/ros2/Publisher.hpp>
#include <se/map/raycaster.hpp>
#include <se/map/impl/raycaster_impl.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <glog/logging.h>
#include <shared_mutex>
#include <vector>
#include <algorithm>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

namespace okvis {

template<typename ServiceT>
void Publisher::handleBoundingBoxTo3D(
    const std::shared_ptr<typename ServiceT::Request> request,
    std::shared_ptr<typename ServiceT::Response> response)
{
  response->success = false;
  response->distance = 0.0f;
  response->error_message = "";
  
  // Check if camera system is available
  if (!nCameraSystem_) {
    response->error_message = "Camera system not initialized";
    LOG(ERROR) << response->error_message;
    return;
  }
  
  // Extract detection target information
  const auto& timestamp_detection_target = request->header.stamp;  
  const auto& frame_id_detection_target = request->header.frame_id;
  const uint32_t camera_id = request->camera_id;
  const float bbox_cx = request->detection_target.cx;  // Top-left corner x
  const float bbox_cy = request->detection_target.cy;  // Top-left corner y
  const float bbox_width = request->detection_target.width;
  const float bbox_height = request->detection_target.height;
  
  // Check if camera ID is valid
  if (camera_id >= nCameraSystem_->numCameras()) {
    response->error_message = "Invalid camera ID: " + std::to_string(camera_id) + 
                             " (max: " + std::to_string(nCameraSystem_->numCameras() - 1) + ")";
    LOG(ERROR) << response->error_message;
    return;
  }
  
  // Convert top-left corner to center coordinates
  const float bbox_center_x = bbox_cx + bbox_width / 2.0f;
  const float bbox_center_y = bbox_cy + bbox_height / 2.0f;
  
  // Get state at requested timestamp from detection target odometry
  State currentState;
  bool stateFound = false;
  

  
  // Check if timestamp is provided (non-zero means a specific time is requested)
  if (timestamp_detection_target.sec != 0 || timestamp_detection_target.nanosec != 0) {
    // Query trajectory for state closest to requested timestamp
    okvis::Time requestedTime(timestamp_detection_target.sec, timestamp_detection_target.nanosec);
    {
      std::shared_lock<std::shared_mutex> lock(stateMutex_);
      stateFound = trajectory_.getState(requestedTime, currentState);
    }
    
    if (!stateFound) {
      response->error_message = "No state available at requested timestamp: " + 
                                std::to_string(timestamp_detection_target.sec) + "." + 
                                std::to_string(timestamp_detection_target.nanosec) +
                                " (frame_id: " + frame_id_detection_target + ")";
      LOG(ERROR) << response->error_message;
      return;
    }
    LOG(INFO) << "Using historical state at timestamp: " << requestedTime 
              << " (frame_id: " << frame_id_detection_target << ")";
  } else {
    // Use current state (backward compatibility)
    {
      std::shared_lock<std::shared_mutex> lock(stateMutex_);
      if (!currentState_.has_value()) {
        response->error_message = "No current state available";
        LOG(ERROR) << response->error_message;
        return;
      }
      currentState = currentState_.value();
      stateFound = true;
    }
    LOG(INFO) << "Using current state (requested frame_id: " << frame_id_detection_target << ")";
  } 
  
  // Get camera geometry
  std::shared_ptr<const okvis::cameras::CameraBase> camera;
  if (nCameraSystem_->cameraType(camera_id).depthType.needRectify) {
    camera = nCameraSystem_->rectifyCameraGeometry(camera_id);
  } else {
    camera = nCameraSystem_->cameraGeometry(camera_id);
  }
  
  if (!camera) {
    response->error_message = "Camera geometry not available for camera ID: " + std::to_string(camera_id);
    LOG(ERROR) << response->error_message;
    return;
  }
  
  // Get camera extrinsics T_SC (sensor to camera)
  okvis::kinematics::Transformation T_SC;
  if (nCameraSystem_->cameraType(camera_id).depthType.needRectify) {
    T_SC = *nCameraSystem_->rectifyT_SC(camera_id);
  } else {
    T_SC = *nCameraSystem_->T_SC(camera_id);
  }
  
  // Compute camera pose in world frame: T_WC = T_WS * T_SC
  const okvis::kinematics::Transformation T_WS = currentState.T_WS;
  const okvis::kinematics::Transformation T_WC = T_WS * T_SC;
  const Eigen::Vector3d camera_origin_W = T_WC.r();
  const Eigen::Matrix3d C_WC = T_WC.C();
  
  // Backproject bounding box center to get ray direction in camera frame
  // Note: bbox_center_x and bbox_center_y are already converted from top-left to center
  Eigen::Vector2d imagePoint(bbox_center_x, bbox_center_y);
  Eigen::Vector3d ray_dir_C;
  if (!camera->backProject(imagePoint, &ray_dir_C)) {
    response->error_message = "Failed to backproject image point at (" + 
                              std::to_string(bbox_center_x) + ", " + 
                              std::to_string(bbox_center_y) + ")";
    LOG(ERROR) << response->error_message;
    return;
  }
  
  // Normalize ray direction
  ray_dir_C.normalize();
  
  // Transform ray direction to world frame
  Eigen::Vector3f ray_dir_W = (C_WC * ray_dir_C).cast<float>();
  ray_dir_W.normalize();
  
  // Ray origin in world frame
  Eigen::Vector3f ray_origin_W = camera_origin_W.cast<float>();
  
  // Get submaps (thread-safe, read lock - allows concurrent readers)
  // Note: We copy the shared_ptrs which is cheap (just reference counting)
  std::unordered_map<uint64_t, okvis::kinematics::Transformation> submapPoses;
  std::unordered_map<uint64_t, std::shared_ptr<okvis::SupereightMapType>> submaps;
  {
    std::shared_lock<std::shared_mutex> lock(submapDataMutex_);
    if (submapLookup_.empty()) {
      response->error_message = "No submaps available";
      LOG(ERROR) << response->error_message;
      return;
    }
    submapPoses = submapPoseLookup_;
    submaps = submapLookup_;
  }
  
  // Use the camera origin (ray origin) as the detection position for submap selection
  // This is the camera position at the detection timestamp
  const Eigen::Vector3d detection_pos_W = camera_origin_W;
  
  // Find closest submaps without sorting all of them
  struct SubmapDistance {
    uint64_t id;
    float distance;
    okvis::kinematics::Transformation T_WK;
    std::shared_ptr<okvis::SupereightMapType> submap;
  };
  
  const size_t max_submaps_to_check = 3;  // Only check closest 3 submaps
  std::vector<SubmapDistance> closest_submaps;
  closest_submaps.reserve(max_submaps_to_check);
  
  // Single pass through all submaps to find the closest ones
  for (const auto& [submap_id, submap] : submaps) {
    if (!submap) {
      continue;
    }
    
    auto pose_it = submapPoses.find(submap_id);
    if (pose_it == submapPoses.end()) {
      continue;
    }
    
    // Calculate distance from detection pose to submap pose
    const okvis::kinematics::Transformation& T_WK = pose_it->second;
    const Eigen::Vector3d submap_pos_W = T_WK.r();
    const float distance = (detection_pos_W - submap_pos_W).norm();
    
    // Insert into closest_submaps if it's closer than the farthest one we have
    if (closest_submaps.size() < max_submaps_to_check) {
      // Still building the list
      closest_submaps.push_back({submap_id, distance, T_WK, submap});
      // Keep sorted (insertion sort for small list)
      std::sort(closest_submaps.begin(), closest_submaps.end(),
                [](const SubmapDistance& a, const SubmapDistance& b) {
                  return a.distance < b.distance;
                });
    } else if (distance < closest_submaps.back().distance) {
      // Replace the farthest one if this is closer
      closest_submaps.back() = {submap_id, distance, T_WK, submap};
      // Re-sort (only 3 elements, very fast)
      std::sort(closest_submaps.begin(), closest_submaps.end(),
                [](const SubmapDistance& a, const SubmapDistance& b) {
                  return a.distance < b.distance;
                });
    }
  }
  
  // Raycast parameters
  const float t_near = 0.1f;  // Start 10cm from camera
  const float t_far = 100.0f;  // Maximum 100m range
  const float mu = 0.0f;  // TSDF truncation distance (0 for surface)
  const float step = 0.1f;  // Step size for raycasting (1cm)
  const float largestep = 0.5f;  // Large step size
  
  // Try raycasting through closest submaps (only the ones we found)
  std::optional<Eigen::Vector4f> best_intersection;
  float best_distance = std::numeric_limits<float>::max();
  bool found_intersection = false;
  
  for (const auto& submap_info : closest_submaps) {
    const uint64_t submap_id = submap_info.id;
    const auto& submap = submap_info.submap;
    const okvis::kinematics::Transformation& T_WK = submap_info.T_WK;
    
    // Transform ray to submap frame K
    const okvis::kinematics::Transformation T_KW = T_WK.inverse();
    // Convert to Vector3d first, then transform, then cast back to float
    const Eigen::Vector3d ray_origin_W_d = ray_origin_W.cast<double>();
    const Eigen::Vector3d ray_origin_K_d = T_KW * ray_origin_W_d;
    const Eigen::Vector3f ray_origin_K = ray_origin_K_d.cast<float>();
    const Eigen::Matrix3f C_KW = T_KW.C().cast<float>();
    const Eigen::Vector3f ray_dir_K = C_KW * ray_dir_W;
    
    // Raycast in submap frame (expensive operation - only do for closest submaps)
    // Note: For occupancy maps, the implementation only uses t_near and t_far (ignores mu, step, largestep)
    // The implementation header defines a 6-parameter version, so we call that directly
    using MapType = std::remove_reference_t<decltype(*submap)>;
    auto intersection = se::raycaster::raycast<MapType>(
        *submap,
        submap->getOctree(),
        ray_origin_K,
        ray_dir_K,
        t_near,
        t_far
    );
    
    if (intersection.has_value()) {
      // intersection is [x, y, z, distance] in submap frame K
      // Use template keyword for dependent name in template context
      const Eigen::Vector3f point_K_f = intersection->template head<3>();
      
      // Transform back to world frame (convert to Vector3d first)
      const Eigen::Vector3d point_K_d = point_K_f.cast<double>();
      const Eigen::Vector3d point_W_d = T_WK * point_K_d;
      const Eigen::Vector3f point_W = point_W_d.cast<float>();
      const float distance_W = (point_W - ray_origin_W).norm();
      
      // Keep the closest intersection
      if (distance_W < best_distance) {
        best_distance = distance_W;
        best_intersection = Eigen::Vector4f(point_W.x(), point_W.y(), point_W.z(), distance_W);
        found_intersection = true;
      }
    }
  }
  
  // If no intersection found, return position 10m along the ray direction
  if (!found_intersection) {
    const float fallback_distance = 10.0f;  // 10 meters
    const Eigen::Vector3f fallback_position = ray_origin_W + ray_dir_W * fallback_distance;
    best_intersection = Eigen::Vector4f(
      fallback_position.x(),
      fallback_position.y(),
      fallback_position.z(),
      fallback_distance
    );
    LOG(WARNING) << "No intersection found in closest " << max_submaps_to_check 
                 << " submaps. Returning fallback position 10m along ray.";
  }
  
  // Return result (best_intersection is always set now - either from raycast or fallback)
  response->position.x = (*best_intersection)[0];
  response->position.y = (*best_intersection)[1];
  response->position.z = (*best_intersection)[2];
  response->distance = (*best_intersection)[3];
  
  // Set response header with frame_id (use world frame for 3D position)
  // Echo back the timestamp from the detection target
  response->header.stamp = timestamp_detection_target;
  response->header.frame_id = "world";  // 3D position is in world frame
  
  // Set success based on whether we found actual intersection or used fallback
  if (found_intersection) {
    response->success = true;
    LOG(INFO) << "Raycast successful: 3D position [" << response->position.x << ", " 
              << response->position.y << ", " << response->position.z << "] at distance " 
              << response->distance << "m (frame_id: " << response->header.frame_id << ")";
  } else {
    response->success = false;
    response->error_message = "No intersection found in closest submaps. Returning fallback position 10m along ray.";
    LOG(WARNING) << response->error_message;
  }
  
  // Publish visualization markers for debugging
  {
    auto marker_array = std::make_shared<visualization_msgs::msg::MarkerArray>();
    int marker_id = 0;
    
    // Get current time for markers
    rclcpp::Time now = node_->now();
    
    // 1. Camera pose marker (sphere)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = now;
      marker.ns = "raycast_debug";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = camera_origin_W.x();
      marker.pose.position.y = camera_origin_W.y();
      marker.pose.position.z = camera_origin_W.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.2;
      marker.scale.y = 0.2;
      marker.scale.z = 0.2;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
      marker_array->markers.push_back(marker);
    }
    
    // 2. Ray direction arrow (from camera to 20m along ray)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = now;
      marker.ns = "raycast_debug";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      geometry_msgs::msg::Point start, end;
      start.x = ray_origin_W.x();
      start.y = ray_origin_W.y();
      start.z = ray_origin_W.z();
      const float ray_length = 20.0f;
      end.x = ray_origin_W.x() + ray_dir_W.x() * ray_length;
      end.y = ray_origin_W.y() + ray_dir_W.y() * ray_length;
      end.z = ray_origin_W.z() + ray_dir_W.z() * ray_length;
      marker.points.push_back(start);
      marker.points.push_back(end);
      marker.scale.x = 0.1;  // shaft diameter
      marker.scale.y = 0.15; // head diameter
      marker.scale.z = 0.2;  // head length
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
      marker_array->markers.push_back(marker);
    }
    
    // 3. All available submaps (gray spheres)
    {
      for (const auto& [submap_id, submap_pose] : submapPoses) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = now;
        marker.ns = "raycast_debug";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        const Eigen::Vector3d submap_pos = submap_pose.r();
        marker.pose.position.x = submap_pos.x();
        marker.pose.position.y = submap_pos.y();
        marker.pose.position.z = submap_pos.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        marker.color.a = 0.5;
        marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
        marker_array->markers.push_back(marker);
        
        // Text label with submap ID
        visualization_msgs::msg::Marker text_marker;
        text_marker.header.frame_id = "world";
        text_marker.header.stamp = now;
        text_marker.ns = "raycast_debug";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose.position.x = submap_pos.x();
        text_marker.pose.position.y = submap_pos.y();
        text_marker.pose.position.z = submap_pos.z() + 1.0;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = 0.3;
        text_marker.color.r = 0.5;
        text_marker.color.g = 0.5;
        text_marker.color.b = 0.5;
        text_marker.color.a = 1.0;
        text_marker.text = "S" + std::to_string(submap_id);
        text_marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
        marker_array->markers.push_back(text_marker);
      }
    }
    
    // 4. Checked submaps (highlighted in blue)
    {
      for (const auto& submap_info : closest_submaps) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = now;
        marker.ns = "raycast_debug";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        const Eigen::Vector3d submap_pos = submap_info.T_WK.r();
        marker.pose.position.x = submap_pos.x();
        marker.pose.position.y = submap_pos.y();
        marker.pose.position.z = submap_pos.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 0.8;
        marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
        marker_array->markers.push_back(marker);
        
        // Text label with distance
        visualization_msgs::msg::Marker text_marker;
        text_marker.header.frame_id = "world";
        text_marker.header.stamp = now;
        text_marker.ns = "raycast_debug";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose.position.x = submap_pos.x();
        text_marker.pose.position.y = submap_pos.y();
        text_marker.pose.position.z = submap_pos.z() + 1.5;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = 0.4;
        text_marker.color.r = 0.0;
        text_marker.color.g = 0.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.text = "S" + std::to_string(submap_info.id) + " (d=" + 
                          std::to_string(submap_info.distance) + "m)";
        text_marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
        marker_array->markers.push_back(text_marker);
      }
    }
    
    // 5. Result position (intersection or fallback)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = now;
      marker.ns = "raycast_debug";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = response->position.x;
      marker.pose.position.y = response->position.y;
      marker.pose.position.z = response->position.z;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.3;
      marker.scale.y = 0.3;
      marker.scale.z = 0.3;
      if (found_intersection) {
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;  // Green for intersection
      } else {
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;  // Yellow for fallback
      }
      marker.color.a = 1.0;
      marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
      marker_array->markers.push_back(marker);
    }
    
    // 6. Detection position (where the detection was made)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = now;
      marker.ns = "raycast_debug";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = detection_pos_W.x();
      marker.pose.position.y = detection_pos_W.y();
      marker.pose.position.z = detection_pos_W.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.15;
      marker.scale.y = 0.15;
      marker.scale.z = 0.15;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;  // Magenta for detection position
      marker.color.a = 1.0;
      marker.lifetime = rclcpp::Duration(std::chrono::seconds(10));
      marker_array->markers.push_back(marker);
    }
    
    // Publish the marker array
    pubRaycastDebug_.publish(marker_array);
    LOG(INFO) << "Published raycast debug visualization with " << marker_array->markers.size() << " markers";
  }
}

template<typename ServiceT>
void Publisher::registerBoundingBoxTo3DService(const std::string& service_name)
{
  if (!node_) {
    LOG(ERROR) << "Cannot register service: node not initialized";
    return;
  }
  
  boundingBoxTo3DService_ = node_->create_service<ServiceT>(
    service_name,
    std::bind(&Publisher::handleBoundingBoxTo3D<ServiceT>, this,
              std::placeholders::_1, std::placeholders::_2)
  );
  
  LOG(INFO) << "Registered bounding box to 3D service: " << service_name;
}

}  // namespace okvis

#endif /* INCLUDE_OKVIS_ROS2_PUBLISHER_BOUNDINGBOXTO3D_IMPL_HPP_ */

