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
 * @file Publisher.cpp
 * @brief Source file for the Publisher class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>
#include <okvis/ros2/Publisher.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/timing/Timer.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Default constructor.
Publisher::Publisher(
  std::shared_ptr<rclcpp::Node> node,
  std::shared_ptr<ThreadedPublisher> threadedOdometryPublisher,
  std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
  std::shared_ptr<ThreadedPublisher> threadedPublisher)
    : threadedOdometryPublisher_(threadedOdometryPublisher),
      threadedImagePublisher_(threadedImagePublisher),
      threadedPublisher_(threadedPublisher),
      trajectoryOutput_(false),
      trajectoryLocked_(false)
{
  setupNode(node);
}

void Publisher::setupNode(std::shared_ptr<rclcpp::Node> node)
{
  // set up node
  node_ = node;

  // set up publishers
  slice_pub_ =              threadedPublisher_->registerPublisher<visualization_msgs::msg::Marker>("se_map_slice");
  pubObometry_ =            threadedOdometryPublisher_->registerPublisher<nav_msgs::msg::Odometry>("okvis_odometry");
  pubPath_ =                threadedPublisher_->registerPublisher<visualization_msgs::msg::Marker>("okvis_path");
  pubTransform_ =           threadedPublisher_->registerPublisher<geometry_msgs::msg::TransformStamped>("okvis_transform");
  pubMesh_ =                threadedPublisher_->registerPublisher<visualization_msgs::msg::Marker>("okvis_mesh");
  pubSubmapMesh_ =          threadedPublisher_->registerPublisher<visualization_msgs::msg::MarkerArray>("okvis_submap_mesh");
  pubPointsMatched_ =       threadedPublisher_->registerPublisher<sensor_msgs::msg::PointCloud2>("okvis_points_matched");
  pubPointsAlignment_ =     threadedPublisher_->registerPublisher<sensor_msgs::msg::PointCloud2>("okvis_points_alignment");
  pubOccupancyGrid_ =       threadedPublisher_->registerPublisher<nav_msgs::msg::OccupancyGrid>("okvis_occupancy_grid");
  pubSubmapBounds_ =        threadedPublisher_->registerPublisher<visualization_msgs::msg::MarkerArray>("okvis_submap_bounds");
  
  // Initialize global occupancy grid (100m x 100m centered at world origin)
  globalGridWidth_ = 2000;   // 100m / 0.05m
  globalGridHeight_ = 2000;
  globalGridOrigin_ = Eigen::Vector2f(-50.0f, -50.0f);  // Centered at (0,0)
  globalOccupancyGrid_.resize(globalGridWidth_ * globalGridHeight_, -1);  // All unknown
  LOG(INFO) << "Initialized global occupancy grid: " << globalGridWidth_ << "x" << globalGridHeight_ 
            << " @ 0.05m resolution, origin: [" << globalGridOrigin_.x() << "," << globalGridOrigin_.y() << "]";
      
  // get the mesh, if there is one
  // where to get the mesh from
  std::string mesh_file;
  bool loaded_mesh;
  node_->declare_parameter("mesh_file", "");
  loaded_mesh = node_->get_parameter("mesh_file", mesh_file);
  meshMsg_ = std::make_shared<visualization_msgs::msg::Marker>();
  if (loaded_mesh) {
    meshMsg_->mesh_resource = mesh_file;

    // fill orientation
    meshMsg_->pose.orientation.x = 0;
    meshMsg_->pose.orientation.y = 0;
    meshMsg_->pose.orientation.z = 0;
    meshMsg_->pose.orientation.w = 1;

    // fill position
    meshMsg_->pose.position.x = 0;
    meshMsg_->pose.position.y = 0;
    meshMsg_->pose.position.z = 0;

    // scale -- needed
    meshMsg_->scale.x = 1.0;
    meshMsg_->scale.y = 1.0;
    meshMsg_->scale.z = 1.0;

    meshMsg_->action = visualization_msgs::msg::Marker::ADD;
    meshMsg_->color.a = 1.0; // Don't forget to set the alpha!
    meshMsg_->color.r = 1.0;
    meshMsg_->color.g = 1.0;
    meshMsg_->color.b = 1.0;

    // embedded material / colour
    //meshMsg_->mesh_use_embedded_materials = true;
  } else {
    LOG(INFO) << "no mesh found for visualisation, set ros param mesh_file, if desired";
    meshMsg_->mesh_resource = "";
  }
}

Publisher::~Publisher()
{
}

void Publisher::setBodyTransform(const okvis::kinematics::Transformation& T_BS) {
  T_BS_ = T_BS;
  T_SB_ = T_BS_.inverse();
}

void Publisher::setCsvFile(const std::string & filename, bool rpg)
{
  trajectoryOutput_.setCsvFile(filename, rpg);
}

bool Publisher::realtimePredictAndPublish(const okvis::Time& stamp,
                         const Eigen::Vector3d& alpha,
                         const Eigen::Vector3d& omega) {

  // store in any case
  imuMeasurements_.push_back(ImuMeasurement(stamp, ImuSensorReadings(omega,alpha)));

  // add to Trajectory if possible
  bool success = false;
  State state;
  if(!trajectoryLocked_) {
    trajectoryLocked_ = true;
    for(auto & imuMeasurement : imuMeasurements_) {
      State propagatedState;
      if(trajectory_.addImuMeasurement(imuMeasurement.timeStamp,
                                       imuMeasurement.measurement.accelerometers,
                                       imuMeasurement.measurement.gyroscopes,
                                       propagatedState)) {
        state = propagatedState;
        success = true;
      }
    }
    imuMeasurements_.clear();
    trajectoryLocked_ = false;
    if(!success) {
      return false;
    }
  } else {
    return false;
  }

  // only publish according to rate
  if((state.timestamp - lastTime_).toSec()
      < (1.0/double(odometryPublishingRate_))) {
    return false;
  }
  lastTime_ = state.timestamp;

  rclcpp::Time t(state.timestamp.sec, state.timestamp.nsec); // Header timestamp.
  const okvis::kinematics::Transformation T_WS = state.T_WS;
  const kinematics::Transformation T_SB = T_BS_.inverse();
  const okvis::kinematics::Transformation T_WB = T_WS * T_SB;

  // Odometry
  auto odometryMsg = std::make_shared<nav_msgs::msg::Odometry>();  // Odometry message.
  odometryMsg->header.frame_id = "world";
  odometryMsg->child_frame_id = "body";
  odometryMsg->header.stamp = t;

  // fill orientation
  const Eigen::Quaterniond q = T_WB.q();
  odometryMsg->pose.pose.orientation.x = q.x();
  odometryMsg->pose.pose.orientation.y = q.y();
  odometryMsg->pose.pose.orientation.z = q.z();
  odometryMsg->pose.pose.orientation.w = q.w();

  // fill position
  const Eigen::Vector3d r = T_WB.r();
  odometryMsg->pose.pose.position.x = r[0];
  odometryMsg->pose.pose.position.y = r[1];
  odometryMsg->pose.pose.position.z = r[2];

  // note: velocity and angular velocity needs to be expressed in child frame, i.e. "body"
  // see http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html

  // fill velocity
  const kinematics::Transformation T_BW = T_WB.inverse();
  const Eigen::Vector3d v = T_BW.C() * state.v_W + T_BS_.C() * T_SB.r().cross(state.omega_S);
  // ...of body orig. represented in body
  odometryMsg->twist.twist.linear.x = v[0];
  odometryMsg->twist.twist.linear.y = v[1];
  odometryMsg->twist.twist.linear.z = v[2];

  // fill angular velocity
  const Eigen::Matrix3d C_BS = T_BS_.C();
  const Eigen::Vector3d omega_B = C_BS * state.omega_S; // of body represented in body
  odometryMsg->twist.twist.angular.x = omega_B[0];
  odometryMsg->twist.twist.angular.y = omega_B[1];
  odometryMsg->twist.twist.angular.z = omega_B[2];

  // publish odometry
  pubObometry_.publish(odometryMsg);
  return true;
}

void Publisher::publishEstimatorUpdate(
  const State& state, const TrackingState & trackingState,
  std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
  std::shared_ptr<const MapPointVector> landmarks) {

  // forward to existing writer
  trajectoryOutput_.processState(state, trackingState, updatedStates, landmarks);

  // pose
  rclcpp::Time t(state.timestamp.sec, state.timestamp.nsec); // Header timestamp.
  const okvis::kinematics::Transformation T_WS = state.T_WS;
  const kinematics::Transformation T_SB = T_BS_.inverse();
  const okvis::kinematics::Transformation T_WB = T_WS * T_SB;

  // publish pose:
  auto poseMsg = std::make_shared<geometry_msgs::msg::TransformStamped>(); // Pose message.
  poseMsg->child_frame_id = "body";
  poseMsg->header.frame_id = "world";
  poseMsg->header.stamp = t;
  //if ((node_->now() - t).seconds() > 10.0)
  //  poseMsg->header.stamp = node_->now(); // hack for dataset processing

  // fill orientation
  Eigen::Quaterniond q = T_WB.q();
  poseMsg->transform.rotation.x = q.x();
  poseMsg->transform.rotation.y = q.y();
  poseMsg->transform.rotation.z = q.z();
  poseMsg->transform.rotation.w = q.w();

  // fill position
  Eigen::Vector3d r = T_WB.r();
  poseMsg->transform.translation.x = r[0];
  poseMsg->transform.translation.y = r[1];
  poseMsg->transform.translation.z = r[2];

  // publish current pose
  pubTransform_.publish(poseMsg);

  // also do the mesh
  meshMsg_->header.frame_id = "world";
  meshMsg_->header.stamp = t;
  //if ((node_->now() - t).seconds() > 10.0)
  //  meshMsg_->header.stamp = node_->now(); // hack for dataset processing
  meshMsg_->type = visualization_msgs::msg::Marker::MESH_RESOURCE;
  
  meshMsg_->pose.position.x = T_WS.r().x();
  meshMsg_->pose.position.y = T_WS.r().y();
  meshMsg_->pose.position.z = T_WS.r().z();

  meshMsg_->pose.orientation.x = T_WS.q().x();
  meshMsg_->pose.orientation.y = T_WS.q().y();
  meshMsg_->pose.orientation.z = T_WS.q().z();
  meshMsg_->pose.orientation.w = T_WS.q().w();

  // publish mesh
  if(!meshMsg_->mesh_resource.empty())
    pubMesh_.publish(meshMsg_);  //publish stamped mesh

  // check if empty
  if(updatedStates->empty()) {
    return; // otherwise it will override landmarks and path (plus create unnecessary traffic)
  }

  // now handle all the (key)frames & path:
  for(const auto & updatedState : *updatedStates) {
    rclcpp::Time updatedState_t(state.timestamp.sec, state.timestamp.nsec); // Header timestamp.
    const okvis::kinematics::Transformation T_WS = updatedState.second.T_WS;

    auto updatedStatePoseMsg = std::make_shared<geometry_msgs::msg::TransformStamped>(); // Pose message.
    const okvis::kinematics::Transformation T_WB = T_WS * T_SB;
    updatedStatePoseMsg->child_frame_id = "body_"+std::to_string(updatedState.second.id.value());
    updatedStatePoseMsg->header.frame_id = "world";
    updatedStatePoseMsg->header.stamp = updatedState_t;
    updatedStatePoseMsg->transform.rotation.x = T_WB.q().x();
    updatedStatePoseMsg->transform.rotation.y = T_WB.q().y();
    updatedStatePoseMsg->transform.rotation.z = T_WB.q().z();
    updatedStatePoseMsg->transform.rotation.w = T_WB.q().w();
    updatedStatePoseMsg->transform.translation.x = T_WB.r()[0];
    updatedStatePoseMsg->transform.translation.y = T_WB.r()[1];
    updatedStatePoseMsg->transform.translation.z = T_WB.r()[2];

    // publish
    pubTransform_.publish(updatedStatePoseMsg);
  }

  // update Trajectory object
  std::set<okvis::StateId> affectedStateIds;
  while(trajectoryLocked_);
  trajectoryLocked_ = true;
  trajectory_.update(trackingState, updatedStates, affectedStateIds);
  trajectoryLocked_ = false;

  // now for the path (in segments of 1000 state IDs)
  auto path = std::make_shared<visualization_msgs::msg::Marker>(); // The path message.
  path->header.stamp = t;
  path->header.frame_id = "world";
  path->ns = "okvis_path";
  path->type = visualization_msgs::msg::Marker::LINE_STRIP; // Type of object
  path->action = 0; // 0 add/modify an object, 1 (dprcd), 2 deletes an object, 3 deletes all objects
  path->pose.position.x = 0.0;
  path->pose.position.y = 0.0;
  path->pose.position.z = 0.0;
  path->pose.orientation.x = 0.0;
  path->pose.orientation.y = 0.0;
  path->pose.orientation.z = 0.0;
  path->pose.orientation.w = 1.0;
  path->color.r = 0.8;
  path->color.g = 0.8;
  path->color.b = 0.0;
  path->color.a = 1.0;
  path->scale.x = 0.015;
  path->lifetime = rclcpp::Duration::from_seconds(0); // 0 for infinity

  // publish paths as batches of 1000 points.
  uint64_t firstId = affectedStateIds.begin()->value();
  uint64_t id = (firstId/1000)*1000;
  if(id==0) {
    id = 1;
  }
  path->id = (id/1000)*1000;
  if(path->id != 0) {
    // link to previous path segment
    okvis::State state;
    if(trajectory_.getState(okvis::StateId(path->id-1), state)) {
      const okvis::kinematics::Transformation T_WB_p = state.T_WS * T_SB;
      const Eigen::Vector3d& r = T_WB_p.r();
      geometry_msgs::msg::Point point;
      point.x = r[0];
      point.y = r[1];
      point.z = r[2];
      path->points.push_back(point);
    }
  }
  const uint64_t latestId = updatedStates->rbegin()->first.value();
  while(id <= latestId) {
    uint64_t roundedId = (id/1000)*1000;
    if(path->id != int64_t(roundedId)) {
      pubPath_.publish(path); // first publish finished segment
      path->id = roundedId;
      // ..object ID useful in conjunction with namespace for manipulating&deleting the object later
      geometry_msgs::msg::Point lastPoint =  path->points.back(); // save last point
      path->points.clear(); // start new segment
      path->points.push_back(lastPoint);
    }
    okvis::State state;
    if(!trajectory_.getState(okvis::StateId(id), state)) continue;
    const okvis::kinematics::Transformation T_WB_p = state.T_WS * T_SB;
    const Eigen::Vector3d& r = T_WB_p.r();
    geometry_msgs::msg::Point point;
    point.x = r[0];
    point.y = r[1];
    point.z = r[2];
    path->points.push_back(point);
    ++id;
  }
  pubPath_.publish(path); // publish last segment

  // finally the landmarks
  pcl::PointCloud<pcl::PointXYZRGB> pointsMatched; // Point cloud for matched points.
  pointsMatched.reserve(landmarks->size());

  // transform points into custom world frame:
  /// \todo properly from ros params -- also landmark thresholds below
  for (const auto & lm : *landmarks) {
    // check infinity
    if (fabs((double) (lm.point[3])) < 1.0e-8)
      continue;

    // check quality
    if (lm.quality < 0.01)
      continue;

    pointsMatched.push_back(pcl::PointXYZRGB());
    const Eigen::Vector4d point = lm.point;
    pointsMatched.back().x = point[0] / point[3];
    pointsMatched.back().y = point[1] / point[3];
    pointsMatched.back().z = point[2] / point[3];
    pointsMatched.back().g = 255 * (std::min(0.1f, (float)lm.quality) / 0.1f);
  }
  pointsMatched.header.frame_id = "world";
  auto pointsMatchedMsg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(pointsMatched, *pointsMatchedMsg);
  pointsMatchedMsg->header.frame_id = "world";

#if PCL_VERSION >= PCL_VERSION_CALC(1,7,0)
  std_msgs::msg::Header header;
  header.stamp = t;
  pointsMatched.header.stamp = pcl_conversions::toPCL(header).stamp;
#else
  pointsMatched.header.stamp=_t;
#endif

  // and now publish them:
  pubPointsMatched_.publish(pointsMatchedMsg);

}

void Publisher::setupImageTopics(const okvis::cameras::NCameraSystem & nCameraSystem) {
  pubImages_.clear();
  for(size_t i=0; i< nCameraSystem.numCameras(); ++i) {
    std::string name;
    if(nCameraSystem.cameraType(i).isColour) {
      name = "rgb"+std::to_string(i);
    } else {
      name = "cam"+std::to_string(i);
    }
    pubImages_[name] = threadedImagePublisher_->registerPublisher<sensor_msgs::msg::Image>(name+"_matches");
  }
  std::string name = "Top Debug View";
  pubImages_[name] = threadedImagePublisher_->registerPublisher<sensor_msgs::msg::Image>("top_debug_view");
}

void Publisher::setMeshesPath(std::string meshesDir){
  meshesDir_ = meshesDir;
}

bool Publisher::publishImages(const std::map<std::string, cv::Mat>& images) const {
  for(const auto & image : images) {
    sensor_msgs::msg::Image::SharedPtr msg
      = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image.second).toImageMsg();
    const auto & pubIter = pubImages_.find(image.first);
    if(pubIter == pubImages_.end()) {
      continue;
    }
    pubIter->second.publish(msg);
  }
  return true;
}

void Publisher::setupNetworkTopics(const std::string & topicName) {
  // TODO: properly set topic names
  std::string name = topicName + "Depth";
  pubImages_[name] = threadedImagePublisher_->registerPublisher<sensor_msgs::msg::Image>("network_depth");

  name = topicName + "Sigma";
  pubImages_[name] = threadedImagePublisher_->registerPublisher<sensor_msgs::msg::Image>("network_sigma");
}

void Publisher::publishSubmapsAsCallback(std::unordered_map<uint64_t, okvis::kinematics::Transformation, std::hash<uint64_t>, std::equal_to<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, okvis::kinematics::Transformation>>> submapPoseLookup,
                                         std::unordered_map<uint64_t, std::shared_ptr<okvis::SupereightMapType>> submapLookup) 
{
  constexpr size_t n = okvis::SupereightMapType::SurfaceMesh::value_type::num_vertexes;

  std_msgs::msg::Header header;
  header.stamp = node_->now();
  header.frame_id = "world";

  // iterate over updated submaps and add them to our attribute holding the different submaps
  for (auto& it: submapLookup) {
    visualization_msgs::msg::Marker meshMarker;
    meshMarker.header = header;
    meshMarker.pose.orientation.w = 1;
    meshMarker.ns = "submaps"; 
    meshMarker.id = it.first;
    meshMarker.scale.x = 1;
    meshMarker.scale.y = 1;
    meshMarker.scale.z = 1;
    meshMarker.color.r = 0.8; 
    meshMarker.color.g = 0.8;
    meshMarker.color.b = 0.8;
    meshMarker.color.a = 1;
    meshMarker.lifetime = rclcpp::Duration(std::chrono::seconds(0)); // 0 for infinity
    meshMarker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    meshMarker.action = visualization_msgs::msg::Marker::ADD;


    if(submapSurfaceMesh_.find(it.first) == submapSurfaceMesh_.end()){
      submapSurfaceMesh_[it.first] = se::algorithms::marching_cube(it.second->getOctree());
    }
    auto & mesh = submapSurfaceMesh_[it.first];

    meshMarker.points.clear();
    meshMarker.points.resize(n * mesh.size());

    Eigen::Matrix4f T_OW = submapPoseLookup[it.first].T().cast<float>();
    Eigen::Matrix4f T_WM_scale = it.second->getTWM().matrix();
    T_WM_scale.topLeftCorner<3, 3>() *= it.second->getRes();
    const Eigen::Matrix4f T_OM = T_OW * T_WM_scale;

    submapPoses_[it.first] = T_OM;
    size_t valMeshCnt = 0;

    #ifdef OKVIS_COLIDMAP
    meshMarker.colors.clear();
    meshMarker.colors.resize(n * mesh.size());
    for (size_t i = 0; i < mesh.size(); i++) {
      const Eigen::Vector3f v0_W = (T_OM * (mesh[i].vertexes[0].homogeneous())).template head<3>();
      const Eigen::Vector3f v1_W = (T_OM * (mesh[i].vertexes[1].homogeneous())).template head<3>();
      const Eigen::Vector3f v2_W = (T_OM * (mesh[i].vertexes[2].homogeneous())).template head<3>();
      const float triangle_max_z_W = Eigen::Vector3f(v0_W.z(), v1_W.z(), v2_W.z()).maxCoeff();
      if (triangle_max_z_W > mesh_cutoff_z_) {
        continue;
      }

      meshMarker.points.resize(n * mesh.size());
      tf::pointEigenToMsg(v0_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 0]);
      tf::pointEigenToMsg(v1_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 1]);
      tf::pointEigenToMsg(v2_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 2]);
      if constexpr(okvis::SupereightMapType::SurfaceMesh::value_type::col_ == se::Colour::On) { // Check for valid colours
        if(!mesh[i].colour.vertexes) {
          continue;
        }
        for(size_t v = 0; v < n; v++) {
          auto& msg_colour = meshMarker.colors[n * valMeshCnt + v];
          const auto& mesh_colour = mesh[i].colour.vertexes.value()[v];
          msg_colour.r = float(mesh_colour.r) / UINT8_MAX;
          msg_colour.g = float(mesh_colour.g) / UINT8_MAX;
          msg_colour.b = float(mesh_colour.b) / UINT8_MAX;
          msg_colour.a = 1.0;
        }
      }
      valMeshCnt ++;
    }
    meshMarker.points.resize(n * valMeshCnt);
    meshMarker.colors.resize(n * valMeshCnt);
    submapMeshLookup_rgb_[it.first] = meshMarker;
    #else
    for (size_t i = 0; i < mesh.size(); i++) {
      const Eigen::Vector3f v0_W = (T_OM * (mesh[i].vertexes[0].homogeneous())).template head<3>();
      const Eigen::Vector3f v1_W = (T_OM * (mesh[i].vertexes[1].homogeneous())).template head<3>();
      const Eigen::Vector3f v2_W = (T_OM * (mesh[i].vertexes[2].homogeneous())).template head<3>();
      const float triangle_max_z_W = Eigen::Vector3f(v0_W.z(), v1_W.z(), v2_W.z()).maxCoeff();
      if (triangle_max_z_W > mesh_cutoff_z_) {
        continue;
      }
  
      tf::pointEigenToMsg(v0_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 0]);
      tf::pointEigenToMsg(v1_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 1]);
      tf::pointEigenToMsg(v2_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 2]);
      valMeshCnt ++;
    }
    meshMarker.points.resize(n * valMeshCnt);
    submapMeshLookup_rgb_[it.first] = meshMarker;
    #endif
  }

  auto markerarraymsg_ = std::make_shared<visualization_msgs::msg::MarkerArray>();
  markerarraymsg_->markers.clear();
  //If too many submaps are to be updated at once (> 1GB) ROS considers it has gone out of sync and does not update.
  // To prevent this, we generate buffers of submaps and send them in smaller packets to bypass this issue
  int submap_publisher_buffer = 0;

  for(auto it: submapMeshLookup_rgb_){
    it.second.header = header;
    markerarraymsg_->markers.push_back(it.second);
    submap_publisher_buffer++;

    if(submap_publisher_buffer == 30){
      submap_publisher_buffer = 0;
      pubSubmapMesh_.publish(markerarraymsg_);
      markerarraymsg_->markers.clear();
    }
  }
  
  
  if(markerarraymsg_->markers.size() > 0) {
      pubSubmapMesh_.publish(markerarraymsg_);
  }
  
  return;
}

void Publisher::publishFieldSliceAsCallback(
    const State& latest_state,
    const AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>>& seSubmapLookup)
{
  // Get the latest submap.
  const auto it = std::max_element(seSubmapLookup.begin(),
                                   seSubmapLookup.end(),
                                   [](const auto& lhs, const auto& rhs) -> bool {
                                     return lhs.first < rhs.first;
                                   });
  if (it == seSubmapLookup.end()) {
    return;
  }
  const kinematics::Transformation T_WK(seSubmapLookup.at(it->first).T_WK.matrix().cast<double>());
  const kinematics::Transformation T_KW = T_WK.inverse();
  const Eigen::Matrix4f T_WK_e = T_WK.T().cast<float>();
  auto msg = std::make_shared<visualization_msgs::msg::Marker>();
  msg->header.stamp.sec = latest_state.timestamp.sec;
  msg->header.stamp.nanosec = latest_state.timestamp.nsec;
  msg->header.frame_id = "world";
  msg->ns = "field_slice";
  msg->id = 0;
  msg->type = visualization_msgs::msg::Marker::TRIANGLE_LIST; //CUBE_LIST
  msg->action = visualization_msgs::msg::Marker::ADD;
  msg->pose.orientation.w = 1;
  msg->scale.x = 1;
  msg->scale.y = 1;
  msg->scale.z = 1;
  const Eigen::Vector3d t_WB = (latest_state.T_WS * T_SB_).r();
  // The slice frame Sl is axis-aligned with the world frame W and has the same
  // origin {O} as the body frame B.
  const kinematics::Transformation T_WSl(t_WB, Eigen::Quaterniond(1, 0, 0, 0));
  Eigen::Matrix4f T_KO = (T_KW * T_WSl).T().cast<float>();
  se::QuadMesh<se::Colour::On, se::Id::Off> quad_mesh;
  quad_mesh.reserve(se::math::sq(it->second.map->getOctree().getSize()));
  const Eigen::Array3f aabb_min = it->second.map->aabb().min();
  const Eigen::Array3f aabb_max = it->second.map->aabb().max();
  const Eigen::Vector3f dim = it->second.map->getDim();
  float res = it->second.map->getRes();
  for (float y = -dim.y(); y < dim.y(); y += res) {
      for (float x = -dim.x(); x < dim.x(); x += res) {
          const Eigen::Vector3f point_O(x, y, 0);
          const Eigen::Vector3f point_W = (T_KO * point_O.homogeneous()).head<3>();
          if ((point_W.array() < aabb_min).any() || (point_W.array() > aabb_max).any()) {
              // Skip points outside the map AABB for speed.
              continue;
          }
          Eigen::Vector3i voxel_coord;
          it->second.map->pointToVoxel<se::Safe::Off>(point_W, voxel_coord);

          // Check occupancy to skip faces in unknown space
          se::field_t occ = se::get_field(se::visitor::getData(it->second.map->getOctree(), voxel_coord));
          se::RGB slice_face_color{0, 0, 0};
          if (occ > 0) {
            slice_face_color.r = 255;
          } else if (occ < -15.0f) {
            slice_face_color.g = 255;
          } else {
            continue;
          }

          quad_mesh.emplace_back();
          quad_mesh.back().colour.face = slice_face_color;
          quad_mesh.back().vertexes[0] = point_O;
          quad_mesh.back().vertexes[1] = point_O + Eigen::Vector3f(res, 0, 0);
          quad_mesh.back().vertexes[2] = point_O + Eigen::Vector3f(res, res, 0);
          quad_mesh.back().vertexes[3] = point_O + Eigen::Vector3f(0, res, 0);
          for (auto& v : quad_mesh.back().vertexes) {
              v = (T_KO * v.homogeneous()).template head<3>();
          }
      }
  }

  // Convert Mesh to triangle mesh
  auto mesh = se::quad_to_triangle_mesh(quad_mesh);
  constexpr int num_face_vertexes = 3;
  msg->points.reserve(num_face_vertexes * mesh.size());
  msg->colors.reserve(num_face_vertexes * mesh.size());
  for (const auto& face : mesh) {
    std_msgs::msg::ColorRGBA face_color;
    face_color.a = 1.0f;
    face_color.r = face.colour.face.value().r / 255.0f;
    face_color.g = face.colour.face.value().g / 255.0f;
    face_color.b = face.colour.face.value().b / 255.0f;
    for (size_t j = 0; j < num_face_vertexes; j++) {
      msg->points.emplace_back();
      const Eigen::Vector3f vertex_W = (T_WK_e * face.vertexes[j].homogeneous()).template head<3>();
      msg->points.back().x = vertex_W.x();
      msg->points.back().y = vertex_W.y();
      msg->points.back().z = vertex_W.z();
      msg->colors.push_back(face_color);
    }
  }
  slice_pub_.publish(msg);
}

bool Publisher::poseChanged(const Eigen::Isometry3f& pose1, 
                             const Eigen::Isometry3f& pose2,
                             float trans_thresh,
                             float rot_thresh) const
{
  // Check translation change
  Eigen::Vector3f trans_diff = pose1.translation() - pose2.translation();
  if (trans_diff.norm() > trans_thresh) {
    return true;
  }
  
  // Check rotation change (using angle-axis)
  Eigen::Matrix3f R_diff = pose1.rotation() * pose2.rotation().transpose();
  Eigen::AngleAxisf angle_axis(R_diff);
  if (std::abs(angle_axis.angle()) > rot_thresh) {
    return true;
  }
  
  return false;
}

void Publisher::extractSubmapOccupancyGrid(
    uint64_t submap_id,
    const se::Submap<okvis::SupereightMapType>& submap)
{
  LOG(INFO) << "Extracting occupancy grid from submap " << submap_id;
  
  // IMPORTANT: aabb() returns bounds in MAP FRAME K (not world frame!)
  const Eigen::Array3f aabb_min_K = submap.map->aabb().min();
  const Eigen::Array3f aabb_max_K = submap.map->aabb().max();
  
  // Get submap pose for transforming K -> W
  const Eigen::Isometry3f& T_WK = submap.T_WK;
  
  // Transform AABB bounds to world frame for grid sizing
  Eigen::Vector3f aabb_min_W_vec = T_WK * aabb_min_K.matrix();
  Eigen::Vector3f aabb_max_W_vec = T_WK * aabb_max_K.matrix();
  
  // Since rotation can change which corner is min/max, recalculate
  Eigen::Array3f aabb_min_W = aabb_min_W_vec.array().min(aabb_max_W_vec.array());
  Eigen::Array3f aabb_max_W = aabb_min_W_vec.array().max(aabb_max_W_vec.array());
  
  const Eigen::Array3f size = aabb_max_W - aabb_min_W;
  const float resolution = occupancy_grid_resolution_;
  const uint32_t width = static_cast<uint32_t>(std::ceil(size.x() / resolution));
  const uint32_t height = static_cast<uint32_t>(std::ceil(size.y() / resolution));
  
  LOG(INFO) << "  Submap bounds (map frame K): [" << aabb_min_K.x() << "," << aabb_min_K.y() << "," << aabb_min_K.z() 
            << "] to [" << aabb_max_K.x() << "," << aabb_max_K.y() << "," << aabb_max_K.z() << "]";
  LOG(INFO) << "  Submap bounds (world frame W): [" << aabb_min_W.x() << "," << aabb_min_W.y() << "," << aabb_min_W.z() 
            << "] to [" << aabb_max_W.x() << "," << aabb_max_W.y() << "," << aabb_max_W.z() << "]";
  LOG(INFO) << "  Grid size: " << width << "x" << height << " @ " << resolution << "m resolution";
  
  // Store grid with world frame origin and pose snapshot for loop closure
  SubmapOccupancyGrid grid;
  grid.origin_W = aabb_min_W;          // World frame origin at extraction time
  grid.T_WK_snapshot = T_WK;            // Save current pose for loop closure detection
  grid.resolution = resolution;
  grid.width = width;
  grid.height = height;
  grid.data.resize(width * height, -1); // Initialize to unknown
  
  size_t valid_cells = 0;
  size_t occupied_cells = 0;
  size_t free_cells = 0;
  size_t unknown_cells = 0;
  size_t near_surface_cells = 0; // Between -1 and +1
  
  // Track occupancy value distribution
  float min_occ = std::numeric_limits<float>::max();
  float max_occ = std::numeric_limits<float>::lowest();
  std::vector<float> sample_values; // Store first 20 valid values for inspection
  
  // Extract occupancy data at the configured height
  // Query using WORLD frame coordinates (pointToVoxel expects world frame)
  size_t total_queries = 0;
  for (uint32_t grid_y = 0; grid_y < height; ++grid_y) {
    for (uint32_t grid_x = 0; grid_x < width; ++grid_x) {
      total_queries++;
      const float world_x = aabb_min_W.x() + (grid_x + 0.5f) * resolution;
      const float world_y = aabb_min_W.y() + (grid_y + 0.5f) * resolution;
      const float world_z = occupancy_grid_height_;
      const Eigen::Vector3f point_W(world_x, world_y, world_z);
      
      // Query voxel (map handles world to voxel conversion internally)
      Eigen::Vector3i voxel_coord;
      submap.map->pointToVoxel<se::Safe::Off>(point_W, voxel_coord);
      auto data = se::visitor::getData(submap.map->getOctree(), voxel_coord);
      
      if (!se::is_valid(data)) {
        unknown_cells++;
        continue; // Leave as unknown
      }
      
      valid_cells++;
      se::field_t occ = se::get_field(data);
      
      // Track value range
      min_occ = std::min(min_occ, occ);
      max_occ = std::max(max_occ, occ);
      if (sample_values.size() < 20) {
        sample_values.push_back(occ);
      }
      
      // Convert to ROS format
      // For TSDF: negative = free space, near zero = surface (occupied), positive = behind surface
      int8_t grid_value = -1;
      
      // Surface/Occupied: values close to zero (±0.5 voxels from surface)
      if (std::abs(occ) < 0.5f) {
        // Very close to surface - definitely occupied
        grid_value = 100; // Fully occupied
        occupied_cells++;
      } else if (occ > 0.5f) {
        // Positive values behind surface - treat as occupied
        // Scale from 0.5 to 2.0 -> 75 to 100
        float normalized = std::min((occ - 0.5f) / 1.5f, 1.0f);
        grid_value = static_cast<int8_t>(75 + normalized * 25);
        occupied_cells++;
      } else if (occ < -3.0f) {
        // Very negative = definitely free space
        grid_value = 0; // Fully free
        free_cells++;
      } else if (occ < -0.5f) {
        // Moderately negative = likely free
        // Scale from -0.5 to -3.0 -> 25 to 0
        float normalized = std::max((occ + 0.5f) / -2.5f, 0.0f);
        grid_value = static_cast<int8_t>(25 - normalized * 25);
        free_cells++;
      } else {
        // Fallback for edge cases
        near_surface_cells++;
      }
      
      const size_t index = grid_y * width + grid_x;
      grid.data[index] = grid_value;
    }
  }
  
  LOG(INFO) << "  Extraction stats:";
  LOG(INFO) << "    Total queries: " << total_queries;
  LOG(INFO) << "    Valid cells: " << valid_cells << " (" 
            << (100.0f * valid_cells / total_queries) << "%)";
  LOG(INFO) << "    Unknown (invalid): " << unknown_cells << " (" 
            << (100.0f * unknown_cells / total_queries) << "%)";
  LOG(INFO) << "    Occupied (near 0 or positive): " << occupied_cells << " (" 
            << (100.0f * occupied_cells / total_queries) << "%)";
  LOG(INFO) << "    Free (negative < -0.5): " << free_cells << " (" 
            << (100.0f * free_cells / total_queries) << "%)";
  LOG(INFO) << "    Unclassified: " << near_surface_cells << " (" 
            << (100.0f * near_surface_cells / total_queries) << "%)";
  
  if (valid_cells > 0) {
    LOG(INFO) << "    Occupancy value range: [" << min_occ << ", " << max_occ << "]";
    LOG(INFO) << "    Sample values: ";
    std::stringstream ss;
    for (size_t i = 0; i < sample_values.size(); ++i) {
      ss << sample_values[i];
      if (i < sample_values.size() - 1) ss << ", ";
    }
    LOG(INFO) << "      " << ss.str();
  }
  
  // Store the grid (positions will be transformed using T_WK during merge)
  submapOccupancyGrids_[submap_id] = std::move(grid);
}

void Publisher::publishOccupancyGridAsCallback(
    const State& latest_state,
    const AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>>& seSubmapLookup)
{
  LOG(INFO) << "========== publishOccupancyGridAsCallback CALLED with " 
            << seSubmapLookup.size() << " active submaps, " 
            << submapOccupancyGrids_.size() << " stored grids ==========";
  
  okvis::TimerSwitchable timer_total("OccupancyGrid: Total");
  
  // VISUALIZATION: Publish submap boundaries for debugging
  LOG(INFO) << "Creating submap boundary markers for " << seSubmapLookup.size() << " submaps";
  auto marker_array_msg = std::make_shared<visualization_msgs::msg::MarkerArray>();
  int marker_id = 0;
  
  // Define colors for different submaps (cycling through rainbow)
  auto getColor = [](int id) -> std::array<float, 4> {
    float hue = (id * 137.5f) / 360.0f; // Golden angle for good color distribution
    hue = hue - std::floor(hue); // Wrap to [0, 1]
    
    // HSV to RGB conversion (S=1, V=1 for bright colors)
    float h6 = hue * 6.0f;
    float x = 1.0f - std::abs(std::fmod(h6, 2.0f) - 1.0f);
    
    if (h6 < 1.0f) return {1.0f, x, 0.0f, 0.5f};
    else if (h6 < 2.0f) return {x, 1.0f, 0.0f, 0.5f};
    else if (h6 < 3.0f) return {0.0f, 1.0f, x, 0.5f};
    else if (h6 < 4.0f) return {0.0f, x, 1.0f, 0.5f};
    else if (h6 < 5.0f) return {x, 0.0f, 1.0f, 0.5f};
    else return {1.0f, 0.0f, x, 0.5f};
  };
  
  for (const auto& [submap_id, submap] : seSubmapLookup) {
    if (!submap.map) continue;
    
    // Get AABB in map frame K
    const Eigen::Array3f aabb_min_K = submap.map->aabb().min();
    const Eigen::Array3f aabb_max_K = submap.map->aabb().max();
    
    // Transform to world frame
    const Eigen::Isometry3f& T_WK = submap.T_WK;
    Eigen::Vector3f aabb_min_W_vec = T_WK * aabb_min_K.matrix();
    Eigen::Vector3f aabb_max_W_vec = T_WK * aabb_max_K.matrix();
    
    // Recalculate min/max after rotation
    Eigen::Array3f aabb_min_W = aabb_min_W_vec.array().min(aabb_max_W_vec.array());
    Eigen::Array3f aabb_max_W = aabb_min_W_vec.array().max(aabb_max_W_vec.array());
    
    // Create wireframe cube marker
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp.sec = latest_state.timestamp.sec;
    marker.header.stamp.nanosec = latest_state.timestamp.nsec;
    marker.ns = "submap_bounds";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05; // Line width
    
    auto color = getColor(submap_id);
    marker.color.r = color[0];
    marker.color.g = color[1];
    marker.color.b = color[2];
    marker.color.a = color[3];
    
    // Create 12 edges of the bounding box
    auto addEdge = [&](float x1, float y1, float z1, float x2, float y2, float z2) {
      geometry_msgs::msg::Point p1, p2;
      p1.x = x1; p1.y = y1; p1.z = z1;
      p2.x = x2; p2.y = y2; p2.z = z2;
      marker.points.push_back(p1);
      marker.points.push_back(p2);
    };
    
    float xmin = aabb_min_W.x(), ymin = aabb_min_W.y(), zmin = aabb_min_W.z();
    float xmax = aabb_max_W.x(), ymax = aabb_max_W.y(), zmax = aabb_max_W.z();
    
    // Bottom face
    addEdge(xmin, ymin, zmin, xmax, ymin, zmin);
    addEdge(xmax, ymin, zmin, xmax, ymax, zmin);
    addEdge(xmax, ymax, zmin, xmin, ymax, zmin);
    addEdge(xmin, ymax, zmin, xmin, ymin, zmin);
    
    // Top face
    addEdge(xmin, ymin, zmax, xmax, ymin, zmax);
    addEdge(xmax, ymin, zmax, xmax, ymax, zmax);
    addEdge(xmax, ymax, zmax, xmin, ymax, zmax);
    addEdge(xmin, ymax, zmax, xmin, ymin, zmax);
    
    // Vertical edges
    addEdge(xmin, ymin, zmin, xmin, ymin, zmax);
    addEdge(xmax, ymin, zmin, xmax, ymin, zmax);
    addEdge(xmax, ymax, zmin, xmax, ymax, zmax);
    addEdge(xmin, ymax, zmin, xmin, ymax, zmax);
    
    marker_array_msg->markers.push_back(marker);
    
    // Add text label with submap ID
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = marker.header;
    text_marker.ns = "submap_labels";
    text_marker.id = marker_id++;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose.position.x = (xmin + xmax) / 2.0f;
    text_marker.pose.position.y = (ymin + ymax) / 2.0f;
    text_marker.pose.position.z = zmax + 0.5f; // Above the box
    text_marker.scale.z = 0.3; // Text height
    text_marker.color = marker.color;
    text_marker.color.a = 1.0; // Opaque text
    text_marker.text = "Submap " + std::to_string(submap_id);
    
    marker_array_msg->markers.push_back(text_marker);
    
    LOG(INFO) << "  Submap " << submap_id << " bounds: [" 
              << xmin << "," << ymin << "," << zmin << "] to ["
              << xmax << "," << ymax << "," << zmax << "]";
  }
  
  // Publish visualization
  LOG(INFO) << "Publishing " << marker_array_msg->markers.size() << " markers to /okvis/okvis_submap_bounds";
  pubSubmapBounds_.publish(marker_array_msg);
  
  // Create the occupancy grid message
  okvis::TimerSwitchable timer_setup("OccupancyGrid: Setup");
  auto grid_msg = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  grid_msg->header.stamp.sec = latest_state.timestamp.sec;
  grid_msg->header.stamp.nanosec = latest_state.timestamp.nsec;
  grid_msg->header.frame_id = "world";

  // Set grid parameters
  grid_msg->info.resolution = occupancy_grid_resolution_;
  grid_msg->info.width = static_cast<uint32_t>(occupancy_grid_width_ / occupancy_grid_resolution_);
  grid_msg->info.height = static_cast<uint32_t>(occupancy_grid_height_dim_ / occupancy_grid_resolution_);

  // Center the grid on the robot's current position
  const Eigen::Vector3d t_WB = (latest_state.T_WS * T_SB_).r();
  grid_msg->info.origin.position.x = t_WB.x() - occupancy_grid_width_ / 2.0;
  grid_msg->info.origin.position.y = t_WB.y() - occupancy_grid_height_dim_ / 2.0;
  grid_msg->info.origin.position.z = occupancy_grid_height_;
  grid_msg->info.origin.orientation.w = 1.0;

  // Initialize grid data to unknown
  grid_msg->data.resize(grid_msg->info.width * grid_msg->info.height, -1);
  timer_setup.stop();

  // Step 2: Query all submaps directly at their current poses (like mesh generation)
  okvis::TimerSwitchable timer_query("OccupancyGrid: Query All Submaps");
  
  size_t total_queries = 0;
  size_t valid_queries = 0;
  size_t occupied_count = 0;
  size_t free_count = 0;
  
  // Query each active submap directly (no caching)
  for (const auto& [submap_id, submap] : seSubmapLookup) {
    if (!submap.map) {
      continue;
    }
    
    LOG(INFO) << "Querying submap " << submap_id << " at current pose";
    
    // Query each cell in the output grid
    for (uint32_t grid_y = 0; grid_y < grid_msg->info.height; ++grid_y) {
      for (uint32_t grid_x = 0; grid_x < grid_msg->info.width; ++grid_x) {
        total_queries++;
        
        // World position of this grid cell
        const float world_x = grid_msg->info.origin.position.x + (grid_x + 0.5f) * grid_msg->info.resolution;
        const float world_y = grid_msg->info.origin.position.y + (grid_y + 0.5f) * grid_msg->info.resolution;
        const float world_z = occupancy_grid_height_;
        const Eigen::Vector3f point_W(world_x, world_y, world_z);
        
        // Query voxel at this world position
        Eigen::Vector3i voxel_coord;
        if (!submap.map->pointToVoxel<se::Safe::On>(point_W, voxel_coord)) {
          continue; // Outside this submap's bounds
        }
        
        auto data = se::visitor::getData(submap.map->getOctree(), voxel_coord);
        if (!se::is_valid(data)) {
          continue; // No observed data
        }
        
        valid_queries++;
        se::field_t occ = se::get_field(data);
        
        // Convert TSDF to occupancy value
        int8_t grid_value = -1;
        
        if (std::abs(occ) < 0.5f) {
          grid_value = 100; // Surface - fully occupied
          occupied_count++;
        } else if (occ > 0.5f) {
          float normalized = std::min((occ - 0.5f) / 1.5f, 1.0f);
          grid_value = static_cast<int8_t>(75 + normalized * 25);
          occupied_count++;
        } else if (occ < -3.0f) {
          grid_value = 0; // Definitely free
          free_count++;
        } else if (occ < -0.5f) {
          float normalized = std::max((occ + 0.5f) / -2.5f, 0.0f);
          grid_value = static_cast<int8_t>(25 - normalized * 25);
          free_count++;
        }
        
        // Write to output grid (only if better than current value)
        const size_t grid_index = grid_y * grid_msg->info.width + grid_x;
        if (grid_value != -1) {
          // If cell is unknown or this is more confident, update it
          if (grid_msg->data[grid_index] == -1 ||
              (grid_value > 50 && grid_msg->data[grid_index] < grid_value) ||  // More occupied
              (grid_value < 50 && grid_msg->data[grid_index] > grid_value)) {  // More free
            grid_msg->data[grid_index] = grid_value;
          }
        }
      }
    }
  }
  
  timer_query.stop();
  
  LOG(INFO) << "OccupancyGrid Query Stats:";
  LOG(INFO) << "  Total queries: " << total_queries;
  LOG(INFO) << "  Valid responses: " << valid_queries << " (" 
            << (total_queries > 0 ? 100.0f * valid_queries / total_queries : 0) << "%)";
  LOG(INFO) << "  Occupied: " << occupied_count;
  LOG(INFO) << "  Free: " << free_count;
  
  // Count final grid statistics
  size_t final_occupied = 0;
  size_t final_free = 0;
  size_t final_unknown = 0;
  
  for (const auto& cell : grid_msg->data) {
    if (cell == -1) final_unknown++;
    else if (cell >= 50) final_occupied++;
    else final_free++;
  }
  
  LOG(INFO) << "OccupancyGrid Final Statistics:";
  LOG(INFO) << "  Occupied: " << final_occupied << " (" 
            << (100.0f * final_occupied / grid_msg->data.size()) << "%)";
  LOG(INFO) << "  Free: " << final_free << " (" 
            << (100.0f * final_free / grid_msg->data.size()) << "%)";
  LOG(INFO) << "  Unknown: " << final_unknown << " (" 
            << (100.0f * final_unknown / grid_msg->data.size()) << "%)";
  
  // Publish the grid
  pubOccupancyGrid_.publish(grid_msg);
  
  timer_total.stop();
  LOG(INFO) << "========== publishOccupancyGridAsCallback COMPLETED ==========";
}

void Publisher::publishAlignmentPointsAsCallback(const okvis::Time& timestamp, const okvis::kinematics::Transformation& T_WS,
                                                 const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& alignPointCloud,
                                                 bool isMapFrame){

  // finally the landmarks
  pcl::PointCloud<pcl::PointXYZRGB> alignPoints; // Point cloud for matched points.
  alignPoints.reserve(alignPointCloud.size());

  rclcpp::Time t(timestamp.sec, timestamp.nsec); // Header timestamp.

  // transform points into custom world frame:
  /// \todo properly from ros params -- also landmark thresholds below
  for (const auto & pt : alignPointCloud) {

    alignPoints.push_back(pcl::PointXYZRGB());
    const Eigen::Vector3f point = T_WS.T3x4().cast<float>() * pt.homogeneous();
    alignPoints.back().x = point[0];
    alignPoints.back().y = point[1];
    alignPoints.back().z = point[2];
    if(isMapFrame){
      alignPoints.back().r = 255.0f;
      alignPoints.back().g = 140.0f;
      alignPoints.back().b = 0.0f;
    }
    else{
      alignPoints.back().r = 255.0f;
      alignPoints.back().g = 215.0f;
      alignPoints.back().b = 0.0f;
    }
  }
  alignPoints.header.frame_id = "world";

#if PCL_VERSION >= PCL_VERSION_CALC(1,7,0)
  std_msgs::msg::Header header;
  header.stamp = t;
  alignPoints.header.stamp = pcl_conversions::toPCL(header).stamp;
#else
  alignPoints.header.stamp=_t;
#endif

  // and now publish them:
  auto alignPointsMsg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(alignPoints, *alignPointsMsg);
  pubPointsAlignment_.publish(alignPointsMsg);
}

void Publisher::publishRealTimePropagation(const okvis::Time& time, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation, 
                                           const Eigen::Vector3d& linear_velocity, const Eigen::Vector3d& angular_velocity)
{
  rclcpp::Time t(time.sec, time.nsec);
  
  // Odometry
  auto odometryMsg = std::make_shared<nav_msgs::msg::Odometry>();  // Odometry message.
  odometryMsg->header.frame_id = "world";
  odometryMsg->child_frame_id = "body";
  odometryMsg->header.stamp = t;

  // fill orientation
  odometryMsg->pose.pose.orientation.x = orientation.x();
  odometryMsg->pose.pose.orientation.y = orientation.y();
  odometryMsg->pose.pose.orientation.z = orientation.z();
  odometryMsg->pose.pose.orientation.w = orientation.w();

  // fill position
  odometryMsg->pose.pose.position.x = position[0];
  odometryMsg->pose.pose.position.y = position[1];
  odometryMsg->pose.pose.position.z = position[2];

  // note: velocity and angular velocity needs to be expressed in child frame, i.e. "body"
  // see http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html

  // fill velocity
  // ...of body orig. represented in body
  odometryMsg->twist.twist.linear.x = linear_velocity[0];
  odometryMsg->twist.twist.linear.y = linear_velocity[1];
  odometryMsg->twist.twist.linear.z = linear_velocity[2];

  // fill angular velocity
  odometryMsg->twist.twist.angular.x = angular_velocity[0];
  odometryMsg->twist.twist.angular.y = angular_velocity[1];
  odometryMsg->twist.twist.angular.z = angular_velocity[2];

  // publish odometry
  pubObometry_.publish(odometryMsg);
  lastOdom_ = *odometryMsg;
}

void Publisher::republishMeshes()
{
  if(submapSurfaceMesh_.size() == 0){
    LOG(ERROR) << "No meshes present for re-meshing";
    return;
  }
  else{
    LOG(INFO) << "Re-Meshing " << submapSurfaceMesh_.size() << " meshes for text query";
  }

  constexpr size_t n = SupereightMapType::SurfaceMesh::value_type::num_vertexes;

  std_msgs::msg::Header header;
  header.stamp = node_->now();
  header.frame_id = "world";

  // Note: ToDo: Text_embedding_vector_ is set before
  for(auto cached_mesh : submapSurfaceMesh_){
    visualization_msgs::msg::Marker meshMarker;
    meshMarker.header = header;
    meshMarker.pose.orientation.w = 1;
    meshMarker.ns = "submaps"; 
    meshMarker.id = cached_mesh.first;
    meshMarker.scale.x = 1;
    meshMarker.scale.y = 1;
    meshMarker.scale.z = 1;
    meshMarker.color.r = 0.8;
    meshMarker.color.g = 0.8;
    meshMarker.color.b = 0.8;
    meshMarker.color.a = 1;
    meshMarker.lifetime = rclcpp::Duration(std::chrono::seconds(0)); // 0 for infinity
    meshMarker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    meshMarker.action = visualization_msgs::msg::Marker::ADD;

    auto & mesh = submapSurfaceMesh_[cached_mesh.first];

    meshMarker.points.clear();
    meshMarker.points.resize(n * mesh.size());
    size_t valMeshCnt = 0;

    #ifdef OKVIS_COLIDMAP
    meshMarker.colors.clear();
    meshMarker.colors.resize(n * mesh.size());
    for (size_t i = 0; i < mesh.size(); i++) {
      const Eigen::Vector3f v0_W = (submapPoses_[cached_mesh.first] * (mesh[i].vertexes[0].homogeneous())).template head<3>();
      const Eigen::Vector3f v1_W = (submapPoses_[cached_mesh.first] * (mesh[i].vertexes[1].homogeneous())).template head<3>();
      const Eigen::Vector3f v2_W = (submapPoses_[cached_mesh.first] * (mesh[i].vertexes[2].homogeneous())).template head<3>();
      const float triangle_max_z_W = Eigen::Vector3f(v0_W.z(), v1_W.z(), v2_W.z()).maxCoeff();
      if (triangle_max_z_W > mesh_cutoff_z_) {
        continue;
      }

      tf::pointEigenToMsg(v0_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 0]);
      tf::pointEigenToMsg(v1_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 1]);
      tf::pointEigenToMsg(v2_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 2]);
    
      if constexpr(okvis::SupereightMapType::SurfaceMesh::value_type::col_ == se::Colour::On) {
        // Check for valid colours
        if(!mesh[i].colour.vertexes) {
          continue;
        }
        for(size_t v = 0; v < n; v++) {
          auto& msg_colour = meshMarker.colors[n * valMeshCnt + v];
          const auto& mesh_colour = mesh[i].colour.vertexes.value()[v];
          msg_colour.r = float(mesh_colour.r) / UINT8_MAX;
          msg_colour.g = float(mesh_colour.g) / UINT8_MAX;
          msg_colour.b = float(mesh_colour.b) / UINT8_MAX;
          msg_colour.a = 1.0;
        }
      }
      valMeshCnt ++;
    }
    meshMarker.points.resize(n * valMeshCnt);
    meshMarker.colors.resize(n * valMeshCnt);
    submapMeshLookup_rgb_[cached_mesh.first] = meshMarker;
    #else
    for (size_t i = 0; i < mesh.size(); i++) {
      const Eigen::Vector3f v0_W = (submapPoses_[cached_mesh.first] * (mesh[i].vertexes[0].homogeneous())).template head<3>();
      const Eigen::Vector3f v1_W = (submapPoses_[cached_mesh.first] * (mesh[i].vertexes[1].homogeneous())).template head<3>();
      const Eigen::Vector3f v2_W = (submapPoses_[cached_mesh.first] * (mesh[i].vertexes[2].homogeneous())).template head<3>();
      const float triangle_max_z_W = Eigen::Vector3f(v0_W.z(), v1_W.z(), v2_W.z()).maxCoeff();
      if (triangle_max_z_W > mesh_cutoff_z_) {
        continue;
      }

      tf::pointEigenToMsg(v0_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 0]);
      tf::pointEigenToMsg(v1_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 1]);
      tf::pointEigenToMsg(v2_W.template cast<double>(), meshMarker.points[n * valMeshCnt + 2]);
      valMeshCnt ++;
    }
    meshMarker.points.resize(n * valMeshCnt);
    submapMeshLookup_rgb_[cached_mesh.first] = meshMarker;
    #endif
  }

  auto markerarraymsg_ = std::make_shared<visualization_msgs::msg::MarkerArray>();
  markerarraymsg_->markers.clear();
  //If too many submaps are to be updated at once (> 1GB) ROS considers it has gone out of sync and does not update.
  // To prevent this, we generate buffers of submaps and send them in smaller packets to bypass this issue
  int submap_publisher_buffer = 0;
  for(auto it: submapMeshLookup_rgb_){
    it.second.header = header;
    markerarraymsg_->markers.push_back(it.second);
    submap_publisher_buffer++;

    if(submap_publisher_buffer == 30){
      submap_publisher_buffer = 0;
      pubSubmapMesh_.publish(markerarraymsg_);
      markerarraymsg_->markers.clear();
    }
  }
  
  if(markerarraymsg_->markers.size() > 0) {
      pubSubmapMesh_.publish(markerarraymsg_);
  }

  return;
}

}  // namespace okvis
