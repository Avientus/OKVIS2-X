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
 * @file Publisher.hpp
 * @brief Header file for the Publisher class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_ROS2_PUBLISHER_HPP_
#define INCLUDE_OKVIS_ROS2_PUBLISHER_HPP_

#include <memory>

#if __has_include(<cv_bridge/cv_bridge.hpp>) // requires GCC >= 5
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h> // ros2 changed to .hpp some point...
#endif
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#pragma GCC diagnostic pop
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <mutex>

#include <okvis/ViInterface.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/ObjectMapping.hpp>
#include <okvis/TrajectoryOutput.hpp>

#include <se/external/tinycolormap.hpp>

#include <okvis/ros2/eigen_conversions.hpp>

#include <okvis/ThreadedPublisher.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief This class handles the publishing to either ROS topics or files.
 */
class Publisher
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief Default constructor.
  Publisher();

  /// \brief Constructor with node.
  /// \param node The ROS2 node.
  Publisher(
    std::shared_ptr<rclcpp::Node> node,
    std::shared_ptr<ThreadedPublisher> threadedOdometryPublisher, 
    std::shared_ptr<ThreadedPublisher> threadedImagePublisher,
    std::shared_ptr<ThreadedPublisher> threadedPublisher);
  ~Publisher();

  /// \name Setters
  /// \{

  /// \brief Set up the whole node.
  /// \param node The ROS2 node.
  void setupNode(std::shared_ptr<rclcpp::Node> node);

  /**
   * @brief Set the body.
   * @param T_BS Transform body-IMU.
   */
  void setBodyTransform(const okvis::kinematics::Transformation& T_BS);

  /**
   * @brief Set the realtime publishing rate.
   * @param odometryPublishingRate The rate.
   */
  void setOdometryPublishingRate(double odometryPublishingRate) {
    RCLCPP_INFO(node_->get_logger(), "Setting odometry publishing rate to %f Hz.", odometryPublishingRate);
    odometryPublishingRate_ = odometryPublishingRate;
  }

  /**
   * @brief Set CSV file.
   * @param filename Write CSV trajectory to this file.
   * @param rpg If true, uses the RPG format, otherwise the EuRoC format.
   */
  void setCsvFile(const std::string & filename, bool rpg = false);

  /// @}

  /**
   * @brief Process the updated states and publish
   *        (plus write it to trajectory file and visualise, if desired). Set as callback.
   * @param state The current state to process.
   * @param trackingState The additional tracking info to process.
   * @param updatedStates All updated states.
   * @param landmarks The currently optimised landmarks.
   */
  void publishEstimatorUpdate(const State& state, const TrackingState & trackingState,
                              std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
                              std::shared_ptr<const okvis::MapPointVector> landmarks);

  /**
   * @brief Set up the topics.
   * @param nCameraSystem Multi-camera sensor setup.
   */
  void setupImageTopics(const okvis::cameras::NCameraSystem & nCameraSystem);

  /**
   * @brief Set up the topics.
   */
  void setupNetworkTopics(const std::string & topicName);
  
  void setMeshesPath(std::string meshesDir);

  /**
   * \brief          Publish any named images as such.
   * \param images   Named images to publish.
   * \param timestamp Optional timestamp for the images (particularly for stereo pairs).
   * \return True on success.
   */
  bool publishImages(const std::map<std::string, cv::Mat>& images, 
                     const okvis::Time& timestamp = okvis::Time(0)) const;

  /**
   * \brief          Add an IMU measurement for propagation and publishing.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \return True on success.
   */
  bool realtimePredictAndPublish(const okvis::Time& stamp,
                                 const Eigen::Vector3d& alpha,
                                 const Eigen::Vector3d& omega);

  /**
   * @brief Submap mesh callback
   */
  void publishSubmapsAsCallback(std::unordered_map<uint64_t, okvis::kinematics::Transformation, std::hash<uint64_t>, std::equal_to<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, okvis::kinematics::Transformation>>> submapPoseLookup, 
                                std::unordered_map<uint64_t, std::shared_ptr<okvis::SupereightMapType>> submapLookup,
                                std::shared_ptr<okvis::ObjectMap> objectMap);

  /**
   * @brief Publish a horizontal slice through the occupancy field
   */
  void publishFieldSliceAsCallback(const State& latest_state,
                                   const AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>>& seSubmapLookup);

  /**
   * @brief Publish a 2D occupancy grid at a fixed height
   * @param latest_state Current robot state
   * @param seSubmapLookup Map of submaps
   */
  void publishOccupancyGridAsCallback(const State& latest_state,
                                      const AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>>& seSubmapLookup);
  
  /**
   * @brief Extract and store occupancy grid from a submap
   * @param submap_id Submap ID
   * @param submap Submap data
   * @param robot_height_z Robot's current height in world frame (z-coordinate)
   */
  void extractSubmapOccupancyGrid(uint64_t submap_id,
                                   const se::Submap<okvis::SupereightMapType>& submap,
                                   float robot_height_z);

  /**
   * @brief Set occupancy grid resolution
   * @param resolution Grid resolution in meters
   */
  void setOccupancyGridResolution(float resolution) { occupancy_grid_resolution_ = resolution; }

  /**
   * @brief Set occupancy grid dimensions
   * @param width Grid width in meters
   * @param height Grid height in meters
   */
  void setOccupancyGridSize(float width, float height) { 
    occupancy_grid_width_ = width; 
    occupancy_grid_height_dim_ = height; 
  }

  /**
   * @brief Set occupancy threshold for classifying cells as occupied
   * @param threshold TSDF threshold value (cells with abs(TSDF_value) < threshold are considered occupied)
   */
  void setOccupancyGridOccupiedThreshold(float threshold) { occupancy_grid_occupied_threshold_ = threshold; }

  /**
   * @brief Map-to-frame points visualization callback
   */
  void publishAlignmentPointsAsCallback(const okvis::Time& timestamp, const okvis::kinematics::Transformation& T_WS,
                                        const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& alignPointCloud,
                                        bool isMapFrame=false);

  /**
   * @brief Set query (embedding) for VL queries
   */
  void setTextEmbeddingVector(const Descriptor& text_embedding_vector){text_embedding_vector_ = text_embedding_vector;}

  /**
   * @brief Set Mesh Publishing Mode (Colors or Activations)
   */
  void setMeshPublishingMode(const std::string& publishMode);

  /**
   * @brief Re-Publish meshes (if publishing mode has changed)
   */
  void republishMeshes();

  /** 
   * @brief Set Cutoff z value for published meshes
   */
  void setMeshCutoffZ(float z_max){mesh_cutoff_z_ = z_max;}

  /**
   * @brief Publish auxiliary state obtained from IMU real-time propagation
   */
  void publishRealTimePropagation(const okvis::Time& time, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation, 
                                  const Eigen::Vector3d& linear_velocity, const Eigen::Vector3d& angular_velocity);

  private:

  /// @name Node and subscriber related
  /// @{

  std::shared_ptr<rclcpp::Node> node_; ///< The node.

  std::shared_ptr<ThreadedPublisher> threadedOdometryPublisher_;  ///< Odometry publishers.
  std::shared_ptr<ThreadedPublisher> threadedImagePublisher_;  ///< Image publishers.
  std::shared_ptr<ThreadedPublisher> threadedPublisher_;  ///< Non-images publishers.

  /// \brief The publisher for matched points.
  okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::PointCloud2> pubPointsMatched_;
  /// \brief The publisher for the odometry.
  okvis::ThreadedPublisher::PublisherHandle<nav_msgs::msg::Odometry> pubObometry_;
  /// \brief The publisher for the path.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::Marker> pubPath_;
  /// \brief The publisher for the transform.
  okvis::ThreadedPublisher::PublisherHandle<geometry_msgs::msg::TransformStamped> pubTransform_;
  /// \brief The publisher for a robot / camera mesh.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::Marker> pubMesh_;
  /// \brief The publisher for submap meshes.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::MarkerArray> pubSubmapMesh_;
  /// \brief The publisher for aligned points.
  okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::PointCloud2> pubPointsAlignment_;
  /// \brief The publisher for the planned path.
  okvis::ThreadedPublisher::PublisherHandle<nav_msgs::msg::Path> pubPlannedPath_;
  /// \brief The publisher for the slice.
  okvis::ThreadedPublisher::PublisherHandle<visualization_msgs::msg::Marker> slice_pub_;
  /// \brief The publisher for the occupancy grid.
  okvis::ThreadedPublisher::PublisherHandle<nav_msgs::msg::OccupancyGrid> pubOccupancyGrid_;

  /// \brief Image publishers.
  std::map<std::string, okvis::ThreadedPublisher::PublisherHandle<sensor_msgs::msg::Image>> pubImages_; ///< Image publisher map.

  /// @}
  /// @name To be published
  /// @{

  std::vector<ImuMeasurement> imuMeasurements_; ///< Buffered IMU measurements (realtime pub.).
  okvis::Trajectory trajectory_; ///< Underlying trajectory object for state queries.
  okvis::kinematics::Transformation T_BS_; ///< transform from imu frame to body frame
  okvis::kinematics::Transformation T_SB_; ///< transform from body frame to imu frame
  visualization_msgs::msg::Marker::SharedPtr meshMsg_; ///< Mesh message.
  double odometryPublishingRate_; ///< Keep track of last publishing (to maintain rate).
  okvis::Time lastTime_ = okvis::Time(0); ///< Publishing rate for realtime propagation.
  TrajectoryOutput trajectoryOutput_; ///< Trajectory output in case demanded.
  std::atomic_bool trajectoryLocked_; ///< Lock the trajectory object (realtime/update are async.).
  std::string meshesDir_; ///< Directory for meshes.
  nav_msgs::msg::Path path_; ///< The path message.
  Eigen::Matrix4d T_SC_; // Tf from Sensor to Camera
  Eigen::Quaterniond q_sc_;
  
  nav_msgs::msg::Odometry lastOdom_; ///< Last odometry message.

  // TF2 transform broadcasting
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; ///< Transform broadcaster for tf2
  okvis::kinematics::Transformation T_odom_world_; ///< Transformation from odom frame to world frame
  okvis::kinematics::Transformation lastWorldPose_; ///< Last published world frame pose (for loop closure detection)
  bool lastWorldPoseValid_; ///< Whether lastWorldPose_ is valid
  okvis::Time lastWorldPoseTime_; ///< Timestamp of last world pose
  okvis::Time lastOdomDriftLogTime_; ///< Timestamp of last odom drift logging (for periodic logging)

  // Submap-related members.
  std::unordered_map<uint64_t, visualization_msgs::msg::Marker> submapMeshLookup_; ///< Lookup for submap meshes.
  std::unordered_map<uint64_t, okvis::SupereightMapType::SurfaceMesh> submapSurfaceMesh_; ///< Surface meshes for submaps.
  std::unordered_map<uint64_t, visualization_msgs::msg::Marker> submapMeshLookup_embedding_; ///< Lookup for submap embedding meshes.
  std::unordered_map<uint64_t, visualization_msgs::msg::Marker> submapMeshLookup_rgb_; ///< Lookup for submap RGB meshes.
  std::map<uint64_t, const okvis::AlignedMap<se::id_t, SubmapObject>*> submapObjects_; ///< Objects in submaps.
  std::map<uint64_t, Eigen::Matrix4f> submapPoses_; ///< Poses of submaps.
  
  // Accumulated occupancy grid storage (similar to mesh storage)
  // Stores world-frame positions with pose snapshot for loop closure detection
  struct SubmapOccupancyGrid {
    Eigen::Array3f origin_W;         ///< Origin in WORLD frame (at extraction time)
    Eigen::Isometry3f T_WK_snapshot; ///< Submap pose at extraction time
    float resolution;                ///< Grid resolution
    uint32_t width;                  ///< Grid width in cells
    uint32_t height;                 ///< Grid height in cells
    std::vector<int8_t> data;        ///< Occupancy data (0=free, 100=occupied, -1=unknown)
  };
  std::unordered_map<uint64_t, SubmapOccupancyGrid> submapOccupancyGrids_; ///< Stored occupancy grids per submap
  
  // Global grid cache for performance (stores merged result)
  std::vector<int8_t> globalOccupancyGrid_;  ///< Persistent global grid (world frame)
  Eigen::Vector2f globalGridOrigin_;         ///< Global grid origin in world frame
  uint32_t globalGridWidth_;                 ///< Global grid width in cells
  uint32_t globalGridHeight_;                ///< Global grid height in cells
  
  // Pose tracking for loop closure detection
  std::unordered_map<uint64_t, Eigen::Isometry3f> lastMergedPoses_; ///< Track T_WK for each merged submap
  
  // Helper to check if pose changed significantly
  bool poseChanged(const Eigen::Isometry3f& pose1, const Eigen::Isometry3f& pose2, 
                   float trans_thresh = 0.01f, float rot_thresh = 0.01f) const;

  /**
   * @brief Update odom frame transformation after loop closure detection
   * @param currentWorldPose Current pose in world frame
   * @param currentTime Current timestamp
   * @param loopClosureDetected True if loop closure was detected (e.g., from trackingState.recognisedPlace or large updatedStates)
   */
  void updateOdomFrameAfterLoopClosure(
      const okvis::kinematics::Transformation& currentWorldPose,
      const okvis::Time& currentTime,
      bool loopClosureDetected);

  /**
   * @brief Publish transforms via tf2: world->odom and odom->body
   * @param T_WB Transformation from world to body frame
   * @param t ROS2 timestamp
   */
  void publishTransforms(
      const okvis::kinematics::Transformation& T_WB,
      const rclcpp::Time& t);

  /**
   * @brief Update the global occupancy grid with data from a single submap
   * @param submap_id Submap ID
   * @param submap Submap data
   * @param grid_width Grid width in cells
   * @param grid_height Grid height in cells
   * @param grid_origin_x Grid origin X in world frame
   * @param grid_origin_y Grid origin Y in world frame
   * @param grid_resolution Grid resolution in meters
   * @param robot_height_z Robot's current height in world frame (z-coordinate)
   * @param global_grid Reference to the global grid to update
   * @return Number of valid cells updated
   */
  size_t updateGlobalGridFromSubmap(uint64_t submap_id,
                                     const se::Submap<okvis::SupereightMapType>& submap,
                                     uint32_t grid_width,
                                     uint32_t grid_height,
                                     float grid_origin_x,
                                     float grid_origin_y,
                                     float grid_resolution,
                                     float robot_height_z,
                                     std::vector<int8_t>& global_grid);
  std::atomic_bool skip_uncolored_ = true; ///< Flag if uncolored vertices should be skipped at mesh publishing

  // Vision-Language Query Related Members
  Descriptor text_embedding_vector_; ///< Text embedding vector.
  std::string publishMode_ = "Colors"; ///< Mode which meshes are published: RGB ("Colors") | Text Query ("Activations")

  float mesh_cutoff_z_ = std::numeric_limits<float>::max(); ///< z cutoff value for visualisation

  // Occupancy grid parameters
  float occupancy_grid_resolution_ = 0.05f; ///< Resolution of the occupancy grid in meters
  float occupancy_grid_width_ = 20.0f; ///< Width of the occupancy grid in meters
  float occupancy_grid_height_dim_ = 20.0f; ///< Height (y-dimension) of the occupancy grid in meters
  float occupancy_grid_occupied_threshold_ = 0.5f; ///< TSDF threshold for classifying cells as occupied (values with abs < threshold are occupied)
  
  // Occupancy grid caching for efficiency
  struct SubmapBounds {
    Eigen::Array3f min;
    Eigen::Array3f max;
  };
  std::unordered_map<uint64_t, SubmapBounds> submapBoundsCache_; ///< Cached AABB for each submap
};

}

#endif /* INCLUDE_OKVIS_ROS2_PUBLISHER_HPP_ */
