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

#include <iostream>

#include <gtest/gtest.h>

#include <okvis/ObjectMapping.hpp>
#include <opencv2/imgcodecs.hpp> 

void setSmallRectangle(cv::Mat& image, float value) {
  for(size_t i = 0; i < 10; i++){
    for(size_t j = 0; j < 20; j++){
      image.at<float>(100 + i, 200 + j) = value;
    }
  }
}

void setBigRectangle(cv::Mat& image, float value) {
  for(size_t i = 0; i < 100; i++){
    for(size_t j = 0; j < 40; j++){
      image.at<float>(110 + i, 190 + j) = value;
    }
  }
}

TEST(ObjectMapping, undersegmentedSAM) {
  // Create Random point cloud (cube of dimension dim and resolution res)
  okvis::CameraData::DeepLearningImageData test_sam;
  test_sam.data_ptr.reset(new float[1 * 480 * 640]);
  test_sam.shape = {1, 480, 640};

  cv::Mat sam_segment(480, 640, CV_32FC1, test_sam.data_ptr.get());
  sam_segment.setTo(cv::Scalar(se::g_no_id)); 
  setSmallRectangle(sam_segment, 1);
  setBigRectangle(sam_segment, 1);
  sam_segment.convertTo(sam_segment, CV_16UC1);

  cv::Mat supereightSegment(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  setSmallRectangle(supereightSegment, 1);
  setBigRectangle(supereightSegment, 2);
  supereightSegment.convertTo(supereightSegment, CV_16UC1);

  se::Image<se::id_t> supereightSegmentsImage(supereightSegment.cols, supereightSegment.rows, supereightSegment.ptr<se::id_t>());

  okvis::AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>> seSubmapLookup;
  okvis::AlignedUnorderedMap<uint64_t, se::TriangleMesh<okvis::SupereightMapType::DataType::col_, okvis::SupereightMapType::DataType::id_>> seMeshLookup;
  okvis::ObjectMap object_map(seSubmapLookup, seMeshLookup);
  cv::Mat invalid_depth_mask(480, 640, CV_8UC1, cv::Scalar(0));
  std::set<se::id_t> ids;
  cv::Mat returned_indexes = object_map.trackSegments(invalid_depth_mask, test_sam, supereightSegmentsImage, ids, 0.000325, 0.7);

  cv::Mat diff;
  cv::compare(returned_indexes, supereightSegment, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  EXPECT_EQ(nz, 0);
}

TEST(ObjectMapping, oversegmentedSAM) {
  okvis::AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>> seSubmapLookup;
  okvis::AlignedUnorderedMap<uint64_t, se::TriangleMesh<okvis::SupereightMapType::DataType::col_, okvis::SupereightMapType::DataType::id_>> seMeshLookup;
  okvis::ObjectMap object_map(seSubmapLookup, seMeshLookup);
  
  // Create Random point cloud (cube of dimension dim and resolution res)
  okvis::CameraData::DeepLearningImageData test_sam;
  test_sam.data_ptr.reset(new float[2 * 480 * 640]);
  test_sam.shape = {2, 480, 640};

  cv::Mat sam_segment(480, 640, CV_32FC1, test_sam.data_ptr.get());
  sam_segment.setTo(cv::Scalar(se::g_no_id)); 
  setSmallRectangle(sam_segment, 1);
  cv::Mat sam_segment_big(480, 640, CV_32FC1, test_sam.data_ptr.get() + 480 * 640);
  sam_segment_big.setTo(cv::Scalar(se::g_no_id)); 
  setBigRectangle(sam_segment_big, 1);
 

  cv::Mat supereightSegment(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  se::id_t current_id = object_map.obtainNewId();
  setSmallRectangle(supereightSegment, static_cast<float>(current_id));
  setBigRectangle(supereightSegment, static_cast<float>(current_id));
  supereightSegment.convertTo(supereightSegment, CV_16UC1);
  cv::Mat test = supereightSegment * 10000.0;
  cv::imwrite("/home/barbas/lmtest/input_tracked.png", test);

  se::Image<se::id_t> supereightSegmentsImage(supereightSegment.cols, supereightSegment.rows, supereightSegment.ptr<se::id_t>());


  cv::Mat invalid_depth_mask(480, 640, CV_8UC1, cv::Scalar(0));
  std::set<se::id_t> ids;
  cv::Mat returned_indexes = object_map.trackSegments(invalid_depth_mask, test_sam, supereightSegmentsImage, ids, 0.000325, 0.7);

  cv::Mat expected_return(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  setSmallRectangle(expected_return, static_cast<float>(current_id + 1));
  setBigRectangle(expected_return, static_cast<float>(current_id));
  expected_return.convertTo(expected_return, CV_16UC1);

  cv::Mat diff;
  cv::compare(returned_indexes, expected_return, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  EXPECT_EQ(nz, 0);
}


TEST(ObjectMapping, oversegmentedSAMFilter) {
  okvis::AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>> seSubmapLookup;
  okvis::AlignedUnorderedMap<uint64_t, se::TriangleMesh<okvis::SupereightMapType::DataType::col_, okvis::SupereightMapType::DataType::id_>> seMeshLookup;
  okvis::ObjectMap object_map(seSubmapLookup, seMeshLookup);
  
  // Create Random point cloud (cube of dimension dim and resolution res)
  okvis::CameraData::DeepLearningImageData test_sam;
  test_sam.data_ptr.reset(new float[2 * 480 * 640]);
  test_sam.shape = {2, 480, 640};

  cv::Mat sam_segment(480, 640, CV_32FC1, test_sam.data_ptr.get());
  sam_segment.setTo(cv::Scalar(se::g_no_id)); 
  setSmallRectangle(sam_segment, 1);
  cv::Mat sam_segment_big(480, 640, CV_32FC1, test_sam.data_ptr.get() + 480 * 640);
  sam_segment_big.setTo(cv::Scalar(se::g_no_id)); 
  setBigRectangle(sam_segment_big, 1);
 

  cv::Mat supereightSegment(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  se::id_t current_id = object_map.obtainNewId();
  setSmallRectangle(supereightSegment, static_cast<float>(current_id));
  setBigRectangle(supereightSegment, static_cast<float>(current_id));
  //Now we set some regions to not 
  supereightSegment.convertTo(supereightSegment, CV_16UC1);

  se::Image<se::id_t> supereightSegmentsImage(supereightSegment.cols, supereightSegment.rows, supereightSegment.ptr<se::id_t>());
  cv::Mat invalid_depth_mask(480, 640, CV_8UC1, cv::Scalar(0));
  std::set<se::id_t> ids;
  cv::Mat returned_indexes = object_map.trackSegments(invalid_depth_mask, test_sam, supereightSegmentsImage, ids, 0.001, 0.7);

  cv::Mat expected_return(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  //As the small rectangle is filtered by the segment size threshold, we only want to update the big one
  setBigRectangle(expected_return, static_cast<float>(current_id));
  expected_return.convertTo(expected_return, CV_16UC1);

  cv::Mat diff;
  cv::compare(returned_indexes, expected_return, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  EXPECT_EQ(nz, 0);
}

TEST(ObjectMapping, oversegmentedSAMUnallocated) {
  okvis::AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>> seSubmapLookup;
  okvis::AlignedUnorderedMap<uint64_t, se::TriangleMesh<okvis::SupereightMapType::DataType::col_, okvis::SupereightMapType::DataType::id_>> seMeshLookup;
  okvis::ObjectMap object_map(seSubmapLookup, seMeshLookup);
  
  // Create Random point cloud (cube of dimension dim and resolution res)
  okvis::CameraData::DeepLearningImageData test_sam;
  test_sam.data_ptr.reset(new float[2 * 480 * 640]);
  test_sam.shape = {2, 480, 640};

  cv::Mat sam_segment(480, 640, CV_32FC1, test_sam.data_ptr.get());
  sam_segment.setTo(cv::Scalar(se::g_no_id)); 
  setSmallRectangle(sam_segment, 1);
  cv::Mat sam_segment_big(480, 640, CV_32FC1, test_sam.data_ptr.get() + 480 * 640);
  sam_segment_big.setTo(cv::Scalar(se::g_no_id)); 
  setBigRectangle(sam_segment_big, 1);
 

  cv::Mat supereightSegment(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  se::id_t current_id = object_map.obtainNewId();
  setSmallRectangle(supereightSegment, static_cast<float>(current_id));
  setBigRectangle(supereightSegment, static_cast<float>(current_id));
  supereightSegment.convertTo(supereightSegment, CV_16UC1);

  for(int i = 0; i < 20; i++) {
    for(int j = 0; j < 100; j++) {
      supereightSegment.at<uint16_t>(110 + i, 210 + j) = se::g_not_mapped;
    }
  }


  se::Image<se::id_t> supereightSegmentsImage(supereightSegment.cols, supereightSegment.rows, supereightSegment.ptr<se::id_t>());


  cv::Mat invalid_depth_mask(480, 640, CV_8UC1, cv::Scalar(0));
  std::set<se::id_t> ids;
  cv::Mat returned_indexes = object_map.trackSegments(invalid_depth_mask, test_sam, supereightSegmentsImage, ids, 0.000325, 0.7);

  cv::Mat expected_return(480, 640, CV_32FC1, cv::Scalar(se::g_no_id));
  //As the small rectangle is filtered by the segment size threshold, we only want to update the big one
  setSmallRectangle(expected_return, static_cast<float>(current_id + 1));
  setBigRectangle(expected_return, static_cast<float>(current_id));
  expected_return.convertTo(expected_return, CV_16UC1);

  cv::Mat diff;
  cv::compare(returned_indexes, expected_return, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  EXPECT_EQ(nz, 0);
}

TEST(ObjectMapping, SAMSquares) {
  okvis::AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>> seSubmapLookup;
  okvis::AlignedUnorderedMap<uint64_t, se::TriangleMesh<okvis::SupereightMapType::DataType::col_, okvis::SupereightMapType::DataType::id_>> seMeshLookup;
  okvis::ObjectMap object_map(seSubmapLookup, seMeshLookup);
  
  // Create Random point cloud (cube of dimension dim and resolution res)
  okvis::CameraData::DeepLearningImageData test_sam;
  test_sam.data_ptr.reset(new float[2 * 480 * 640]);
  test_sam.shape = {2, 480, 640};

  cv::Mat supereightSegment(480, 640, CV_16UC1, cv::Scalar(se::g_no_id));
  se::id_t current_id = object_map.obtainNewId();

  cv::Mat sam_segment_big(480, 640, CV_32FC1, test_sam.data_ptr.get() + 480 * 640);
  sam_segment_big.setTo(cv::Scalar(se::g_no_id));
  for(int i = 0; i < 80; i++){
    for(int j = 0; j < 80; j++){
      sam_segment_big.at<float>(160 + i, 160 + j) = 1.f;
      supereightSegment.at<uint16_t>(160 + i, 160 + j) = current_id;
    }
  }

  current_id = object_map.obtainNewId();

  cv::Mat sam_segment(480, 640, CV_32FC1, test_sam.data_ptr.get());
  sam_segment.setTo(cv::Scalar(se::g_no_id)); 
  for(int i = 0; i < 40; i++){
    for(int j = 0; j < 40; j++){
      sam_segment.at<float>(180 + i, 180 + j) = 1.f;
      supereightSegment.at<uint16_t>(180 + i, 180 + j) = current_id;
    }
  }

  current_id = object_map.obtainNewId();
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 20; j++){
      supereightSegment.at<uint16_t>(190 + i, 190 + j) = current_id;
    }
  }
  se::Image<se::id_t> supereightSegmentsImage(supereightSegment.cols, supereightSegment.rows, supereightSegment.ptr<se::id_t>());
  cv::Mat invalid_depth_mask(480, 640, CV_8UC1, cv::Scalar(0));
  std::set<se::id_t> ids;
  cv::Mat returned_indexes = object_map.trackSegments(invalid_depth_mask, test_sam, supereightSegmentsImage, ids, 0.000325, 0.7);

  cv::Mat diff;
  cv::compare(returned_indexes, supereightSegment, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  EXPECT_EQ(nz, 0);
}

TEST(ObjectMapping, emptySeSegments) {
  okvis::AlignedUnorderedMap<uint64_t, se::Submap<okvis::SupereightMapType>> seSubmapLookup;
  okvis::AlignedUnorderedMap<uint64_t, se::TriangleMesh<okvis::SupereightMapType::DataType::col_, okvis::SupereightMapType::DataType::id_>> seMeshLookup;
  okvis::ObjectMap object_map(seSubmapLookup, seMeshLookup);
  
  // Create Random point cloud (cube of dimension dim and resolution res)
  okvis::CameraData::DeepLearningImageData test_sam;
  test_sam.data_ptr.reset(new float[2 * 480 * 640]);
  test_sam.shape = {2, 480, 640};

  cv::Mat expectedVal(480, 640, CV_16UC1, cv::Scalar(se::g_no_id));
  se::id_t current_id = object_map.obtainNewId();

  cv::Mat sam_segment_big(480, 640, CV_32FC1, test_sam.data_ptr.get() + 480 * 640);
  sam_segment_big.setTo(cv::Scalar(se::g_no_id));
  for(int i = 0; i < 80; i++){
    for(int j = 0; j < 80; j++){
      sam_segment_big.at<float>(160 + i, 160 + j) = 1.f;
      expectedVal.at<uint16_t>(160 + i, 160 + j) = current_id + 2;
    }
  }

  cv::Mat sam_segment(480, 640, CV_32FC1, test_sam.data_ptr.get());
  sam_segment.setTo(cv::Scalar(se::g_no_id)); 
  for(int i = 0; i < 40; i++){
    for(int j = 0; j < 40; j++){
      sam_segment.at<float>(180 + i, 180 + j) = 1.f;
      expectedVal.at<uint16_t>(180 + i, 180 + j) = current_id + 1;
    }
  }

  cv::Mat supereightSegment(480, 640, CV_16UC1, cv::Scalar(se::g_no_id));

  se::Image<se::id_t> supereightSegmentsImage(supereightSegment.cols, supereightSegment.rows, supereightSegment.ptr<se::id_t>());
  cv::Mat invalid_depth_mask(480, 640, CV_8UC1, cv::Scalar(0));
  std::set<se::id_t> ids;
  cv::Mat returned_indexes = object_map.trackSegments(invalid_depth_mask, test_sam, supereightSegmentsImage, ids, 0.000325, 0.7);
  
  cv::Mat diff;
  cv::compare(returned_indexes, expectedVal, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  EXPECT_EQ(nz, 0);
}