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

#include "glog/logging.h"
#include "ceres/ceres.h"
#include "ceres/covariance.h"
#include <gtest/gtest.h>
#include <iostream>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

#include <okvis/ceres/RadarErrorAsynchronous.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

// Define error thresholds
const double jacobianThresh = 1e-03; // Tolerance for error of analytic Jacobian compared to numerical Jacobian
const double posThresh = 1e-01; // Position estimate tolerance [m]
const double rotThresh = 1e-02; // Orientation estimate tolerance [°] ; 1e-02 rad = 0.57 °
const double velThresh = 5e-02; // Velocity estimate threshold [m/s]
const double bgThresh = 1e-03; // Gyroscope bias estimate threshold [rad/s]
const double baThresh = 1e-03; // Acclerometer bias estimate threshold [m/s²]

// SpeedAndBias Signal disturbances
const double bg_dist_std = 2e-03; // Gyroscope bias disturbance (standard deviation) 2e-03 [rad/s] => 0.57 [°]
const double ba_dist_std = 1e-02; // Accelerometer bias disturbance (standard deviation) [m/s²]
const double v_dist_std = 0.2;  // SpeedAndBias Velocity disturbance [m/s]

// Radar uncertainty
const double radar_std = 0.3; // Radar velocity accuracy uncertainty [m/s] ==> 5cm/s

// Odometry uncertainties (relative Pose errors)
const double odometry_trans_std = 3e-01; // 30 cm odometry error
const double odometry_rot_std = 4e-01;  // 23 degree odometry error

// Trajectory parameters (circular motion with constant speed)
const double omega = 0.2; // rotation rate 0.2 [rad/s] => 11.5 [°/s]
const double radius = 5.0; // radius

// sampling rates - REALISTIC SCENARIO
const double radarRate = 10.0; // 10 Hz - realistic radar measurement rate (per radar)
const double frameRate = 15.0; // 15 Hz - realistic keyframe/camera frame rate
const int numRadars = 3; // Number of radars (unsynchronized)


/**
 * @brief Test for RadarErrorAsynchronous with IMU pre-integration
 * 
 * HOW THIS TEST WORKS:
 * ====================
 * 
 * 1. TRAJECTORY GENERATION:
 *    - Generates a circular trajectory (constant angular velocity)
 *    - Simulates IMU measurements at 200Hz (realistic IMU rate)
 *    - Creates keyframes at 15Hz (camera/visual frames)
 *    - Creates radar measurements from 3 radars, each at 10Hz (unsynchronized)
 * 
 * 2. ASYNCHRONOUS NATURE:
 *    - Keyframes occur at times tk (15Hz)
 *    - Radar measurements occur at times tr_i (10Hz per radar, unsynchronized)
 *    - tr_i != tk in general (asynchronous)
 *    - Each radar has different measurement times (not synchronized)
 *    - IMU pre-integration propagates state from tk to tr_i for each radar
 * 
 * 3. STATE ESTIMATION:
 *    - Initial states are disturbed (pose, velocity, biases)
 *    - Ceres optimizer estimates true states from:
 *      a) Radar velocity measurements (at tr)
 *      b) IMU pre-integration (from tk to tr)
 *      c) Relative pose constraints (between keyframes)
 *      d) IMU error terms (between keyframes)
 * 
 * 4. VERIFICATION:
 *    - Numerical Jacobian verification (compares analytical vs numerical)
 *    - State convergence verification (estimated vs ground truth)
 * 
 * KEY CONCEPT - ASYNCHRONOUS PRE-INTEGRATION:
 * -------------------------------------------
 * When a radar measurement arrives at time tr, we need the robot state at tr.
 * But we only have state estimates at keyframe times tk.
 * Solution: Use IMU pre-integration to propagate state from tk to tr.
 * 
 * Example timeline (with 3 unsynchronized radars):
 *   tk=0.0s    -> keyframe (state estimate available)
 *   tr₁=0.05s  -> radar 1 measurement (need state here)
 *   tr₂=0.08s  -> radar 2 measurement (different time!)
 *   tr₃=0.12s  -> radar 3 measurement (different time!)
 *   tk=0.067s  -> next keyframe
 *   tr₁=0.15s  -> next radar 1 measurement
 *   ...
 * 
 * The RadarErrorAsynchronous class handles this by:
 * - Taking IMU measurements between tk and tr
 * - Pre-integrating them to get pose/velocity at tr
 * - Computing expected radar velocity at tr
 * - Comparing with measured radar velocity
 */
TEST(okvisTestSuite, RadarErrorAsynchronous){

    // ========================================================================
    // SETUP: Define trajectory and sensor parameters
    // ========================================================================
    
    Eigen::Vector3d omegaVec(omega,omega,omega); // Angular velocity in world frame [rad/s]
    Eigen::Vector3d p_c(0.0,0.0,1.0); // Center point of circular trajectory
    Eigen::Vector3d p0 = p_c + Eigen::Vector3d(0,radius,1); // Starting point at (0, radius, 1)

    // IMU parameters - realistic rate for pre-integration
    const double imuRate = 200; // 200 Hz - realistic IMU rate
    const double duration = 10.0; // Test duration [s]
    okvis::ImuMeasurementDeque imuMeasurements; // Queue to store IMU measurements
    const double dt=1.0/double(imuRate); // Time step [s]

    okvis::ImuParameters imuParameters;
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;

    // Set the radar parameters for multiple radars (each with different extrinsics)
    std::vector<okvis::RadarParameters> radarParametersList;
    for(int r = 0; r < numRadars; ++r) {
        okvis::RadarParameters radarParams;
        radarParams.T_IR.setRandom(0.5, 0.1); // Random extrinsics for each radar
        radarParametersList.push_back(radarParams);
    }

    // counter variables for each radar (unsynchronized)
    std::vector<size_t> countRadar(numRadars, 0);
    size_t countFrames=0;
    
    // Phase offsets for unsynchronized radars (staggered start times)
    // This ensures radars don't all measure at the same time
    std::vector<double> radarPhaseOffsets = {0.0, 0.033, 0.067}; // Staggered by ~33ms

    // ========================================================================
    // INITIALIZE: Ceres problem and state variables
    // ========================================================================
    
    ::ceres::Problem problem;
    std::cout << "Setting up test case (circular movement) with \n"
              << "Duration: " << duration << " [s]\n"
              << "IMU rate: " << imuRate << " [Hz] (realistic)\n"
              << "Number of radars: " << numRadars << "\n"
              << "Radar rate: " << radarRate << " [Hz] per radar (unsynchronized)\n"
              << "Frame rate: " << frameRate << " [Hz] (realistic)\n"
              << "Verifying Jacobians and State Estimation Results..." << std::endl;

    // Set local parametrization for pose (create once, reuse for all poses)
    auto* poseLocalParameterization = new okvis::ceres::PoseManifold();

    // Initialize ground truth state
    okvis::kinematics::Transformation T_WS(p0, Eigen::Quaterniond::Identity());
    Eigen::Quaterniond q = T_WS.q();
    Eigen::Vector3d r = T_WS.r();

    // Initial velocity (tangential to circle)
    Eigen::Vector3d v_w = omegaVec.cross(r - p_c);
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setZero();
    speedAndBias.head<3>() = v_w; // Velocity in world frame

    // ========================================================================
    // SIMULATION LOOP: Generate trajectory, IMU, keyframes, and radar data
    // ========================================================================
    
    double time = 0;
    okvis::Time tkPrev(time); // Previous keyframe time
    okvis::Time tk(time);     // Current keyframe time (15Hz)
    okvis::Time tr(time);     // Current radar measurement time (10Hz)
    okvis::kinematics::Transformation T_kPrev; // Previous keyframe pose
    okvis::kinematics::Transformation T_k;     // Current keyframe pose (ground truth)
    okvis::kinematics::Transformation T_k_dist; // Disturbed keyframe pose (initial estimate)
    okvis::SpeedAndBias sb_k;      // SpeedAndBias at keyframe (ground truth)
    okvis::SpeedAndBias sb_k_dist; // Disturbed SpeedAndBias (initial estimate)

    // Storage for parameter blocks and ground truth
    std::vector< std::shared_ptr< okvis::ceres::PoseParameterBlock> > robotPoseParameterBlocks;
    std::vector< std::shared_ptr< okvis::ceres::SpeedAndBiasParameterBlock> > speedAndBiasParameterBlocks;
    std::vector< okvis::kinematics::Transformation> gtRobotPoses; // Ground truth poses
    std::vector< okvis::SpeedAndBias> gtSbs; // Ground truth speed and biases

    // Main simulation loop: iterate at IMU rate 200Hz)
    for(size_t i=0; i<size_t(duration*imuRate); ++i){

        time = double(i)/imuRate; // Current time [s]

        // ====================================================================
        // STEP 1: Propagate ground truth trajectory (circular motion)
        // ====================================================================
        
        // Compute velocity vector (tangential to circle)
        v_w = omegaVec.cross(r - p_c); // v = omega × (r - p_c)
        // Compute acceleration vector (centripetal)
        Eigen::Vector3d a_W = omegaVec.cross(v_w);

        // Propagate position
        r += v_w*dt;

        // Propagate orientation (rotation around circle)
        Eigen::Quaterniond dq;
        const double alpha_half = omegaVec.norm()*dt*0.5;
        const double sinc_alpha_half = okvis::kinematics::sinc(alpha_half);
        const double cos_alpha_half = cos(alpha_half);
        dq.vec() = sinc_alpha_half * 0.5 * omegaVec * dt;
        dq.w() = cos_alpha_half;
        q = q * dq;

        // Update ground truth pose
        T_WS = okvis::kinematics::Transformation(r,q);
        speedAndBias.head<3>() = v_w; // Update velocity

        // ====================================================================
        // STEP 2: Generate IMU measurements (with noise)
        // ====================================================================
        
        // Generate noisy gyroscope measurement
        Eigen::Vector3d gyr = omegaVec + imuParameters.sigma_g_c/sqrt(dt)*Eigen::Vector3d::Random();
        // Generate noisy accelerometer measurement (in IMU frame)
        Eigen::Vector3d acc = T_WS.inverse().C()*(a_W+Eigen::Vector3d(0,0,imuParameters.g)) 
                            + imuParameters.sigma_a_c/sqrt(dt)*Eigen::Vector3d::Random();
        imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(time),okvis::ImuSensorReadings(gyr,acc)));

        // ====================================================================
        // STEP 3: Create KEYFRAME at frameRate (15Hz)
        // ====================================================================
        // Keyframes are camera/visual frames where we estimate state
        if (time > countFrames/frameRate){

            // Save previous keyframe (for relative pose constraints)
            T_kPrev = T_k;
            tkPrev = tk;
            
            // Current keyframe state (ground truth)
            T_k = T_WS;
            tk = okvis::Time(time);
            sb_k = speedAndBias;

            // ================================================================
            // Create DISTURBED initial estimates (what optimizer starts with)
            // ================================================================
            okvis::kinematics::Transformation T_disturb;
            T_disturb.setRandom(1, M_PI); // Random pose disturbance
            T_k_dist = T_k*T_disturb; // Disturbed pose
            
            sb_k_dist = speedAndBias;
            sb_k_dist.head<3>() += v_dist_std * Eigen::Vector3d::Random(); // Disturbed velocity
            sb_k_dist.segment<3>(3) += bg_dist_std * Eigen::Vector3d::Random(); // Disturbed gyro bias
            sb_k_dist.tail<3>() += ba_dist_std * Eigen::Vector3d::Random(); // Disturbed accel bias

            // Create Ceres parameter blocks (optimization variables)
            std::shared_ptr<okvis::ceres::PoseParameterBlock> robotPoseParameterBlock(
                new okvis::ceres::PoseParameterBlock(T_k_dist,0,tk));
            robotPoseParameterBlocks.push_back(robotPoseParameterBlock);
            gtRobotPoses.push_back(T_k); // Store ground truth for verification
            
            std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(
                new okvis::ceres::SpeedAndBiasParameterBlock(sb_k_dist,0,tk));
            speedAndBiasParameterBlocks.push_back(speedAndBiasParameterBlock);
            gtSbs.push_back(sb_k); // Store ground truth for verification

            ++countFrames;
            
            // ====================================================================
            // STEP 4a: Add ADDITIONAL CONSTRAINTS when we have a keyframe
            // ====================================================================
            // These constraints are added once per keyframe (not per radar measurement)
            
            // Relative pose constraints between keyframes (odometry-like)
            if(robotPoseParameterBlocks.size() > 1){
                okvis::kinematics::Transformation T_rel;
                T_rel = T_k.inverse() * T_kPrev; // Relative transformation
                okvis::kinematics::Transformation T_rel_dist;
                T_rel_dist.setRandom(odometry_trans_std, odometry_rot_std);
                ::ceres::CostFunction* cost_function_rel = new okvis::ceres::RelativePoseError(
                    pow(odometry_trans_std,2.0), pow(odometry_rot_std,2.0), T_rel*T_rel_dist);
                problem.AddResidualBlock(cost_function_rel, NULL, 
                                         robotPoseParameterBlocks.back()->parameters(), 
                                         robotPoseParameterBlocks.at(robotPoseParameterBlocks.size()-2)->parameters());
            }
            else if(robotPoseParameterBlocks.size() == 1){
                // For first pose: add prior (weak constraint to fix gauge freedom)
                ::ceres::CostFunction* cost_function_abs = new okvis::ceres::PoseError(T_k,1e-06,1e-06);
                problem.AddResidualBlock(cost_function_abs, NULL, robotPoseParameterBlocks.back()->parameters());
            }

            // IMU error terms between keyframes (constrains velocity and biases)
            if(speedAndBiasParameterBlocks.size() > 1){
                okvis::ceres::ImuError* cost_function_imu = new okvis::ceres::ImuError(
                    imuMeasurements, imuParameters, tkPrev, tk);
                problem.AddResidualBlock(cost_function_imu, NULL,
                                         robotPoseParameterBlocks.at(robotPoseParameterBlocks.size()-2)->parameters(),
                                         speedAndBiasParameterBlocks.at(speedAndBiasParameterBlocks.size()-2)->parameters(),
                                         robotPoseParameterBlocks.back()->parameters(),
                                         speedAndBiasParameterBlocks.back()->parameters());
            }
            else if(speedAndBiasParameterBlocks.size() == 1){
                // For first state: add prior on speed and biases
                ::ceres::CostFunction* cost_function_speedAndBias = new okvis::ceres::SpeedAndBiasError(
                    sb_k,1e-06,1e-06,1e-06);
                problem.AddResidualBlock(cost_function_speedAndBias, NULL,
                                         speedAndBiasParameterBlocks.back()->parameters());
            }
        }
        
        // ====================================================================
        // STEP 4: Create RADAR MEASUREMENTS from multiple radars (10Hz each, unsynchronized)
        // ====================================================================
        // Each radar measures at 10Hz but at different times (unsynchronized)
        // This is the key test: can we estimate state at tk from radars at tr_i?
        
        // Check each radar for new measurements
        for(int radarIdx = 0; radarIdx < numRadars; ++radarIdx) {
            // Use phase offset to desynchronize radars
            double radarTime = time - radarPhaseOffsets[radarIdx];
            if(radarTime > 0 && radarTime > countRadar[radarIdx]/radarRate) {
                
                okvis::Time tr = okvis::Time(time); // Actual radar measurement time
                
                // ============================================================
                // Generate radar velocity measurement (ground truth)
                // ============================================================
                
                // Get angular velocity at radar time (from current IMU measurement)
                Eigen::Vector3d omega_S_tr = gyr; // Angular velocity in IMU frame at time tr
                
                // Compute expected radar velocity at time tr for this specific radar
                // Formula: v_R = R_RI * (R_IW_tr * v_W_tr + (ω_tr - b_g) × p_IR)
                Eigen::Matrix3d C_WS_tr = T_WS.C(); // Rotation from world to sensor at tr
                Eigen::Matrix3d C_SW_tr = C_WS_tr.transpose(); // R_IW at time tr
                Eigen::Vector3d v_I_tr = C_SW_tr * v_w; // Velocity in IMU frame at time tr
                Eigen::Vector3d b_g = speedAndBias.segment<3>(3); // Gyro bias at time tr
                Eigen::Vector3d omega_S_corrected_tr = omega_S_tr - b_g; // Corrected angular velocity
                Eigen::Vector3d p_IR_I = radarParametersList[radarIdx].T_IR.r(); // Radar position in IMU frame
                Eigen::Vector3d omega_cross_p_tr = omega_S_corrected_tr.cross(p_IR_I); // Tangential velocity
                Eigen::Matrix3d C_RI = radarParametersList[radarIdx].T_IR.C().transpose(); // Rotation IMU->Radar
                Eigen::Vector3d v_R_true = C_RI * (v_I_tr + omega_cross_p_tr); // Expected radar velocity

                // Add measurement noise
                v_R_true += radar_std * Eigen::Vector3d::Random();

                // ============================================================
                // Add RADAR ERROR TERM to optimization problem
                // ============================================================
                // This is the key: RadarErrorAsynchronous will:
                // 1. Take state at tk (from parameter blocks)
                // 2. Use IMU pre-integration to propagate from tk to tr
                // 3. Compute expected radar velocity at tr
                // 4. Compare with measured radar velocity (v_R_true)
                
                // Add parameter blocks to problem (if not already added)
                robotPoseParameterBlocks.back()->setLocalParameterizationPtr(poseLocalParameterization); 
                problem.AddParameterBlock(robotPoseParameterBlocks.back()->parameters(), 
                                         okvis::ceres::PoseParameterBlock::Dimension, 
                                         poseLocalParameterization);
                problem.SetParameterBlockVariable(robotPoseParameterBlocks.back()->parameters());
                
                problem.AddParameterBlock(speedAndBiasParameterBlocks.back()->parameters(),
                                         okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
                problem.SetParameterBlockVariable(speedAndBiasParameterBlocks.back()->parameters());
                
                // Create radar error term for this specific radar
                // Parameters: radarId, measurement, information, IMU data, times, omega, radar params
                Eigen::Matrix3d information = (1.0/(radar_std*radar_std)) * Eigen::Matrix3d::Identity();
                ::ceres::CostFunction* cost_function = new okvis::ceres::RadarErrorAsynchronous(
                    radarIdx + 1, // Radar ID (1, 2, or 3)
                    v_R_true, information, imuMeasurements, imuParameters, 
                    tk, tr, omega_S_tr, radarParametersList[radarIdx]);
                
                // Add residual block: connects radar measurement to state at tk
                // The error term handles the tk->tr propagation internally
                problem.AddResidualBlock(cost_function, NULL, 
                                         robotPoseParameterBlocks.back()->parameters(), 
                                         speedAndBiasParameterBlocks.back()->parameters());
                
                ++countRadar[radarIdx];
                
                // ====================================================================
                // STEP 5: VERIFY JACOBIANS (only for first radar measurement to avoid redundancy)
                // ====================================================================
                // This ensures the analytical Jacobians are correct
                if(radarIdx == 0 && countRadar[0] == 1){

                double* parameters[2];
                parameters[0]=robotPoseParameterBlocks.back()->parameters();
                parameters[1]=speedAndBiasParameterBlocks.back()->parameters();
                double* jacobians[2];
                Eigen::Matrix<double,3,7,Eigen::RowMajor> J0;
                Eigen::Matrix<double,3,9,Eigen::RowMajor> J1;
                jacobians[0]=J0.data();
                jacobians[1]=J1.data();
                Eigen::Matrix<double,3,1> residuals;
                // evaluate twice to be sure that we will be using the linearisation of the biases (i.e. no preintegrals redone)
                static_cast<okvis::ceres::RadarErrorAsynchronous*>(cost_function)->EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,NULL);
                static_cast<okvis::ceres::RadarErrorAsynchronous*>(cost_function)->EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,NULL);

                // and now num-diff:
                double dx=1e-6;

                // w.r.t. robot pose
                Eigen::Matrix<double,3,6> J0_numDiff;
                for(size_t i=0; i<6; ++i){
                  Eigen::Matrix<double,6,1> dp_0;
                  Eigen::Matrix<double,3,1> residuals_p;
                  Eigen::Matrix<double,3,1> residuals_m;
                  dp_0.setZero();
                  dp_0[i]=dx;
                  poseLocalParameterization->Plus(parameters[0],dp_0.data(),parameters[0]);
                  static_cast<okvis::ceres::RadarErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_p.data(),NULL);
                  robotPoseParameterBlocks.back()->setEstimate(T_k_dist); // reset
                  dp_0[i]=-dx;
                  poseLocalParameterization->Plus(parameters[0],dp_0.data(),parameters[0]);
                  static_cast<okvis::ceres::RadarErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_m.data(),NULL);
                  robotPoseParameterBlocks.back()->setEstimate(T_k_dist); // reset
                  J0_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));

                }

                // Use lift Jacobian for non-minimal Jacobian
                Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J0_numDiff_lift;
                Eigen::Matrix<double, 6, 7, Eigen::RowMajor> liftJac0;
                okvis::ceres::PoseManifold::minusJacobian(parameters[0], liftJac0.data());
                J0_numDiff_lift = J0_numDiff * liftJac0;

                EXPECT_TRUE((J0_numDiff_lift-J0).norm() < jacobianThresh) << " Jacobian Evaluation leads error  " << (J0_numDiff_lift-J0).norm() << " > " << jacobianThresh << std::endl;

                // w.r.t. SpeedAndBias
                Eigen::Matrix<double,3,9> J1_numDiff;
                for(size_t i=0; i<9; ++i){
                  Eigen::Matrix<double,9,1> dp_1;
                  Eigen::Matrix<double,3,1> residuals_p;
                  Eigen::Matrix<double,3,1> residuals_m;
                  dp_1.setZero();
                  dp_1[i]=dx;
                  speedAndBiasParameterBlocks.back()->plus(parameters[1],dp_1.data(),parameters[1]);
                  static_cast<okvis::ceres::RadarErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_p.data(),NULL);
                  speedAndBiasParameterBlocks.back()->setEstimate(sb_k_dist); // reset
                  dp_1[i]=-dx;
                  speedAndBiasParameterBlocks.back()->plus(parameters[1],dp_1.data(),parameters[1]);
                  static_cast<okvis::ceres::RadarErrorAsynchronous*>(cost_function)->Evaluate(parameters,residuals_m.data(),NULL);
                  speedAndBiasParameterBlocks.back()->setEstimate(sb_k_dist); // reset
                  J1_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));

                }

//                std::cout << "Jacobian evaluates to: \n" << J1 << std::endl;
//                std::cout << "Numerical Jacobian evaluates to: \n" << J1_numDiff << std::endl;

                EXPECT_TRUE((J1_numDiff-J1).norm() < jacobianThresh) << " Jacobian Evaluation leads error  " << (J1_numDiff-J1).norm() << " > " << jacobianThresh << std::endl;

                } // End Jacobian verification (only for first radar)
                } // End if for this radar measurement
            } // End radar loop

      } // End simulation loop

    // ========================================================================
    // OPTIMIZATION: Solve the problem
    // ========================================================================
    std::cout << "Running Ceres solver... " << std::endl;

    ::ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ::FLAGS_stderrthreshold=google::WARNING; // Enable console warnings
    ::ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // ========================================================================
    // VERIFICATION: Check if estimated states match ground truth
    // ========================================================================
    
    // Verify estimated poses match ground truth
    for(size_t j = 0; j<gtRobotPoses.size(); j++){
        EXPECT_TRUE((gtRobotPoses.at(j).r() - robotPoseParameterBlocks.at(j)->estimate().r()).norm() < posThresh)
                << " Estimated robot position has an error   "
                << (gtRobotPoses.at(j).r() - robotPoseParameterBlocks.at(j)->estimate().r()).norm()
                << " > " << posThresh << std::endl;
        EXPECT_TRUE(2*(gtRobotPoses.at(j).q() * robotPoseParameterBlocks.at(j)->estimate().q().inverse()).vec().norm() < rotThresh)
                << " Estimated robot orientation has an error   "
                << 2*(gtRobotPoses.at(j).q() * robotPoseParameterBlocks.at(j)->estimate().q().inverse()).vec().norm()
                << " > " << rotThresh << std::endl;
    }
    // Verify correctness of estimated speed and biases
    for(size_t jj = 0; jj<speedAndBiasParameterBlocks.size(); jj++){
        double velocityError = (gtSbs.at(jj).head<3>() - speedAndBiasParameterBlocks.at(jj)->estimate().head<3>()).norm();
        double bgError = (gtSbs.at(jj).segment<3>(3) - speedAndBiasParameterBlocks.at(jj)->estimate().segment<3>(3)).norm();
        double baError = (gtSbs.at(jj).tail<3>() - speedAndBiasParameterBlocks.at(jj)->estimate().tail<3>()).norm();
        EXPECT_TRUE(velocityError < velThresh)
                << " Velocity error of   " << velocityError << " > " << velThresh << std::endl;
        EXPECT_TRUE(bgError < bgThresh)
                << " Gyr bias error of   " << bgError << " > " << bgThresh << std::endl;
        EXPECT_TRUE(baError <baThresh)
                << " Acc bias error of   " << baError << " > " << baThresh << std::endl;
    }

}

TEST(okvisTestSuite, RadarErrorAsynchronousInternalVerification) {
    // 1. Setup Random State at keyframe time tk
    okvis::kinematics::Transformation T_WS_tk;
    T_WS_tk.setRandom(10.0, M_PI);
    
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setRandom();
    
    okvis::RadarParameters radarParameters;
    radarParameters.T_IR.setRandom(0.5, 0.1);
    
    // 2. Setup times: keyframe at tk, radar measurement at tr (after tk)
    okvis::Time tk(0.0);
    okvis::Time tr(0.1); // Radar measurement 100ms after keyframe
    
    // 3. Setup IMU parameters
    okvis::ImuParameters imuParameters;
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;
    
    // 4. Generate IMU measurements between tk and tr
    // This simulates realistic IMU data needed for pre-integration
    okvis::ImuMeasurementDeque imuMeasurements;
    const double imuRate = 200.0; // 200 Hz IMU rate
    const double dt = 1.0 / imuRate;
    
    // Generate IMU measurements with some motion
    Eigen::Vector3d omega_base(0.1, 0.05, -0.08); // Base angular velocity
    Eigen::Vector3d acc_base(0.5, -0.3, 9.81); // Base acceleration (with gravity)
    
    for (double t = tk.toSec(); t < tr.toSec(); t += dt) {
        // Add some variation to make it realistic
        Eigen::Vector3d omega = omega_base + 0.01 * Eigen::Vector3d::Random();
        Eigen::Vector3d acc = acc_base + 0.1 * Eigen::Vector3d::Random();
        
        imuMeasurements.push_back(
            okvis::ImuMeasurement(
                okvis::Time(t),
                okvis::ImuSensorReadings(omega, acc)
            )
        );
    }
    
    // 5. Angular velocity at radar measurement time tr
    Eigen::Vector3d omega_S_tr = imuMeasurements.back().measurement.gyroscopes;
    
    // 6. Random measurement - exact value does not matter for Jacobian test
    Eigen::Vector3d measurement = Eigen::Vector3d::Random();
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
    
    // 7. Instantiate Error Term
    okvis::ceres::RadarErrorAsynchronous radarError(
        1, measurement, information, imuMeasurements, imuParameters, 
        tk, tr, omega_S_tr, radarParameters);
    
    // 8. Prepare parameters - SAFELY
    // Create a local buffer and fill it manually to avoid the "cached mode" error
    Eigen::Matrix<double, 7, 1> T_WS_tk_params_buffer;
    T_WS_tk_params_buffer.head<3>() = T_WS_tk.r();
    T_WS_tk_params_buffer[3] = T_WS_tk.q().x();
    T_WS_tk_params_buffer[4] = T_WS_tk.q().y();
    T_WS_tk_params_buffer[5] = T_WS_tk.q().z();
    T_WS_tk_params_buffer[6] = T_WS_tk.q().w();
    
    // Create a buffer for speedAndBias
    Eigen::Matrix<double, 9, 1> speedAndBias_buffer = speedAndBias;
    
    double* parameters[2] = { T_WS_tk_params_buffer.data(), speedAndBias_buffer.data() };
    
    // 9. Compute Analytical Jacobians 
    double residuals[3];
    double j0_data[3 * 7], j1_data[3 * 9];
    double* jacobians[2] = { j0_data, j1_data };
    
    // Fill analytical Jacobians 
    radarError.EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
    
    // 10. CALL INTERNAL VERIFIER
    // This will verify analytical Jacobians against numerical differentiation
    bool success = radarError.VerifyJacobianNumDiff(parameters, jacobians);
    
    EXPECT_TRUE(success) << "RadarErrorAsynchronous Jacobian verification failed! Check console for output.";
}
