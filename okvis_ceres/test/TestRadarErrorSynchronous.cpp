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
#include <gtest/gtest.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/assert_macros.hpp>

#include <okvis/ceres/RadarErrorSynchronous.hpp>

// Rotation-only observability: velocity/bias fixed, solve rotation
TEST(okvisTestSuite, RadarErrorRotationOnly){
    ::ceres::Problem problem;

    // Ground-truth state
    okvis::RadarParameters radarParameters;
    radarParameters.T_IR.setRandom(0.5, 0.1);
    okvis::kinematics::Transformation T_WS;
    T_WS.setRandom(10.0, M_PI);
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setRandom();

    // Measurement generation
    Eigen::Vector3d omega_S = Eigen::Vector3d::Random();
    Eigen::Matrix3d C_WS = T_WS.C();
    Eigen::Matrix3d C_SW = C_WS.transpose();
    Eigen::Vector3d v_W = speedAndBias.head<3>();
    Eigen::Vector3d b_g = speedAndBias.segment<3>(3);
    Eigen::Vector3d v_I = C_SW * v_W;
    Eigen::Vector3d omega_S_corrected = omega_S - b_g;
    Eigen::Vector3d p_IR_I = radarParameters.T_IR.r();
    Eigen::Vector3d omega_cross_p = omega_S_corrected.cross(p_IR_I);
    Eigen::Matrix3d C_RI = radarParameters.T_IR.C().transpose();
    Eigen::Vector3d v_R_true = C_RI * (v_I + omega_cross_p);

    // Parameter blocks
    okvis::kinematics::Transformation T_disturb;
    T_disturb.setRandom(1, 0.01);
    okvis::kinematics::Transformation T_WS_init = T_disturb * T_WS; // disturbed rotation
    okvis::ceres::PoseParameterBlock robotPoseParameterBlock(T_WS_init, 1, okvis::Time(0));
    problem.AddParameterBlock(robotPoseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
    problem.SetParameterBlockVariable(robotPoseParameterBlock.parameters());
    okvis::ceres::PoseManifold* poseManifold = new okvis::ceres::PoseManifold;
    problem.SetManifold(robotPoseParameterBlock.parameters(), poseManifold);

    okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock(speedAndBias, 2, okvis::Time(0));
    problem.AddParameterBlock(speedAndBiasParameterBlock.parameters(), okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
    problem.SetParameterBlockConstant(speedAndBiasParameterBlock.parameters()); // fix velocity and biases

    // Residual
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
    ::ceres::CostFunction* cost_function = new okvis::ceres::RadarErrorSynchronous(
        1, v_R_true, information, radarParameters, omega_S);
    problem.AddResidualBlock(cost_function, NULL,
                             robotPoseParameterBlock.parameters(),      // ← parameters[0]
                             speedAndBiasParameterBlock.parameters());  // ← parameters[1]

    // Solve
    ::ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ::ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Check rotation convergence (position unobservable, ignored)
    double rotation_error = 2*(T_WS.q()*robotPoseParameterBlock.estimate().q().inverse()).vec().norm();
    if (rotation_error >= 1e-2) {
        std::cout << "Rotation-only debug\n"
                  << "  q_WS_gt   : " << T_WS.q().coeffs().transpose() << "\n"
                  << "  q_WS_init : " << T_WS_init.q().coeffs().transpose() << "\n"
                  << "  q_WS_est  : " << robotPoseParameterBlock.estimate().q().coeffs().transpose() << "\n"
                  << "  err_norm  : " << rotation_error << std::endl;
    }
    EXPECT_TRUE(rotation_error < 1e-2) << "Rotation error: " << rotation_error;
}

// Velocity-only observability: pose fixed, bias zeroed, solve velocity
TEST(okvisTestSuite, RadarErrorVelocityOnly){
    ::ceres::Problem problem;

    okvis::RadarParameters radarParameters;
    radarParameters.T_IR.setRandom(0.5, 0.1);
    okvis::kinematics::Transformation T_WS;
    T_WS.setRandom(10.0, M_PI);

    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setZero();
    speedAndBias.head<3>() = Eigen::Vector3d::Random(); // true velocity
    Eigen::Vector3d omega_S = Eigen::Vector3d::Zero();  // remove bias coupling

    // Measurement generation (bias zero, omega zero -> only velocity term)
    Eigen::Matrix3d C_WS = T_WS.C();
    Eigen::Matrix3d C_SW = C_WS.transpose();
    Eigen::Vector3d v_I = C_SW * speedAndBias.head<3>();
    Eigen::Matrix3d C_RI = radarParameters.T_IR.C().transpose();
    Eigen::Vector3d v_R_true = C_RI * v_I;

    // Parameter blocks
    okvis::ceres::PoseParameterBlock robotPoseParameterBlock(T_WS, 1, okvis::Time(0));
    problem.AddParameterBlock(robotPoseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
    problem.SetParameterBlockConstant(robotPoseParameterBlock.parameters()); // fix pose
    okvis::ceres::PoseManifold* poseManifold = new okvis::ceres::PoseManifold;
    problem.SetManifold(robotPoseParameterBlock.parameters(), poseManifold);

    okvis::SpeedAndBias speedAndBias_init = speedAndBias;
    speedAndBias_init.head<3>() += Eigen::Vector3d(0.2, -0.1, 0.05); // disturb velocity
    okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock(speedAndBias_init, 2, okvis::Time(0));
    problem.AddParameterBlock(speedAndBiasParameterBlock.parameters(), okvis::ceres::SpeedAndBiasParameterBlock::Dimension);
    problem.SetParameterBlockVariable(speedAndBiasParameterBlock.parameters());

    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
    // Three radars with varying extrinsics added via loop to reduce redundancy
    for (int i = 0; i < 3; ++i) {
        okvis::RadarParameters params;
        params.T_IR.setRandom(0.5, 0.1);
        Eigen::Vector3d v_R = params.T_IR.C().transpose() * v_I;
        problem.AddResidualBlock(
            new okvis::ceres::RadarErrorSynchronous(i + 1, v_R, information, params, omega_S),
            NULL,
            robotPoseParameterBlock.parameters(),
            speedAndBiasParameterBlock.parameters());
    }

    // Solve
    ::ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ::ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Check velocity convergence (bias stays near zero due to zero gradient)
    double velocity_error = (speedAndBias.head<3>() - speedAndBiasParameterBlock.estimate().head<3>()).norm();
    if (velocity_error >= 1e-2) {
        std::cout << "Velocity-only debug\n"
                  << "  v_W_gt      : " << speedAndBias.head<3>().transpose() << "\n"
                  << "  v_W_init    : " << speedAndBias_init.head<3>().transpose() << "\n"
                  << "  v_W_est     : " << speedAndBiasParameterBlock.estimate().head<3>().transpose() << "\n"
                  << "  v_R_true    : " << v_R_true.transpose() << "\n"
                  << "  v_R_residual: " << (speedAndBiasParameterBlock.estimate().head<3>() - speedAndBias.head<3>()).transpose() << "\n"
                  << "  err_norm    : " << velocity_error << std::endl;
    }
    EXPECT_TRUE(velocity_error < 1e-2) << "Velocity error: " << velocity_error;
}

// Gyro-bias-only observability: pose fixed, velocity zero, solve bias via omega x p term
TEST(okvisTestSuite, RadarErrorGyroBiasOnly) {
    ::ceres::Problem problem;

    // 1. Setup Ground Truth
    okvis::kinematics::Transformation T_WS;
    T_WS.setRandom(10.0, M_PI);
    
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setZero();
    Eigen::Vector3d b_g_gt(0.02, -0.01, 0.015);
    speedAndBias.segment<3>(3) = b_g_gt;

    // 2. Parameter Blocks
    okvis::ceres::PoseParameterBlock robotPoseParameterBlock(T_WS, 1, okvis::Time(0));
    problem.AddParameterBlock(robotPoseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
    problem.SetParameterBlockConstant(robotPoseParameterBlock.parameters());

    okvis::SpeedAndBias speedAndBias_init = speedAndBias;
    speedAndBias_init.segment<3>(3) += Eigen::Vector3d(0.05, 0.05, -0.03); // Störung
    okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock(speedAndBias_init, 2, okvis::Time(0));
    problem.AddParameterBlock(speedAndBiasParameterBlock.parameters(), okvis::ceres::SpeedAndBiasParameterBlock::Dimension);

    // 3. Szenarien für Observability (verschiedene omegas)
    std::vector<Eigen::Vector3d> omegas = {
        Eigen::Vector3d(0.3, -0.2, 0.1),
        Eigen::Vector3d(0.1, 0.3, -0.15),
        Eigen::Vector3d(-0.2, 0.15, 0.25)
    };

    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();

    // 4. Residuen in Schleife hinzufügen
    for (size_t i = 0; i < omegas.size(); ++i) {
        okvis::RadarParameters params;
        params.T_IR.setRandom(0.5, 0.1); // Jedes Residuum kriegt einen zufälligen Hebelarm
        
        // Messung basierend auf Ground Truth Bias generieren
        Eigen::Vector3d omega_corr = omegas[i] - b_g_gt;
        Eigen::Vector3d v_R_true = params.T_IR.C().transpose() * omega_corr.cross(params.T_IR.r());

        problem.AddResidualBlock(
            new okvis::ceres::RadarErrorSynchronous(static_cast<int>(i) + 1, v_R_true, information, params, omegas[i]),
            nullptr, 
            robotPoseParameterBlock.parameters(), 
            speedAndBiasParameterBlock.parameters()
        );
    }

    // 5. Solve & Verify
    ::ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ::ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    double bias_error = (b_g_gt - speedAndBiasParameterBlock.estimate().segment<3>(3)).norm();
    EXPECT_LT(bias_error, 1e-3) << "Gyro bias error too high: " << bias_error;
}

TEST(okvisTestSuite, RadarErrorInternalVerification) {
    // 1. Setup Random State
    okvis::kinematics::Transformation T_WS;
    T_WS.setRandom(10.0, M_PI);
    
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setRandom();
    
    okvis::RadarParameters radarParameters;
    radarParameters.T_IR.setRandom(0.5, 0.1);
    
    Eigen::Vector3d omega_S = Eigen::Vector3d::Random();
    Eigen::Vector3d measurement = Eigen::Vector3d::Random();
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();

    // 2. Instantiate Error Term
    okvis::ceres::RadarErrorSynchronous radarError(
        1, measurement, information, radarParameters, omega_S);

    // 3. Prepare parameters - SAFELY
    // Create a local buffer and fill it manually to avoid the "cached mode" error
    Eigen::Matrix<double, 7, 1> T_WS_params_buffer;
    T_WS_params_buffer.head<3>() = T_WS.r();
    T_WS_params_buffer[3] = T_WS.q().x();
    T_WS_params_buffer[4] = T_WS.q().y();
    T_WS_params_buffer[5] = T_WS.q().z();
    T_WS_params_buffer[6] = T_WS.q().w();

    double* parameters[2] = { T_WS_params_buffer.data(), speedAndBias.data() };

    // 4. Compute Analytical Jacobians 
    double residuals[3];
    double j0_data[3 * 7], j1_data[3 * 9];
    double* jacobians[2] = { j0_data, j1_data };
    
    // Fill analytical Jacobians
    radarError.Evaluate(parameters, residuals, jacobians);

    // 5. CALL YOUR INTERNAL VERIFIER
    // This will now work because 'parameters' is just a raw pointer to our safe buffer
    bool success = radarError.VerifyJacobianNumDiff(parameters, jacobians);
    
    EXPECT_TRUE(success) << "Jacobian verification failed! Check console for LOG(WARNING) output.";
}