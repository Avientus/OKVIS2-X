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
 * @file ceres/AltimeterErrorAsynchronous.hpp
 * @brief Header file for the asynchronous AltimeterError class.
 */

#ifndef INCLUDE_OKVIS_CERES_ALTIMETERERRORASYNCHRONOUS_HPP_
#define INCLUDE_OKVIS_CERES_ALTIMETERERRORASYNCHRONOUS_HPP_

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <ceres/sized_cost_function.h>
#include <ceres/covariance.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
namespace ceres {

/// \brief Implements a nonlinear vertical-velocity factor derived from a downward
///        rangefinder, with IMU pre-integration bridging state time to measurement time.
///
/// The measurement is NOT an absolute height: it is a vertical rate (world frame,
/// positive up), obtained on the ROS side from a least-squares slope fit through a
/// window of raw distance-to-ground samples. This makes the factor relative -- it
/// needs no known ground level and no sensor lever arm -- and it only ever
/// constrains velocity, so it continuously arrests vertical drift rather than
/// claiming to know the true altitude.
class AltimeterErrorAsynchronous :
    public ::ceres::SizedCostFunction<1 /* number of residuals */,
        7 /* size of first parameter (RobotPoseParameterBlock T_WS at t=k ) */,
        9 /* size of second parameter (SpeedAndBiasParameterBlock at t=k)*/>,
    public ErrorInterface {

 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static std::atomic_bool redoPropagationAlways;
  static std::atomic_bool useImuCovariance;

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<1, 7, 9> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 1;

  /// \brief The type of the measurement (vertical velocity, world frame) [m/s].
  typedef double measurement_t;

  /// \brief The type of the measurement variance [(m/s)^2].
  typedef double covariance_t;

  /// \brief The type of the information (inverse variance).
  typedef double information_t;

  /// \brief Default constructor.
  AltimeterErrorAsynchronous();

  /// \brief Construct with measurement and information.
  /// @param[in] measurement The vertical velocity measurement (world frame, positive up) at time tr.
  /// @param[in] information The information (inverse variance) of the measurement.
  /// @param[in] imuMeasurements Queue containing IMU measurements
  /// @param[in] imuParameters IMU sensor parameters
  /// @param[in] tk Timestamp of previous state / camera frame
  /// @param[in] tr Timestamp of the altimeter measurement
  AltimeterErrorAsynchronous(const measurement_t & measurement, const information_t & information,
                             const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::ImuParameters & imuParameters,
                             const okvis::Time& tk, const okvis::Time& tr);

  /// \brief Trivial destructor.
  virtual ~AltimeterErrorAsynchronous()
  {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight).
  virtual void setInformation(const information_t& information);

  /// \brief Set the time.
  /// @param[in] tr The timestamp of the altimeter measurement.
  void setTr(const okvis::Time& tr)
  {
    tr_ = tr;
  }

  /// \brief Set the time.
  /// @param[in] tk The timestamp of the latest camera frame.
  void setTk(const okvis::Time& tk)
  {
    tk_ = tk;
  }

  /// \brief (Re)set the parameters.
  /// \@param[in] imuParameters The parameters to be used.
  void setImuParameters(const okvis::ImuParameters& imuParameters){
      imuParameters_ = imuParameters;
  }

  /// \brief (Re)set the measurements
  /// \@param[in] imuMeasurements All the IMU measurements.
  void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  // getters

  /// \brief Get the measurement.
  /// \return The measurement.
  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  /// \brief Get the information.
  /// \return The information (weight).
  virtual const information_t& information() const
  {
    return altimeterInformation_;
  }

  /// \brief Get the covariance (variance).
  /// \return The inverse information (variance).
  virtual const covariance_t& covariance() const
  {
    return altimeterCovariance_;
  }

  /// \brief Get the time.
  /// \return The timestamp of the altimeter measurement
  okvis::Time tr() const
  {
    return tr_;
  }

  /// \brief Get the time.
  /// \return The timestamp of the latest corresponding camera frame
  okvis::Time tk() const
  {
    return tk_;
  }

  /// \brief Get the IMU Parameters.
  /// \return the IMU parameters.
  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  /// \brief Get the IMU measurements.
  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  /// \brief Get unweighted error.
  double error() const {
      return error_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of the evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements using pre-integration scheme.
   * @warning This is not actually const, since the re-propagation must somehow be stored...
   * @param[in] T_WS Start pose.
   * @param[in] speedAndBiases Start speed and biases.
   * @return Number of integration steps.
   */
  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  int parameterBlockDim(int parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "AltimeterErrorAsynchronous";
  }

 protected:

  // the measurement
  measurement_t measurement_ = 0.0; ///< The vertical velocity measurement (world frame) at time tr.

  // the error
  mutable double error_ = 0.0; ///< The unweighted error

  // Altimeter information / covariance
  information_t altimeterInformation_ = 0.0; ///< The information (inverse variance).
  covariance_t altimeterCovariance_ = 0.0; ///< The variance.

  // IMU parameters
  okvis::ImuParameters imuParameters_; ///< The IMU parameters.
  // IMU measurements
  okvis::ImuMeasurementDeque imuMeasurements_; ///< The IMU measurements used. Must be spanning tk_ - tr_.

  // times
  okvis::Time tk_; ///< The start time (i.e. time of the first set of states).
  okvis::Time tr_; ///< The end time (i.e. time of the altimeter measurement).

  // ----- preintegration stuff -----
  // the mutable is a TERRIBLE HACK, but what can I do (same pattern as RadarErrorAsynchronous).

  mutable std::mutex preintegrationMutex_; //< Protect access of intermediate results.
  // increments (initialise with identity)
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // cross matrix accumulatrion
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  /// \brief The Jacobian of the increment (w/o biases).
  mutable Eigen::Matrix<double,15,15> P_delta_ = Eigen::Matrix<double,15,15>::Zero();

  // for gradient/hessian w.r.t. the sigmas:
  mutable AlignedVector<Eigen::Matrix<double,15,15>> dPdsigma_;

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true; ///< Keeps track of whether or not this redoPreintegration() needs to be called.
  mutable int redoCounter_ = 0; ///< Counts the number of preintegrations for statistics.
  // ----- preintegration stuff end -----

  mutable double squareRootInformation_ = 0.0; ///< The overall square root information for this error term (altimeter + IMU covariances).

};

}  // namespace ceres
}  // namespace okvis

#endif // INCLUDE_OKVIS_CERES_ALTIMETERERRORASYNCHRONOUS_HPP_
