#include "ukf.h"
#include "Eigen/Dense"

#include<iostream>
#include<stdexcept>

#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // predicted Sigma points (re-use for Radar updates)
  MatrixXd Xsig_pred_ = MatrixXd(15, 5);

  /* 
   * Process noise standard deviation longitudinal acceleration in m/s^2
   * According to: https://www.researchgate.net/figure/Summary-of-gradient-speed-and-acceleration-data_tbl1_223922575
   * the max acceleration of a male is 0.71 m/s**2
   * As a rule of thumb, we use the half of that.
   */
  std_a_ = 0.3505;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Derived via method from class room
  std_yawdd_ = 0.6;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  //Initialization if needed
  std::cout << "ProcessMeasurement called" << std::endl;
  if(!is_initialized_){
    std::cout << "processing initialization" << std::endl;

    // save timestamp
    time_us_ = meas_package.timestamp_;

    if(meas_package.sensor_type_  == MeasurementPackage::LASER) {
        // init based on a Laser measurement
        float px = meas_package.raw_measurements_(0);
        float py = meas_package.raw_measurements_(1);

        x_(0) = px;
        x_(1) = py;

        P_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 1, 0,
              0, 0, 0, 0, 1;

    } else if (meas_package.sensor_type_  == MeasurementPackage::RADAR) {
        // init based on a Radar measurement
        float rho = meas_package.raw_measurements_(0);
        float yaw = meas_package.raw_measurements_(1);
        float rhod = meas_package.raw_measurements_(2);

        float px = cos(yaw) * rho;
        float py = sin(yaw) * rho;
        float v = rhod;

        x_(0) = px;
        x_(1) = py;
        x_(2) = v;

        P_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0,
              0, 0, 1, 0, 0,
              0, 0, 0, 1, 0,
              0, 0, 0, 0, 1;

    } else {
      // invalid measurement - raise exception
      throw std::invalid_argument( "Invalid masurement type given." );
    }
    is_initialized_ = true;
    std::cout << "x_ after init = " << std::endl << x_ << std::endl;
    std::cout << "P_ after init = " << std::endl << P_ << std::endl;
    return;
  }

  // Compute time since last measurement and store
  float delta_t = meas_package.timestamp_ - time_us_;
  delta_t = delta_t/1000000.;
  time_us_ = meas_package.timestamp_;

  // Prediction for time of delta_t
  Prediction(delta_t);

  // Process an incomming measurement
  if(meas_package.sensor_type_  == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_  == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
  } else {
      throw std::invalid_argument( "Invalid masurement type given." );
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // Generate Sigma points
  MatrixXd Xsig_aug = MatrixXd(7, 15);
  GenerateSigmaPoints(&Xsig_aug);

  // Predict Sigma points and store in Xsig_pred_
  SigmaPointPrediction(Xsig_aug, delta_t);
  
  // Estimate a Gaussian based on predicted Sigma points
  // This method directly updates x and P of the UKF.
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    // number of features of a radar
    int n_z_ = 2;

    // create example matrix with predicted sigma points
    VectorXd z_pred = VectorXd(n_z_);

    // create example matrix for predicted measurement covariance
    MatrixXd S = MatrixXd(n_z_,n_z_);

    // create example matrix with sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

    PredictLaserMeasurement(&z_pred, &S, Zsig);

    // Update P_ and z_
    UpdateState(Zsig, z_pred, S, meas_package.raw_measurements_);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // number of features of a radar
  int n_z_ = 3;
  
  // create example matrix with predicted sigma points
  VectorXd z_pred = VectorXd(n_z_);

  // create example matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z_,n_z_);

  // create example matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  PredictRadarMeasurement(&z_pred, &S, Zsig);

  // Update P_ and z_
  UpdateState(Zsig, z_pred, S, meas_package.raw_measurements_);
}

/**
 * Generate augmented sigma points
 */
void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.fill(0.0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
   x_aug.head(5) = x_;
   x_aug(5) = 0; //0 mean for this noise process
   x_aug(6) = 0; //0 mean for this noise process

  // create augmented covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_*std_a_, 0,
        0,  std_yawdd_*std_yawdd_;
  
   P_aug.topLeftCorner(n_x_, n_x_) = P_;
   P_aug.bottomRightCorner(2,2) = Q;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  
  float factor = sqrt(lambda_ + n_aug_);
  for(int i = 1; i <= n_aug_; i++) {
      Xsig_aug.col(i) = x_aug + factor * A.col(i-1);
      Xsig_aug.col(i+n_aug_) = x_aug - factor * A.col(i-1);
  }

  // print result
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd &Xsig_aug, double delta_t) {

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // predict sigma points
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
      
      float p_x = Xsig_aug(0, i);
      float p_y = Xsig_aug(1, i);
      float v = Xsig_aug(2, i);
      float phi = Xsig_aug(3, i);
      float phi_dot = Xsig_aug(4, i);
      float nu_a = Xsig_aug(5, i);
      float nu_yawdd = Xsig_aug(6, i);
      
      // set p_x and p_y
      if(fabs(phi_dot) > 0.001) {
          Xsig_pred(0, i) = p_x + (v/phi_dot) * (sin(phi + phi_dot * delta_t) - sin(phi));
          Xsig_pred(1, i) = p_y + (v/phi_dot) * (-cos(phi + phi_dot * delta_t) + cos(phi));
          std::cout << "Xsig_pred(0, i)" << Xsig_pred(0, i) << std::endl;
          std::cout << "Xsig_pred(1, i)" << Xsig_pred(1, i) << std::endl;
      } else {
          // avoid division by zero
          Xsig_pred(0, i) = p_x + v * cos(phi)*delta_t;
          Xsig_pred(1, i) = p_y + v * sin(phi)*delta_t;
          std::cout << "Xsig_pred(0, i)" << Xsig_pred(0, i) << std::endl;
          std::cout << "Xsig_pred(1, i)" << Xsig_pred(1, i) << std::endl;
      }

      
      // set other values
      Xsig_pred(2, i) = v;
      Xsig_pred(3, i) = phi + phi_dot * delta_t;
      Xsig_pred(4, i) = phi_dot;
      
      // add measurement noise
      float delta_t2 = delta_t * delta_t;
      Xsig_pred(0, i) += 0.5 * delta_t2 * cos(phi) * nu_a;
      Xsig_pred(1, i) += 0.5 * delta_t2 * sin(phi) * nu_a;
      Xsig_pred(2, i) += delta_t * nu_a;
      Xsig_pred(3, i) += 0.5 * delta_t2 * nu_yawdd;
      Xsig_pred(4, i) += delta_t * nu_yawdd;
      
  }
  // write predicted sigma points into right column

  // print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  // write result
  Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {

  // create vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
  
  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);
   
   // set weights
   for (int i = 0; i < 2*n_aug_+1; i++) {
       if(i == 0) {
           weights(0) = lambda_ / (lambda_ + n_aug_);
       } else {
           weights(i) = 1 / (2*(lambda_ + n_aug_));
       }
   }
  
  // predict state mean
  for (int i = 0; i < 2*n_aug_+1; i++){
      x += weights(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  for (int i = 0; i < 2*n_aug_+1; i++){
      VectorXd diff = Xsig_pred_.col(i) - x;

      //the yaw component in diff might be broken - fix!
      while (diff(3) > M_PI) diff(3) -= 2. * M_PI;
      while (diff(3)< -M_PI) diff(3) += 2. * M_PI;

      P += weights(i) * diff * diff.transpose();
  }

  // print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P << std::endl;

  // write result
  x_ = x;
  P_ = P;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd& Zsig) {

  // set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights(i) = weight;
  }

  // number of radar measurement features
  int n_z_ = 3;

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++){
      float p_x = Xsig_pred_(0, i);
      float p_y = Xsig_pred_(1, i);
      float v = Xsig_pred_(2, i);
      float yaw = Xsig_pred_(3, i);
      //Note: yawd not needed here
      
      Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
      Zsig(1, i) = atan2(p_y , p_x); 
      Zsig(2, i) = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / (sqrt(p_x * p_x + p_y * p_y));

  }
  
  // calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_pred += weights(i) * Zsig.col(i);
  }
  std::cout << "z_pred (temp 1): " << std::endl << z_pred << std::endl;
  //the angle component in diff might be broken - fix!
  while (z_pred(1) > M_PI) z_pred(1) -= 2. * M_PI;
  while (z_pred(1)< -M_PI) z_pred(1) += 2. * M_PI;
  
  // calculate innovation covariance matrix S
  for (int i = 0; i < 2*n_aug_ + 1; i++) {
      VectorXd diff = Zsig.col(i) - z_pred;

      //the angle component in diff might be broken - fix!
      while (diff(1) > M_PI) diff(1) -= 2. * M_PI;
      while (diff(1)< -M_PI) diff(1) += 2. * M_PI;

      S += weights(i) * diff * diff.transpose();
  }
  
  // add linear measurement noise
  // Todo make class attribute
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;
  
  S += R;

  // print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

  // write result
  *z_out = z_pred;
  *S_out = S;
}

void UKF::PredictLaserMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd& Zsig){
  // set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights(i) = weight;
  }

  // number of laser measurement features
  int n_z_ = 2;

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++){
      float p_x = Xsig_pred_(0, i);
      float p_y = Xsig_pred_(1, i);
      //Note: yawd not needed here
      
      Zsig(0, i) = p_x;
      Zsig(1, i) = p_y;
  }
  
  // calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_pred += weights(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  for (int i = 0; i < 2*n_aug_ + 1; i++) {
      VectorXd diff = Zsig.col(i) - z_pred;
      S += weights(i) * diff * diff.transpose();
  }
  
  // add linear measurement noise for Laser
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R << std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  
  S += R;

  // print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

  // write result
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateState(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S, VectorXd& z) {

  // set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights(i) = weight;
  }

  // set measurement dimension, radar can measure r, phi, and r_dot
  //int n_z_ = 3;
  int n_z_ = z_pred.size();

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd diff1 = Xsig_pred_.col(i) - x_;
      // angle normalization
      while (diff1(3)> M_PI) diff1(3)-=2.*M_PI;
      while (diff1(3)<-M_PI) diff1(3)+=2.*M_PI;

      VectorXd diff2 = Zsig.col(i) - z_pred;
      // angle normalization for Radar measurements
      if (n_z_ == 3) {
        while (diff2(1)> M_PI) diff2(1)-=2.*M_PI;
        while (diff2(1)<-M_PI) diff2(1)+=2.*M_PI;
      }
      Tc += weights(i) * diff1 * diff2.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization for Radar measurements
  if (n_z_ == 3) {
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Also use the z_diff and S to calculate the Normalized Innovation squared (NIS)
  //float nis = z_diff.transpose() * S.inverse() * z_diff;

  // print result
  std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
}