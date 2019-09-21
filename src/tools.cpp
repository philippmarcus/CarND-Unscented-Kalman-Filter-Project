#include "tools.h"

using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
   /**
   * Calculate the RMSE here.
   * 
   * Sum up the differences. Divide by the number of summed up elements. Take the square root. 
   */
   assert (estimations.size() == ground_truth.size());

   VectorXd mean_square = VectorXd(4);
   mean_square << 0, 0, 0, 0;
   // Iterate over all the measurements
   for (int i = 0; i < estimations.size(); i++) {
      // Iterate over elements of a measurement
      for (int elem = 0; elem < 4; elem++){
         mean_square(elem) += pow(estimations[i](elem) - ground_truth[i](elem), 2);
      }
   }
   mean_square = mean_square / estimations.size();
   VectorXd rmse = VectorXd(4);
   for (int elem = 0; elem < 4; elem++){
      rmse(elem) = sqrt(mean_square(elem));
   }
  // return a vector
  return rmse;
}