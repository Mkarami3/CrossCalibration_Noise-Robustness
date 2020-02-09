#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <stdlib.h>  
#include <random>

using namespace std;
using namespace cv;

double cal_normal_random(double mean, double sigma)
{
	double rand_num[12], r, r2, c1, c2, c3, c4, c5, gauss_rand;
	unsigned long max;
	int i;

	max = 2147483647;
	for (i = 0; i < 12; i++) {
		rand_num[i] = ((double) random()) / ((double)(max));
	}
	c1 = 0.029899776;
	c2 = 0.008355968;
	c3 = 0.076542912;
	c4 = 0.252408784;
	c5 = 3.949846138;

	r = 0.0;
	for (i = 0; i < 12; i++)
		r += rand_num[i];
	r = (r - 6.0) / 4.0;
	r2 = r * r;
	gauss_rand = ((((c1 * r2 + c2) * r2 + c3) * r2 + c4) * r2 + c5) * r;
	return(mean + sigma * gauss_rand);
}

void CrossPoints(std::vector<cv::Point3d> Gaze, std::vector<cv::Point2d> &ImagePoints) {

	for (auto i = Gaze.begin(); i != Gaze.end(); i++) {
		double x = (1 / i->z) * i->x;
		double y = (1 / i->z) * i->y;
		ImagePoints.push_back(cv::Point2d(x, y));
	}
}

void add_noise_gaze(std::vector<cv::Point3d> GazeVector, int k, std::vector<cv::Point3d> &GazeVector_noise) {

	for (auto i = GazeVector.begin(); i != GazeVector.end(); i++) {
		double mag = sqrt(pow(i->x, 2) + pow(i->y, 2) + pow(i->z, 2));
		double random = cal_normal_random(0, (k+1) * 0.01 * mag);
		//cout << "random = " << random << endl;
		GazeVector_noise.push_back(cv::Point3d(i->x + random, i->y + random, i->z + random));
	}
}

void unknown_scale(cv::Mat rot_Mat, cv::Mat tvecs, std::vector<cv::Point3d> WorldCoord, cv::Mat camera_matrix, std::vector<double> &Scale) {

	for (auto i = WorldCoord.begin(); i != WorldCoord.end(); i++) {
		cv::Mat Sample_3Dpoint = (cv::Mat_<double>(3, 1) << i->x, i->y, i->z);
		cv::Mat Rs = (rot_Mat * Sample_3Dpoint) + tvecs;
		cv::Mat Pixel_coord = camera_matrix * Rs;
		Scale.push_back(Pixel_coord.at<double>(2));
	}

}

void Reprojection(std::vector<cv::Point2d> ImagePoints_noise, std::vector<double> Scale, cv::Mat rot_Mat, cv::Mat tvecs, cv::Mat camera_matrix, cv::Mat &ReprojectionPoints) {
	
	cv::Mat ImagePoints_noise_conc;// = cv::Mat::zeros(6, 3, CV_64F);
	cv::Mat tmpMat1 = cv::Mat(ImagePoints_noise).reshape(1); // converting vector to Matrix
	cv::Mat tmpMat2 = cv::Mat::ones(6, 1, CV_64F);
	hconcat(tmpMat1, tmpMat2, ImagePoints_noise_conc);
	cv::Mat first_part = rot_Mat.inv() * camera_matrix.inv();
	cv::Mat third_part = rot_Mat.inv() * tvecs;
	int counter = 0;
	for (auto i = Scale.begin(); i != Scale.end(); i++) {
		cv::Mat temp = *i * ImagePoints_noise_conc.row(counter); 
		cv::Mat secon_part = first_part * (temp.t());
		ReprojectionPoints.row(counter) = (secon_part - third_part).t();
		counter++;
	}

}

void Reprojection_error(cv::Mat ReprojectionPoints, std::vector<cv::Point3d> WorldCoord, std::vector<double> &error) {
	
	std::vector<double> diff;
	cv::Mat WorldCoordMat = cv::Mat(WorldCoord).reshape(1);
	for (int i = 0; i < 6; i++) {
		diff.push_back(cv::norm(ReprojectionPoints.row(i) - WorldCoordMat.row(i)));
	}
	float average = accumulate(diff.begin(), diff.end(), 0.0) / diff.size();
	error.push_back(average);
}

int main(int argc, char** argv)
{
	FILE* Output;
	Output = fopen("Error.txt", "w");

	std::vector<cv::Point3d> WorldCoord;
	WorldCoord.push_back(cv::Point3d(0, 3, 50));
	WorldCoord.push_back(cv::Point3d(2, -5, 47));
	WorldCoord.push_back(cv::Point3d(-1, 7, 60));
	WorldCoord.push_back(cv::Point3d(5, -1, 40));
	WorldCoord.push_back(cv::Point3d(0, 2, 45));
	WorldCoord.push_back(cv::Point3d(3, -4, 44));

	std::vector<cv::Point3d> GazeVector;
	GazeVector.push_back(cv::Point3d(0, 3, 30));
	GazeVector.push_back(cv::Point3d(-3, -5, 28));
	GazeVector.push_back(cv::Point3d(10, 7, 31));
	GazeVector.push_back(cv::Point3d(-10, 1, 25));
	GazeVector.push_back(cv::Point3d(-5, 2, 30));
	GazeVector.push_back(cv::Point3d(-6, -4, 27));

	cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	std::vector<cv::Point2d> ImagePoints;
	std::vector<double> dist_coeffs(4,0);
	std::vector<double> error_PnP;
	std::vector<double> error_PnPRansac;
	for (int k = -1; k < 20; k++) {
		std::vector<cv::Point2d> ImagePoints_noise;
		std::vector<cv::Point3d> GazeVector_noise;

		cv::Mat rvecs_PnP = cv::Mat::zeros(3, 1, CV_64FC1);
		cv::Mat tvecs_PnP = cv::Mat::zeros(3, 1, CV_64FC1);
		std::vector<double> Scale_PnP;
		cv::Mat rot_Mat_PnP = cv::Mat::zeros(3, 3, CV_64FC1);
		cv::Mat ReprojectionPoints_PnP = cv::Mat::zeros(cv::Size(3, 6), CV_64FC1);

		cv::Mat rvecs_PnPRansac = cv::Mat::zeros(3, 1, CV_64FC1);
		cv::Mat tvecs_PnPRansac = cv::Mat::zeros(3, 1, CV_64FC1);
		std::vector<double> Scale_PnPRansac;
		cv::Mat rot_Mat_PnPRansac = cv::Mat::zeros(3, 3, CV_64FC1);
		cv::Mat ReprojectionPoints_PnPRansac = cv::Mat::zeros(cv::Size(3, 6), CV_64FC1);

		add_noise_gaze(GazeVector, k,  GazeVector_noise);															// Output: GazeVector_noise
		CrossPoints(GazeVector_noise, ImagePoints_noise);															// Output: ImagePoints_noise

		cv::solvePnP(WorldCoord, ImagePoints_noise, camera_matrix, dist_coeffs, rvecs_PnP, tvecs_PnP, CV_EPNP);			// Outputs: rvecs, tvecs
		cv::solvePnPRansac(WorldCoord, ImagePoints_noise, camera_matrix, dist_coeffs, rvecs_PnPRansac, tvecs_PnPRansac);
		cv::Rodrigues(rvecs_PnP, rot_Mat_PnP);																			//Output: rot_Mat
		cv::Rodrigues(rvecs_PnPRansac, rot_Mat_PnPRansac);
		unknown_scale(rot_Mat_PnP, tvecs_PnP, WorldCoord, camera_matrix, Scale_PnP);									// Output: Scale
		unknown_scale(rot_Mat_PnPRansac, tvecs_PnPRansac, WorldCoord, camera_matrix, Scale_PnPRansac);
		Reprojection(ImagePoints_noise, Scale_PnP, rot_Mat_PnP, tvecs_PnP, camera_matrix, ReprojectionPoints_PnP);		// Output: ReprojectionPoints
		Reprojection(ImagePoints_noise, Scale_PnPRansac, rot_Mat_PnPRansac, tvecs_PnPRansac, camera_matrix, ReprojectionPoints_PnPRansac);
		Reprojection_error(ReprojectionPoints_PnP, WorldCoord, error_PnP);												// Output: error
		Reprojection_error(ReprojectionPoints_PnPRansac, WorldCoord, error_PnPRansac);
		if (k == -1) {
			cout << "RotationMatrix:" << endl << rot_Mat_PnP << endl;
			cout << "TranslationMatrix:" << endl << tvecs_PnP << endl;
		}

		//cout << error_PnP.at(k) << "\t" << error_PnPRansac.at(k) << endl;
	}

	fprintf(Output,"PnP PnPRansac\n");
	for (int i = 0; i < 20; i++)
		fprintf(Output, "%f %f\n", error_PnP.at(i), error_PnPRansac.at(i)); // Write to file

	fclose(Output);
	return 0;

}