#ifndef CONFIGS_HPP
#define CONFIGS_HPP



const int PWMFREQ = 480; // Frequência do PWM em Hz

const int V_MAX_PWM = 100; // Max PWM value for speed (0-100%)


const int SERVO_MIN_PWM = 5; // PWM mínimo para o servo (em %)
const int SERVO_MAX_PWM = 10; // PWM máximo para o servo (em %)
const int SERVO_CENTER_PWM = 7; // PWM central para o servo (em %)
const int SERVO_LEFT_PWM = 5; // PWM para o servo esquerdo (em %)
const int SERVO_RIGHT_PWM = 10; // PWM para o servo direito (em %)
const int SERVO_MAX_ANGLE = 30; // Ângulo máximo do servo (em graus)
const int SERVO_MIN_ANGLE = -30; // Ângulo mínimo do servo (em graus)


// JetRacer parameters
const double L = 0.15;          // Wheelbase (m)
const double DT = 0.1;          // Time step (s)

// JetRacer MPC parameters
const double V_MAX = 2.5;      // Max velocity (m/s)
const double DELTA_MAX = 0.3; // Max steering angle (rad, 30 deg)
const double DELTA_RATE_MAX = 0.1; // Max steering rate (rad/step)
const double A_MAX = 2.0;      // Max acceleration (m/s^2)
const double V_REF = 0.4;      // Reference velocity (m/s)
const double V_REF_PWM = 50;      // Reference velocity (m/s) in % of PWM



// MPC parameters
const int N = 10;               // Prediction horizon
// MPC cross-track error and heading error weights
const double Q_EY = 1100.0;     // Weight for cross-track error
const double QF_EY = 20.0;     // Weight for cross-track error
// MPC heading error weights
const double Q_YAW = 14.0; // Weight for heading error
const double QF_YAW = 28.0; // Weight for heading error
// MPC velocity error weight
const double Q_V = 1.0;        // Weight for velocity error
// MPC control effort weights
const double R = 1.0 / (DELTA_MAX) * (DELTA_MAX);         // Control cost for smoothness
const double R_DELTA = 1.0;    // Weight for steering effort
const double R_A = 1.0;        // Weight for acceleration effort


// Lane detection parameters
#define ROI_SY_PERCENT 0.5f // ROI starts at 50% of image height
#define ROI_EY_PERCENT 0.90f   // ROI ends at 80% of image height
#define ROI_X_BORDER 0 // Pixels from the left and right edges to avoid noise
#define LANE_THRESHOLD 0.5f // Threshold for binary mask
#define MIN_EDGE_POINTS 10 // Coefficient for angle calculation
#define MAX_SEARCH_DISTANCE 310	// Max distance (pixels) to search for edges

#define I_W 256 // Inference image with
#define I_H 128 // Inference image height
#define F_W 640 // Frame width
#define F_H 360 // Frame height

#define KALMAN false // Use Kalman filter for smoothing offset and angle
#define CAR_CM false // Use car center of mass for calculations


// Car frame coordinates : dash cam
#define X_CAR_FRAME_CENTER 0.41f // X coordinate of the car center in the image frame
#define X_CAR_FRAME_BOTTOM 0.20f // X coordinate of the bottom ROI in the image frame
#define CAMERA_Y_POS 0.0f	 // 0 cm in meters, centered on the car's CM
#define CAMERA_X_POS 0.09f	// 9 cm in meters, forward of the car's CM
#define CAMERA_Z_POS 0.115   // 11.5 cm in meters
#define CAMERA_OFFSET 0 // Offset in pixels, adjust if needed
#define CAMERA_TILT 19.0 * CV_PI / 180.0 // 19 degrees in radians (19 * pi/180)

// Fixed parameters as constants
#define CAMERA_FOCAL_LENGTH = 0.00315 // Focal length in meters (2 mm)

//New members for lane geometry historical data
#define MAX_HISTORY_SIZE 10 // Maximum size of the history

// Coefficients for distance calculation, converting pixels to meters
// Equations :
//  d(m) = s(y) * x
// 	s(y) = a * y + b
// x and y are pixel coordinates
#define Asy -2.6e-6 // Coefficient for distance calculation
#define Bsy 1.35e-3 // Coefficient for distance calculation

typedef struct s_carFrame {
	float xT{X_CAR_FRAME_CENTER};	// X coordinate
	float yT{0.0f};	// Intercept of the left lane line
	float xB{X_CAR_FRAME_BOTTOM};	// Slope of the right lane line
	float yB{0.0f};	// Intercept of the right lane line
	float xDelta{X_CAR_FRAME_CENTER - X_CAR_FRAME_BOTTOM};	// Delta X between top and bottom points
	float slope{0.0f};	// Slope of the right lane line
	float intercept{0.0f};	// Intercept of the right lane line
	float angle{0.0f};	// Angle of the lane in radians
} t_carFrame;

typedef struct s_imgFrame {
	int xltPX{0};	// Left edge at top
	int xrtPX{0};	// Right edge at top
	int xlbPX{0};	// Left edge at bottom
	int xrbPX{0};	// Right edge at bottom
	int xmtPX{0};	// Midpoint at top
	int xmbPX{0};	// Midpoint at bottom
	int xcPX{F_W / 2 - CAMERA_OFFSET};	// Center of the image
	float xmt{0.0f};	// Midpoint at top in meters
	float xmb{0.0f};	// Midpoint at bottom in meters
} t_imgFrame;

enum KalmanStateIndex { OFFSET = 0, OFFSET_VEL = 1, ANGLE = 2 };
enum KalmanMeasurementIndex { MEASUREMENT_OFFSET = 0, MEASUREMENT_ANGLE = 1 };
enum KalmanPredictionIndex { PREDICTION_OFFSET = 0, PREDICTION_ANGLE = 1 };
enum KalmanPredictionCovIndex { PREDICTION_COV_OFFSET = 0, PREDICTION_COV_ANGLE = 1 };
enum KalmanMeasurementCovIndex { MEASUREMENT_COV_OFFSET = 0, MEASUREMENT_COV_ANGLE = 1 };
enum KalmanErrorCovIndex { ERROR_COV_OFFSET = 0, ERROR_COV_ANGLE = 1 };
enum KalmanProcessCovIndex { PROCESS_COV_OFFSET = 0, PROCESS_COV_ANGLE = 1 };
enum KalmanTransitionIndex { TRANSITION_OFFSET = 0, TRANSITION_VEL = 1, TRANSITION_ANGLE = 2 };
enum KalmanMeasurementMatrixIndex { MEASUREMENT_MATRIX_OFFSET = 0, MEASUREMENT_MATRIX_ANGLE = 1 };

#endif // CONFIGS_HPP