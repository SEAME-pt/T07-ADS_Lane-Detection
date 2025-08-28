#ifndef CONFIGS_HPP
#define CONFIGS_HPP


const int V_MAX_PWM = 100; // Max PWM value for speed (0-100%)

const int SERVO_MIN_PWM = 5; // Min PWM for servo (%)
const int SERVO_MAX_PWM = 10; // Max PWM for servo (%)
const int SERVO_CENTER_PWM = 7; // Center PWM for servo (%)
const int SERVO_LEFT_PWM = 5; // Left PWM for servo (%)
const int SERVO_RIGHT_PWM = 10; // Right PWM for servo (%)
const int SERVO_MAX_ANGLE = 30; // Max servo angle (degrees)
const int SERVO_MIN_ANGLE = -30; // Min servo angle (degrees)

// JetRacer parameters
const double L = 0.15;		  // Wheelbase (m)
const double DT = 0.1;		  // Time step (s)

// Car velovity limits
const double V_MAX = 2.5;		// Maximal velocity (m/s)
const double V_MIN = 0.1;		// Minimal velocity for MPC update (m/s)
const double V_REF_PWM = 24;	// Reference velocity (%) v = V_REF_PWM * V_MAX / V_MAX_PWM
// Car steering limits
const double DELTA_MAX = 0.5;		// Max steering angle (rad, ~30 deg)
const double DELTA_RATE_MAX = 0.1;	// Max steering rate (rad/step)
// Car max acceleration
const double A_MAX = 2.0;			// Max acceleration (m/s^2)
// Car tration control PWM frequency. Higher frequency -> lower momentum (20...960)
const int V_PWM_FREQ = 120; // PWM frequency in Hz *** tested [120] high pwm -> low momentum

// MPC parameters
const int N = 10;			// Prediction horizon steps
const int MPC_ITER = 500;	// Max gradient descent iterations

// MPC cross-track error and heading error weights
const double Q_EY = 10.0;	 // Weight for cross-track error
const double QF_EY = 50.0;	// Terminal weight for cross-track error

// MPC heading error weights
const double Q_YAW = 0.5;	// Weight for heading error original 5
const double QF_YAW = 2.5;	// Terminal weight for heading error

// MPC velocity error weight
const double Q_V = 1.0;		// Weight for velocity error

// MPC control effort weights
const double R = 1.0 / (DELTA_MAX) * (DELTA_MAX);		 // Control cost for smoothness
const double R_DELTA = 1.0;	// Weight for steering effort
const double R_A = 1.0;		// Weight for acceleration effort
const double R_V = 1.0;		// Weight for velocity effort
const double R_DELTA_RATE = 0.25; // Weight for steering rate effort

// Lane detection parameters
#define ROI_SY_PERCENT 0.5f // ROI starts at 50% of image height
#define ROI_EY_PERCENT 0.90f // ROI ends at 90% of image height
#define ROI_X_BORDER 0 // Pixels from edges to avoid noise
#define LANE_THRESHOLD 0.5f // Threshold for binary mask
#define MIN_EDGE_POINTS 10 // Coefficient for angle calculation
#define MAX_SEARCH_DISTANCE 310 // Max distance (pixels) to search for edges

#define I_W 256 // Inference image width
#define I_H 128 // Inference image height
#define F_W 640 // Frame width
#define F_H 360 // Frame height

#define KALMAN false // Use Kalman filter for smoothing
#define CAR_CM false // Use car center of mass

// Car frame coordinates: dash cam
#define X_CAR_FRAME_CENTER 0.41f // X coordinate of car center
#define X_CAR_FRAME_BOTTOM 0.20f // X coordinate of bottom ROI
#define CAMERA_Y_POS 0.0f // Centered on car's CM (m)
#define CAMERA_X_POS 0.09f // 9 cm forward of car's CM
#define CAMERA_Z_POS 0.115 // 11.5 cm height (m)
#define CAMERA_OFFSET 0 // Pixel offset
#define CAMERA_TILT 19.0 * CV_PI / 180.0 // 19 degrees in radians

#define CAMERA_FOCAL_LENGTH 0.00315 // Focal length in meters (3.15 mm)

// Lane geometry history
#define MAX_HISTORY_SIZE 10 // Max history size

// Distance calculation coefficients
#define Asy -2.6e-6 // Coefficient for distance
#define Bsy 1.35e-3 // Coefficient for distance

typedef struct s_carFrame {
	float xT{X_CAR_FRAME_CENTER}; // X coordinate
	float yT{0.0f}; // Intercept of left lane line
	float xB{X_CAR_FRAME_BOTTOM}; // Slope of right lane line
	float yB{0.0f}; // Intercept of right lane line
	float xDelta{X_CAR_FRAME_CENTER - X_CAR_FRAME_BOTTOM}; // Delta X
	float slope{0.0f}; // Slope of right lane line
	float intercept{0.0f}; // Intercept of right lane line
	float angle{0.0f}; // Angle of lane in radians
} t_carFrame;

typedef struct s_imgFrame {
	int xltPX{0}; // Left edge at top
	int xrtPX{0}; // Right edge at top
	int xlbPX{0}; // Left edge at bottom
	int xrbPX{0}; // Right edge at bottom
	int xmtPX{0}; // Midpoint at top
	int xmbPX{0}; // Midpoint at bottom
	int xcPX{F_W / 2 - CAMERA_OFFSET}; // Image center
	float xmt{0.0f}; // Midpoint at top (m)
	float xmb{0.0f}; // Midpoint at bottom (m)
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