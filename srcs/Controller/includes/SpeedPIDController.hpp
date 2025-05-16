// SpeedPIDController.hpp**
#pragma once

class SpeedPIDController {
public:
    SpeedPIDController();

    float update(float v_current, float v_target, float dt);
    void reset();

private:
    float kp_ = 0.3 , ki_= 0.15, kd_ = 0.2;
    float pwm_min_ = 0, pwm_max_ = 100;
    float prev_error_ = 0.0, integral_ = 0.0;
    
    
    static float integral;
};