#ifndef JETCAR_HPP
#define JETCAR_HPP

#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>    // Para open e O_RDWR
#include <unistd.h>   // Para close
#include <sys/ioctl.h>   // Para ioctl
#include <linux/i2c-dev.h>  // Para I2C_SLAVE
#include <cmath>      // Para std::floor
#include "Configs.hpp"

// const int V_MAX_PWM = 100; // Max PWM value for speed (0-100%)
// const int PWMFREQ = 480; // Frequência do PWM em Hz
// const int SERVO_MIN_PWM = 5; // PWM mínimo para o servo (em %)
// const int SERVO_MAX_PWM = 10; // PWM máximo para o servo (em %)
// const int SERVO_CENTER_PWM = 7; // PWM central para o servo (em %)
// const int SERVO_LEFT_PWM = 5; // PWM para o servo esquerdo (em %)
// const int SERVO_RIGHT_PWM = 10; // PWM para o servo direito (em %)
// const int SERVO_MAX_ANGLE = 30; // Ângulo máximo do servo (em graus)
// const int SERVO_MIN_ANGLE = -30; // Ângulo mínimo do servo (em graus)

class JetCar {
public:
    JetCar(int motorAddr, int servoAddr);
    ~JetCar();

    void set_servo_angle(int angle);  // Função para ajustar o ângulo do servo
    void set_motor_speed(int speed);  // Função para controlar a velocidade do motor
	float get_servo_angle() const;  // Função para obter o ângulo atual do servo

private:
    void open_motor_i2c_bus();
    void open_servo_i2c_bus();
    bool init_motors();
    bool init_servo();
    void writeByteData(int fd, uint8_t reg, uint8_t value);
    uint8_t readByteData(int fd, uint8_t reg);
    bool setMotorPwm(const int channel, int value);

    int _fdMotor;
    int _fdServo;
    int _motorAddr;
    int _servoAddr;

    int _maxAngle;
    int _servoLeftPwm;
    int _servoRightPwm;
    int _servoCenterPwm;
    int _steeringChannel;
    float _currentAngle;
};

#endif  // JETCAR_HPP
