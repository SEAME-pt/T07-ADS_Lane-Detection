#include "JetCar.hpp"

// Implementação da classe JetCar

JetCar::JetCar(int motorAddr, int servoAddr)
    : _motorAddr(motorAddr), _servoAddr(servoAddr), _fdMotor(-1), _fdServo(-1),
      _maxAngle(30), _servoLeftPwm(170), _servoRightPwm(430), _servoCenterPwm(300),
      _steeringChannel(0), _currentAngle(0) {

    _currentMode = MODE_JOYSTICK;
    V_REF_PWM = 35.0;
    _isTurnOn = 1;

    // Inicializar servo e motores
    open_servo_i2c_bus();
    if (!init_servo()) {
        throw std::runtime_error("Erro ao inicializar o servo.");
    }

    open_motor_i2c_bus();
    if (!init_motors()) {
        throw std::runtime_error("Erro ao inicializar os motores.");
    }

    std::cout << "JetCar inicializado com sucesso!" << std::endl;
}

JetCar::~JetCar() {
    set_motor_speed(0);
    close(_fdMotor);
    close(_fdServo);
    std::cout << "Destruindo o JetCar..." << std::endl;
}

void JetCar::set_servo_angle(int angle) {
    
    int servoAngle = static_cast<int>((angle / 32768.0) * 90);
    servoAngle = std::max(-90, std::min(90, servoAngle));
    
	// converet to rad
	_currentAngle = (servoAngle * 3.1415f / 180.0f);  // Atualiza o ângulo atual do servo
    servoAngle = std::max(-_maxAngle, std::min(_maxAngle, servoAngle));

    int pwm;
    if (servoAngle < 0) {
        pwm = static_cast<int>(_servoCenterPwm + (static_cast<float>(servoAngle) / _maxAngle) * (_servoCenterPwm - _servoLeftPwm));
    } else if (servoAngle > 0) {
        pwm = static_cast<int>(_servoCenterPwm + (static_cast<float>(servoAngle) / _maxAngle) * (_servoRightPwm - _servoCenterPwm));
    } else {
        pwm = _servoCenterPwm;
    }

    writeByteData(_fdServo, 0x06 + 4 * _steeringChannel, 0);
    writeByteData(_fdServo, 0x07 + 4 * _steeringChannel, 0);
    writeByteData(_fdServo, 0x08 + 4 * _steeringChannel, pwm & 0xFF);
    writeByteData(_fdServo, 0x09 + 4 * _steeringChannel, pwm >> 8);
    _currentAngle = angle;
}

void JetCar::set_motor_speed(int speed) {
    int pwmValue;

    // Inverte o sinal se necessário
    speed = -speed;

    // Limita o valor entre -V_MAX_PWM e V_MAX_PWM
    speed = std::max(-V_MAX_PWM, std::min(V_MAX_PWM, speed));

    std::cout << "[Motor] Speed set to: " << speed << std::endl;

    // Converte para valor de PWM (0-4095)
    pwmValue = static_cast<int>(std::abs(speed) / 100.0 * 4095);

    if (speed > 0) {
        // --- MOVIMENTO PARA FRENTE ---
        // Motor A
        setMotorPwm(0, pwmValue); // IN1
        setMotorPwm(1, 0);        // IN2
        setMotorPwm(2, pwmValue); // ENA

        // Motor B
        setMotorPwm(5, pwmValue); // IN3
        setMotorPwm(6, 0);        // IN4
        setMotorPwm(7, pwmValue); // ENB

    } else if (speed < 0) {
        // --- MOVIMENTO PARA RÉ ---
        // Motor A
        setMotorPwm(0, 0);        // IN1
        setMotorPwm(1, pwmValue); // IN2
        setMotorPwm(2, pwmValue); // ENA

        // Motor B
        setMotorPwm(5, 0);        // IN3
        setMotorPwm(6, pwmValue); // IN4
        setMotorPwm(7, pwmValue); // ENB

    } else {
        // --- FREIO ATIVO SEGURO ---
        // Motor A
        setMotorPwm(0, 0); // IN1
        setMotorPwm(1, 0); // IN2
        setMotorPwm(2, 0); // ENA

        // Motor B
        setMotorPwm(5, 0); // IN3
        setMotorPwm(6, 0); // IN4
        setMotorPwm(7, 0); // ENB
}
}


// Métodos privados para controle de I2C

void JetCar::open_motor_i2c_bus() {
    std::string i2cDevice = "/dev/i2c-1";
    _fdMotor = open(i2cDevice.c_str(), O_RDWR);
    if (_fdMotor < 0) throw std::runtime_error("Erro ao abrir o barramento I2C para o motor");

    if (ioctl(_fdMotor, I2C_SLAVE, _motorAddr) < 0) {
        close(_fdMotor);
        throw std::runtime_error("Erro ao configurar o endereço I2C do motor");
    }
}

void JetCar::open_servo_i2c_bus() {
    std::string i2cDevice = "/dev/i2c-1";
    _fdServo = open(i2cDevice.c_str(), O_RDWR);
    if (_fdServo < 0) throw std::runtime_error("Erro ao abrir o barramento I2C para o servo");

    if (ioctl(_fdServo, I2C_SLAVE, _servoAddr) < 0) {
        close(_fdServo);
        throw std::runtime_error("Erro ao configurar o endereço I2C do servo");
    }
}

bool JetCar::init_motors() {
    try {
        writeByteData(_fdMotor, 0x00, 0x20);
        usleep(1000);

        int preScale;
        uint8_t oldMode, newMode;

        oldMode = readByteData(_fdMotor, 0x00);
        preScale = static_cast<int>(std::floor((25000000.0 / 4096.0 / V_PWM_FREQ) + 0.5) - 1);
        newMode = (oldMode & 0x7F) | 0x10;

        writeByteData(_fdMotor, 0x00, newMode);
        writeByteData(_fdMotor, 0xFE, preScale);
        writeByteData(_fdMotor, 0x00, oldMode);

        usleep(5000);

        writeByteData(_fdMotor, 0x00, oldMode | 0xa1);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erro na inicialização dos motores: " << e.what() << std::endl;
        return false;
    }
}

bool JetCar::init_servo() {
    try {
        writeByteData(_fdServo, 0x00, 0x06);  // Reset do servo
        usleep(100000);

        writeByteData(_fdServo, 0x00, 0x10);  // Configura o controle
        usleep(100000);

        writeByteData(_fdServo, 0xFE, 0x79);  // Frequência do PWM
        usleep(100000);

        writeByteData(_fdServo, 0x01, 0x04);  // Configura o MODE2
        usleep(100000);

        writeByteData(_fdServo, 0x00, 0x20);  // Habilita o auto-incremento
        usleep(100000);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erro na inicialização do servo: " << e.what() << std::endl;
        return false;
    }
}

void JetCar::writeByteData(int fd, uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    if (write(fd, buffer, 2) != 2) {
        throw std::runtime_error("Erro ao escrever no dispositivo I2C.");
    }
}

uint8_t JetCar::readByteData(int fd, uint8_t reg) {
    if (write(fd, &reg, 1) != 1)
        throw std::runtime_error("Erro ao enviar o registrador ao dispositivo I2C.");

    uint8_t value;
    if (read(fd, &value, 1) != 1)
        throw std::runtime_error("Erro ao ler o registrador ao dispositivo I2C.");

    return value;
}

bool JetCar::setMotorPwm(const int channel, int value) {
    value = std::max(0, std::min(4095, value));
    writeByteData(_fdMotor, 0x06 + 4 * channel, 0);
    writeByteData(_fdMotor, 0x07 + 4 * channel, 0);
    writeByteData(_fdMotor, 0x08 + 4 * channel, value & 0xFF);
    writeByteData(_fdMotor, 0x09 + 4 * channel, value >> 8);
    return true;
}

// get servor angle
float JetCar::get_servo_angle() const {
	return _currentAngle;
}

void JetCar::setCurrentMode(int mode) {
    if (mode == MODE_JOYSTICK || mode == MODE_AUTONOMOUS) {
        _currentMode = mode;
    }
}

int JetCar::getCurrentMode() const {
    return _currentMode;
}

void JetCar::changeCurrentMode() {
    if (_currentMode == MODE_JOYSTICK) {
        _currentMode = MODE_AUTONOMOUS;
        // std::cout << "Modo autônomo ativado!" << std::endl;
    } else {
        _currentMode = MODE_JOYSTICK;
        set_motor_speed(0);
        set_servo_angle(0);
        // std::cout << "Modo joystick ativado!" << std::endl;
    }
    std::cout << "Modo atual: " << (_currentMode == MODE_JOYSTICK ? "Joystick" : "Autônomo") << std::endl;
}

int JetCar::turnOff() {
    if (_isTurnOn) {
        set_motor_speed(0);
        set_servo_angle(0);
    }
    return 1;
}



void JetCar::increaseSpeedby1() {
    V_REF_PWM = std::min(V_REF_PWM + 1.0, static_cast<double>(V_MAX_PWM));
    std::cout << "Velocidade aumentada para: " << V_REF_PWM << std::endl;
}

void JetCar::slowSpeedby1() {
    V_REF_PWM = std::max(V_REF_PWM - 1.0, 0.0);
    std::cout << "Velocidade reduzida para: " << V_REF_PWM << std::endl;
}

void JetCar::setVRefPwm(double value) {
    V_REF_PWM = value;
}

double JetCar::getVRefPwm() const {
    return V_REF_PWM;
}

void JetCar::setTurnOn(const int &value) {
    _isTurnOn = value;
}

int JetCar::getTurnOn() const {
    return _isTurnOn;
}

void JetCar::manualSteering(int angle) {
    set_servo_angle(angle);
}

void JetCar::manualMotorSpeed(int speed) {
    set_motor_speed(speed);
}