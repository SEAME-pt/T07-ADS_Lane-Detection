#include <iostream>
#include "JetCar.hpp"
#include "LaneDetector.hpp"
#include <csignal>
#include "Controller.hpp"

const std::string VERSAO = "1.0.0";

JetCar jetCar(0x60, 0x40);

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0);
    exit(signum);
}

void handleSteering(int value) {
    int servoAngle = static_cast<int>((value / 32768.0) * 90);
    servoAngle = std::max(-90, std::min(90, servoAngle));
    jetCar.set_servo_angle(servoAngle);
}

void handleMotors(int value) {
    value *= -1;
    int motorSpeed = static_cast<int>((value / 32768.0) * 100);
    jetCar.set_motor_speed(motorSpeed);
}

int changeMode(int mode, Controller &controller, JetCar &jetCar) {
    controller.setMode(mode == MODE_JOYSTICK ? MODE_AUTONOMOUS : MODE_JOYSTICK);
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0);
    std::cout << "Modo " << (mode == MODE_JOYSTICK ? "autônomo" : "joystick") << " ativado!" << std::endl;

    return 0;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <path_to_model>" << std::endl; return 1;
	}
	std::string modelLanePath = argv[1];
	std::string modelObjectPath = argv[2];
    auto laneDetector = std::make_unique<LaneDetector>(modelLanePath);
	auto objectDetector = std::make_unique<ObjectDetector>(modelObjectPath);


	std::cout << "[Main] LaneDetector inicializado com sucesso!" << std::endl;
	std::cout << "[Main] Versao..." << VERSAO << std::endl;

    std::cout << "Sistema iniciado com sucesso! Pressione 'q' para sair." << std::endl;
    signal(SIGINT, signalHandler);


    Actions changeModeActions;
    try {
        Controller controller(&jetCar); // Passar ponteiro para JetCar
        controller.setLaneDetector(std::move(laneDetector)); // Transferir posse
        controller.setObjectDetector(std::move(objectDetector)); // Transferir posse


        std::cout << "[Main] Initializing controller" << std::endl;
        if (!controller.initialize()) {
            std::cerr << "Error initializing camera pipeline" << std::endl;
            return -1;
        }

        changeModeActions.onPress = nullptr;
        changeModeActions.onRelease = [&](){
            changeMode(controller.getMode(), controller, jetCar);
        };

        controller.setAxisAction(3, handleMotors);
        controller.setAxisAction(0, handleSteering);
        controller.setButtonAction(BTN_START, changeModeActions);


        controller.listen();
    } catch (const std::runtime_error& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}