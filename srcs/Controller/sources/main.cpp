#include <iostream>
#include "JetCar.hpp"
#include "LaneDetector.hpp"
#include "ObjectDetector.hpp"
#include <csignal>
#include "Controller.hpp"

const std::string VERSAO = "1.0.0";

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <path_to_model>" << std::endl; return 1;
	}

    std::string modelPath = argv[1];
    std::string modelObjectPath = argv[2];
    auto laneDetector = std::make_unique<LaneDetector>(modelPath); // Criar com unique_ptr
    auto objectDetector = std::make_unique<ObjectDetector>(modelObjectPath, 320);

	std::cout << "[Main] LaneDetector inicializado com sucesso!" << std::endl;
	std::cout << "[Main] Versao..." << VERSAO << std::endl;

    std::cout << "Sistema iniciado com sucesso! Pressione 'q' para sair." << std::endl;
    signal(SIGINT, signalHandler);


    try {
        Controller controller; // Passar ponteiro para JetCar
        controller.setLaneDetector(std::move(laneDetector)); // Transferir posse
        controller.setObjectDetector(std::move(objectDetector));

        std::cout << "[Main] Initializing controller" << std::endl;
        if (!controller.initialize()) {
            std::cerr << "Error initializing camera pipeline" << std::endl;
            return -1;
        }

        controller.listen();
    } catch (const std::runtime_error& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}