#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <iostream>
#include "JetSnailsCar.hpp"
#include "vehicle.h"
#include "SpeedSensor.hpp"
#include "VehicleInfoSubscriber.hpp"
#include "VehicleSubscriber.hpp"
#include "ControllerSubscriber.hpp"
#include "Battery.hpp"
#include <pigpio.h>
#include <zmq.hpp>

std::atomic<bool> running{true};

void signalHandler(int signum) {
    running = false;
}

// Loop único: leitura + publicação da velocidade
void sensorLoop(SpeedSensor& speedSensor, JetSnailsCar& delorean) {
    const int updateInterval = 5; // ajuste aqui a frequência (5ms recomendado)

    while (running) {
        // Leitura do CAN
        speedSensor.readData();

        // Publica no objeto (vai disparar callback ZMQ)
        float speed = speedSensor.getValue();
        delorean.vehicle->setSpeed(speed);

        // Log no console
        std::cout << "Speed: " << speed << std::endl;

        // Controla taxa de atualização
        std::this_thread::sleep_for(std::chrono::milliseconds(updateInterval));
    }
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);

    // Configuração do CAN
    std::string can_device = (argc > 1) ? argv[1] : "can0";
    CANBus canBus(can_device, 500000);
    JetSnailsCar delorean;

    // Sensor de velocidade
    SpeedSensor speedSensor(canBus, 0x100);

    // Configuração ZMQ Publisher
    zmq::context_t context(1);
    zmq::socket_t publisher_sensors(context, zmq::socket_type::pub);
    try {
        publisher_sensors.bind("tcp://*:5555");
    } catch (const zmq::error_t& e) {
        std::cerr << "Erro ao fazer bind no ZMQ: " << e.what() << std::endl;
        return 1;
    }

    // Controller subscriber
    ControllerSubscriber controllerSubscriber("tcp://localhost:5556");
    controllerSubscriber.startListening();

    // Subscribers de sensores e info do veículo
    vehicleSensors vehicle_sensors(publisher_sensors);
    vehicleInformation vehicle_info(publisher_sensors);

    delorean.vehicle->_getPublisher().subscribeToAllChanges(vehicle_sensors);

    // Thread única de leitura+publicação
    std::thread sensorThread(sensorLoop, std::ref(speedSensor), std::ref(delorean));

    // Espera até interrupção
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Finaliza thread
    if (sensorThread.joinable()) sensorThread.join();

    std::cout << "Encerrado com sucesso.\n";
    return 0;
}
