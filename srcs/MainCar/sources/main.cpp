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

void sensorReadingLoop(SpeedSensor& speedSensor) {
    while (running) {
        speedSensor.readData(); // Apenas leitura do CAN
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // ou 1ms dependendo da taxa
    }
}

void publishSensorLoop(SpeedSensor& speedSensor, JetSnailsCar& delorean) {
    const int updateInterval = 30;

    while (running) {
        float speed = speedSensor.getValue();
        delorean.vehicle->setSpeed(speed);  // Vai disparar callback ZMQ
        std::cout << "Speed: " << speed << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(updateInterval));
    }
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);


    std::string can_device = (argc > 1) ? argv[1] : "can0";
    CANBus canBus(can_device, 500000);
    JetSnailsCar delorean;

    SpeedSensor speedSensor(canBus, 0x100);

    zmq::context_t context(1);
    zmq::socket_t publisher_sensors(context, zmq::socket_type::pub);
    try {
        publisher_sensors.bind("tcp://*:5555");
    } catch (const zmq::error_t& e) {
        std::cerr << "Erro ao fazer bind no ZMQ: " << e.what() << std::endl;
        return 1;
    }

    ControllerSubscriber controllerSubscriber("tcp://localhost:5556");
    controllerSubscriber.startListening();

    vehicleSensors vehicle_sensors(publisher_sensors);
    vehicleInformation vehicle_info(publisher_sensors);

    delorean.vehicle->_getPublisher().subscribeToAllChanges(vehicle_sensors);

    // Inicia ambas as threads
    std::thread readerThread(sensorReadingLoop, std::ref(speedSensor));
    std::thread publisherThread(publishSensorLoop, std::ref(speedSensor), std::ref(delorean));

    // Espera até interrupção
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Finaliza threads
    if (readerThread.joinable()) readerThread.join();
    if (publisherThread.joinable()) publisherThread.join();

    std::cout << "Encerrado com sucesso.\n";
    return 0;
}
