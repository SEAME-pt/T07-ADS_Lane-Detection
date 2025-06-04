#pragma once
#include <zmq.hpp>
#include <thread>
#include <atomic>
#include <functional>
#include <string>
#include <iostream>

class SpeedSubscriber {
public:
    SpeedSubscriber(const std::string& address = "tcp://localhost:5555")
        : context(1), subscriber(context, zmq::socket_type::sub), running(false) {
        subscriber.connect(address);
        // Assine o tópico "speed"
        subscriber.setsockopt(ZMQ_SUBSCRIBE, "speed", 5);
        // Opcional: habilita conflate para só receber a última mensagem
        // int one = 1;
        // subscriber.setsockopt(ZMQ_CONFLATE, &one, sizeof(one));
    }

    void start(std::function<void(float)> callback) {
        running = true;
        listenThread = std::thread([this, callback]() {
            zmq::pollitem_t items[] = {
                { static_cast<void*>(subscriber), 0, ZMQ_POLLIN, 0 }
            };
            while (running) {
                // Poll espera até 100 ms por dados
                int rc = zmq::poll(items, 1, 100);
                if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
                    zmq::message_t message;
                    // Recebe com dontwait porque já sabemos que tem dados
                    if (subscriber.recv(message, zmq::recv_flags::dontwait)) {
                        std::string msgStr(static_cast<char*>(message.data()), message.size());
                        // std::cout << "Recebido: " << msgStr << std::endl;

                        auto delimiterPos = msgStr.find(' ');
                        if (delimiterPos != std::string::npos) {
                            std::string key = msgStr.substr(0, delimiterPos);
                            std::string valueStr = msgStr.substr(delimiterPos + 1);
                            try {
                                float value = std::stof(valueStr);
                                callback(value);
                            } catch (...) {

                            }
                        }
                    }
                }
            }
        });
    }

    void stop() {
        running = false;
        if (listenThread.joinable()) {
            listenThread.join();
        }
    }

    ~SpeedSubscriber() {
        stop();
    }

private:
    zmq::context_t context;
    zmq::socket_t subscriber;
    std::thread listenThread;
    std::atomic<bool> running;
};
