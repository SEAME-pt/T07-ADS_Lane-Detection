#ifndef CONTROLLER_SUBSCRIBER_HPP
#define CONTROLLER_SUBSCRIBER_HPP

#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>

class ControllerSubscriber {
public:
    ControllerSubscriber(const std::string& subAddress, zmq::socket_t& publisher)
        : context(1),
          subscriber(context, zmq::socket_type::sub),
          publisher_(publisher)
    {
        subscriber.connect(subAddress);
        subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    }

    void startForwarding() {
        std::thread listener([this]() {
            while (true) {
                zmq::message_t message;
                auto result = subscriber.recv(message, zmq::recv_flags::none);

                if (!result) {
                    std::cerr << "Erro ao receber mensagem." << std::endl;
                    continue;
                }

                std::string received(static_cast<char*>(message.data()), message.size());
                std::cout << "[ControllerSubscriber] Repassando no mesmo PUB (5555): " << received << std::endl;

                // repassa no publisher já existente (5555)
                publisher_.send(zmq::buffer(received), zmq::send_flags::none);
            }
        });

        listener.detach();
    }

private:
    zmq::context_t context;
    zmq::socket_t subscriber;
    zmq::socket_t& publisher_; // referência ao publisher principal
};

#endif // CONTROLLER_SUBSCRIBER_HPP
