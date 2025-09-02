#ifndef JOYPADMAPPING_HPP
#define JOYPADMAPPING_HPP

#include <functional>
#include <unordered_map>
#include "JetCar.hpp"
#include <SDL2/SDL.h>

#define BTN_A 0
#define BTN_B 1
#define BTN_X 3
#define BTN_Y 4
#define BTN_LB 6
#define BTN_RB 7
#define BTN_SELECT 10
#define BTN_START 11
#define BTN_HOME 12
#define BTN_LSTICK 13
#define BTN_RSTICK 14

class JetCar;

/**
 * @brief Stores actions for a button (onPress and onRelease).
 */
struct ButtonActions {
    std::function<void()> onPress;
    std::function<void()> onRelease;
};

/**
 * @brief Maps joystick buttons and axes to JetCar actions.
 */
class JoypadMapping {
public:
    explicit JoypadMapping(JetCar* car);

    void processEvent(const SDL_Event& event);

private:
    JetCar* jetCar;

    zmq::context_t zmq_context_;
    zmq::socket_t zmq_publisher_;

    std::unordered_map<int, ButtonActions> buttonMappings;
    std::unordered_map<int, std::function<void(int)>> axisMappings;
};

#endif // JOYPADMAPPING_HPP

