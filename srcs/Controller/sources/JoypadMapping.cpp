#include "JoypadMapping.hpp"
#include "devices.hpp"
#include "JetCar.hpp"

JoypadMapping::JoypadMapping(JetCar* car) : jetCar(car) {

    // Change Mode
    buttonMappings[BTN_START] = { 
        [this]() { nullptr; },
        [this]() { jetCar->changeCurrentMode(); }
    };

    // Exit
    buttonMappings[BTN_SELECT] = {
        [this]() { nullptr; },
        [this]() { jetCar->setTurnOn(0); }
    };

    // Slow Speed -1 on release btn L
    buttonMappings[BTN_LB] = {
        [this]() { nullptr; },
        [this]() { jetCar->decreaseSpeed(jetCar->getCruiseSpeed(), 2); }
    };

    // IncreaseSpeed
    buttonMappings[BTN_RB] = {
        [this]() { nullptr; },
        [this]() { jetCar->increaseSpeed(jetCar->getCruiseSpeed(), 2); }
    };

    // Configure axis actions
    axisMappings[0] = [this](int value) { jetCar->set_servo_angle(value); };
    axisMappings[3] = [this](int value) { jetCar->manualMotorSpeed(value); };
}

void JoypadMapping::processEvent(const SDL_Event& event) {
    if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) {
        bool pressed = (event.type == SDL_JOYBUTTONDOWN);
        int button = event.jbutton.button;
        auto it = buttonMappings.find(button);
        if (it != buttonMappings.end()) {
            if (pressed && it->second.onPress) it->second.onPress();
            else if (!pressed && it->second.onRelease) it->second.onRelease();
        }
    } else if (event.type == SDL_JOYAXISMOTION && jetCar->getCurrentMode() != MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        auto it = axisMappings.find(axis);
        if (it != axisMappings.end() && it->second) {
            it->second(value);
        }
    } else if( event.type == SDL_JOYAXISMOTION && jetCar->getCurrentMode() == MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        auto it = axisMappings.find(axis);
        if (axis == 3) {
            if (it != axisMappings.end() && it->second) {
                it->second(value);
            }
        }
    }

}

