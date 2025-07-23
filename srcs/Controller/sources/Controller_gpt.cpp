void Controller::listen() {
    SDL_Event event;

    startProcessingThread();  // Start it once before loop

    while (true) {
		// Read frame from camera
        if (!cap_.read(frame) || frame.empty()) {
            std::cerr << "Fail to obtain frame!" << std::endl;
            continue;
        }
		// Process joystick events
        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }

        if (new_result_ready_) {
            FrameResult result_copy;

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_copy = frame_result_;
                new_result_ready_ = false;
            }

            if (_currentMode == MODE_AUTONOMOUS) {
                delta_ = jetCar->get_servo_angle();
                autonomous(delta_, result_copy.ey, result_copy.yaw);
                visualize_mask_ = false;
            } else {
                visualize_mask_ = true;
            }

            // Show mode on screen
            std::string modeText = (_currentMode == MODE_JOYSTICK) ? "Joystick Mode" : "Autonomous Mode";
            cv::putText(result_copy.output_frame, modeText, cv::Point(330, 340),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            video_writer.write(result_copy.output_frame);
        }

        if (buttonStates[BTN_SELECT] && buttonStates[BTN_START]) {
            break;
        }

        if (!joystick) {
            std::cout << "No joystick connected, quitting..." << std::endl;
            break;
        }

        SDL_Delay(10);
    }

    stop_processing_ = true;
}