# start instructions


## start can
file `start_can0.sh`

```bash
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

services:

can

    [Unit]
    Description=Start can0
    After=network.target

    [Service]
    ExecStart=/home/team07/car_services/start_can.sh

    [Install]
    WantedBy=multi-user.target



MainCar

    [Unit]
    Description=Covesa MQQt server
    After=network.target

    [Service]
    ExecStart=/home/team07/car_services/MainCarServer

    [Install]
    WantedBy=multi-user.target


# Launch app Conytroller

```bash
./Controller 


streamer

gst-launch-1.0 -v udpsrc udpsrc address=239.255.0.1 port=5000 caps="application/x-rtp, payload=96, encoding-name=H264" ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false