FROM hmacielp/crosscompile-covesa

RUN apt update && apt-get install -y -f \
    libzmq3-dev \
    build-essential \
    libgpiod-dev \
    libi2c-dev

# Instalar a versão mais recente do cppzmq
RUN git clone https://github.com/zeromq/cppzmq.git /tmp/cppzmq && \
    cp /tmp/cppzmq/zmq.hpp /usr/include/ && \
    rm -rf /tmp/cppzmq

WORKDIR /workspace

CMD ["/bin/bash"]
