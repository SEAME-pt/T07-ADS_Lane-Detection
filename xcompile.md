# X compiling

## location:

.
├── jetson-sysroot
└── T07-ADS_Lane-Detection
	└── at this location

SEAME/T07-ADS_Lane_Detection

```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

Depois activar o container:
```bash
docker compose up
```

Isso gera os volumes bind onde a pasta local eh a mesma no servidor entao pode editar o codigo normalmente

Para compilar o controller por exemplo:
```bash
docker exec -it [container do controller] /bin/bash
```

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake -DCMAKE_BUILD_TYPE=Release
make
```

Copy  object file to the car:
Be sure car is active (Jetson nano is ON and at the specified address on the same network)
```bash
scp ControllerExec team07@10.21.221.47:/home/team07
```
isso vai copiar o executavel pra home do carro , dai so testar.
