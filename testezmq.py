import zmq

# Cria contexto
context = zmq.Context()

# Cria socket subscriber
socket = context.socket(zmq.SUB)

# Substitua pelo IP do computador onde o publisher roda
publisher_ip = "192.168.3.217"   # <- coloque aqui o IP do PC do publisher
socket.connect(f"tcp://{publisher_ip}:5555")

# Inscreve em todos os tópicos (string vazia = recebe tudo)
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("Subscriber conectado, aguardando mensagens...")

while True:
    msg = socket.recv_string()
    print(f"Recebido: {msg}")
