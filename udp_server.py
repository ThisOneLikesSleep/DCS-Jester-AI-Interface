import socket

def start_udp(sock_d_inbound):
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the address and port
    server_socket.bind(('127.0.0.1', 6217))

    while True:
        data, _ = server_socket.recvfrom(1028)
        message = data.decode('utf-8')
        while not sock_d_inbound.empty():
            _ = sock_d_inbound.get()
        sock_d_inbound.put(message)

