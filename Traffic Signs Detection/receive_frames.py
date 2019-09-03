import socket
from process_frame import process_frame
import cv2

# initialize the connection
def socket_init(host, port):
    try : 
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("[INFO] Created socket!")

        server_socket.bind((host, port))
        server_socket.listen(1)
        print("[INFO] Listening on IP {} and port {}".format(socket.gethostbyname(socket.gethostname()), port))

    except : 	
        print("[ERR] Couldn't create server socket")

    try:
        conn, addr = server_socket.accept()
        print("[INFO] Connected via {}: {}".format(addr[0], addr[1]))

    except:
        print("[ERR] Couldn't connect with the client")
    
    return server_socket, conn


# receive a frame
def receive_one_frame(connection, buff_size):
    data = ''
    n = 0

    while n < 240:
        n += 1
        part = connection.recv(buff_size)
        if not part: break
        
        # if for some reason the model received a portion of the supposed buffer size
        while len(part) < buff_size: 
            part += connection.recv(buff_size - len(part))

        part_hex = bytes.hex(part)
        data += part_hex

    return data