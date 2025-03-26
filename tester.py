import socket
import csv
import time
import struct

HOST_SOCKET = "/tmp/illixr-host"
POSE_FILE = '/scratch/prashanth/ILLIXR/build/poses.csv'

sample_conn = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
sample_conn.connect(HOST_SOCKET)


with open(POSE_FILE, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for i, row in enumerate(csv_reader):

        float_values = [float(value) for value in row]

        if (i%10==0):
            float_values = [float(0)] + float_values
            data = struct.pack('8f', *float_values)
            sample_conn.sendall(data)

            data = sample_conn.recv(16)
            d1, d2 = struct.unpack('dd', data)
            print("Received time, bytes:", d1, d2)

            time.sleep(5)

        if (i%10==1):
            float_values = [float(1)] + float_values
            data = struct.pack('8f', *float_values)
            sample_conn.sendall(data)

            data = sample_conn.recv(16)
            d1, d2 = struct.unpack('dd', data)
            print("Received time, bytes:", d1, d2)
            
            time.sleep(5)
