import socket
import json
import time
from particle_filter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt

sigma_cam = np.eye(2)
sigma_rf  = np.eye(2) * 0.1

pf = ParticleFilter(num_particles=500,
                    init_state=np.array([0, 0]),
                    init_cov=np.eye(2)*1.0,
                    smooth_alpha_mean=0.9,
                    smooth_alpha_cov=0.9)

plt.ion()
_, ax = plt.subplots()

HOST = '169.254.149.136'
PORT = 8000
BACKLOG = 1 

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Listening from camera centroids on {HOST}:{PORT}…")

try:
    conn, addr = server.accept()
    print(f"Accepted connection from {addr}")

    buf = []
    cam_measurement = (0,0)
    while True:
        chunk = conn.recv(1024)
        if not chunk:
            break

        buf.append(chunk)
        raw = b''.join(buf)

        try:
            payload = json.loads(raw.decode('utf-8'))
            cam_measurement = (payload['x'], payload['y'])
            buf = []
        except (ValueError, KeyError) as e:
            print("Failed to parse JSON:", e)

        pf.predict(pf.motion_model)
        pf.update(cam_measurement, sigma_cam, 'cam')
        pf.resample()
        pf.plot(ax)

        

finally:
    conn.close()
    server.close()
