import serial
import math
import time
import codecs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
from queue import Queue
from collections import deque
from threading import Lock


# Anchor positions for atrium demo
anchor_positions = {
    1: [-2.9, -5.5, 2.83],
    2: [-5.2, 7.75, 2.95],
    3: [3.0, 4.0, 0.8],
    4: [2.0, 6.2, 2.80],
    5: [6.15, -5.5, 1.65],
}

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# Readline function borrowed from https://github.com/pyserial/pyserial/issues/216#issuecomment-369414522
# It seems to be necessary to speed up PySerial on Windows but not on Linux
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s
    
    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

# Create serial port. Ensure that the port is actually where the Wio-E5 mini is connected.
com_port = 14
recv_ser = serial.Serial(
    port='COM'+str(com_port),
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
)

rl = ReadLine(recv_ser)

def send_command(ser: serial.Serial, cmd: str):
    ser.write((cmd+'\r\n').encode('utf-8'))

# Initialize receiver
send_command(recv_ser, 'AT+MODE=TEST')
time.sleep(0.1)
send_command(recv_ser, 'AT+TEST=RFCFG,915,SF8,500,12,15,22,ON,OFF,OFF')
time.sleep(0.1)
send_command(recv_ser, 'AT+TEST=RXLRPKT')
time.sleep(1.0)

# Position calculation. 
# For more information, see https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:446:2016:446-2016-08:algebraicmultilaterationnorrdine.pdf
def compute_position(tag_dists, anchor_positions):
    anchors = []
    dists = []
    norms = []
    for anchor_num in tag_dists.keys():
        if anchor_num not in anchor_positions.keys():
            continue
        dists.append([tag_dists[anchor_num]])
        anchors.append(anchor_positions[anchor_num])

    anchors = np.array(anchors)
    dists = np.array(dists)
    A = np.concatenate((np.ones((anchors.shape[0], 1)), -2*anchors), axis=1)
    b = np.square(dists) - np.sum(np.square(anchors), axis=1)[:, np.newaxis]
    A_pinv = np.linalg.inv(A.T@A)
    x = A_pinv@A.T@b

    return x.flatten()[1:]


def update_position(queue_out):
    tags = {}
    delta_t = 0.032
    freq1 = 10
    freq2 = 10
    num1 = (2*math.pi*delta_t*freq1)
    alpha1 = num1/(num1+1)
    num2 = (2*math.pi*delta_t*freq2)
    alpha2 = num2/(num2+1)
    tag_pos = {}
    
    while(True):
        # If incoming messages look garbled, comment the below line and uncomment the line after.
        recv_msg = rl.readline()
        # recv_msg = recv_ser.readline()

        if recv_msg[7:10] != bytearray('RX ', 'utf-8'):
            continue
        try:
            msg = str(codecs.decode(recv_msg[11:-3], 'hex').decode('utf-8')).split(",")
        except:
            continue

        # Error checking / correction for LoRa transmission
        anchor_nums = msg[:10:2]
        if len(anchor_nums) < 5:
            continue
        anchor_num = max(set(anchor_nums), key=anchor_nums.count)
        if anchor_nums.count(anchor_num) < 4:
            continue
        anchor_num = int(anchor_num)

        tag_nums = msg[1:10:2]
        if len(tag_nums) < 5:
            continue
        tag_num = max(set(tag_nums), key=tag_nums.count)
        if tag_nums.count(tag_num) < 4:
            continue
        tag_num = int(tag_num)

        if tag_num not in tags.keys():
            tags[tag_num] = {}

        dists = msg[9:]
        if len(dists) < 5:
            continue
        dist = max(set(dists), key=dists.count)
        if dists.count(dist) < 4:
            continue
        try:
            dist = float(dist)
        except:
            continue

        # Discrete low pass filter on ranges. 
        # For more information, see https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
        if anchor_num not in tags[tag_num].keys():
            tags[tag_num][anchor_num] = dist
        else:
            tags[tag_num][anchor_num] = alpha1 * dist + (1-alpha1) * tags[tag_num][anchor_num]


        for tag in tags.keys():
            if len(tags[tag].keys()) < 4:
                continue

            if tag not in tag_pos.keys():
                tag_pos[tag] = compute_position(tags[tag], anchor_positions)
            else:
                tag_pos[tag] = alpha2 * compute_position(tags[tag], anchor_positions) + (1-alpha2) * tag_pos[tag]

            queue_out.put(tag_pos)


tag_data = deque(maxlen=50)
all_pos = []
def animate(i):
    ax1.clear()
    for anchor_pos in anchor_positions.values():
        ax1.scatter([anchor_pos[0]], [anchor_pos[1]])

    while not tag_queue.empty():
        tag_data.append(tag_queue.get())
    if len(tag_data) == 0:
        return

    # Moving average over position. Can be removed to reduce latency.
    weights = np.linspace(0.2, 1.0, len(tag_data))
    tag_pos = {}
    for tag in tag_data[0].keys():
        tag_positions = np.array([tag_pos[tag] for tag_pos in tag_data])
        ax1.plot(tag_positions[:,0], tag_positions[:,1])
        tag_positions = weights[:, None] * tag_positions
        tag_pos[tag] = tag_positions.sum(0) / weights.sum()

        tag_pos[tag] = tag_data[-1][tag]

    for tag in tag_pos.values():
        print(tag)
        ax1.scatter([tag[0]], [tag[1]])
        all_pos.append([tag[0], tag[1]])


tag_queue = Queue()
serial_thread = Thread(target = update_position, args=(tag_queue,))
serial_thread.start()
ani = animation.FuncAnimation(fig, animate, interval = 40, cache_frame_data=False)
plt.show()

np.save("./positions", all_pos)
