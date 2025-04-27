########################################################################
#
# Full pipeline for person tracking and tag association
# 
# Computer vision module adapted from: https://github.com/stereolabs/zed-sdk/blob/master/object%20detection/image%20viewer/python/object_detection_image_viewer.py
#
########################################################################


# Package imports 
import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import serial
import math
import time
import codecs
import socket
import json
from threading import Lock
from threading import Thread
from queue import Queue
from collections import deque

# Input stream arguments 
def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")


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
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

com_port = 14
recv_ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    #parity=serial.PARITY_NONE,
    #stopbits=serial.STOPBITS_ONE,
    #bytesize=serial.EIGHTBITS,
)

rl = ReadLine(recv_ser)

def send_command(ser: serial.Serial, cmd: str):
    ser.write((cmd+'\r\n').encode('utf-8'))

print ("Initializing LoRa Receiver")
send_command(recv_ser, 'AT+MODE=TEST')
time.sleep(0.1)
send_command(recv_ser, 'AT+TEST=RFCFG,915,SF8,500,12,15,22,ON,OFF,OFF')
time.sleep(0.1)
send_command(recv_ser, 'AT+TEST=RXLRPKT')
time.sleep(1.0)
anchor_positions = {
    1: [-2.9, -5.5, 2.83],
    2: [-5.2, 7.75, 2.95],
    3: [-0.7, -2.8, 1.0],
    4: [2.0, 6.2, 2.80],
    5: [6.15, -5.5, 1.65],
}

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

def send_command(ser: serial.Serial, cmd: str):
    ser.write((cmd+'\r\n').encode('utf-8'))

    send_command(recv_ser, 'AT+MODE=TEST')
    time.sleep(0.1)
    send_command(recv_ser, 'AT+TEST=RFCFG,915,SF8,500,12,15,22,ON,OFF,OFF')
    time.sleep(0.1)
    send_command(recv_ser, 'AT+TEST=RXLRPKT')
    time.sleep(1.0)

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
        # recv_msg = rl.readline()
        recv_msg = recv_ser.readline()
        if recv_msg[7:10] != bytearray('RX ', 'utf-8'):
            continue
        try:
            msg = str(codecs.decode(recv_msg[11:-3], 'hex').decode('utf-8')).split(",")
        except:
            continue


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

tag_queue = Queue()
serial_thread = Thread(target = update_position, args=(tag_queue,))
serial_thread.start()


def main():
    id_mapping={}
    IP = '169.254.149.136'
    PORT = 8000
    def get_color(track_id):
        if track_id == 0:       # TAG 0 IS BRIGHT RED
            id_mapping[track_id] = tuple([0,0,255])
        elif track_id not in id_mapping: 
            id_mapping[track_id] = tuple(random.randint(175,255) for _ in range(3))
        return id_mapping[track_id]

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable object detection module
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

    if obj_param.enable_tracking:
        zed.enable_positional_tracking()

    zed.enable_object_detection(obj_param)

    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 60
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]

    objects = sl.Objects()
    image = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()
    centroid_file = open("centroids.txt", "w")

    # Compute homography matrix from Image space -> World space
    image_space_markers = []
    world_space_markers = [(0,0), (3,0), (6,3), (3,3), (0,3), (-3,3)]

    def on_mouse_click(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            image_space_markers.append((x,y))
            cv2.circle(img, (x,y), 5, (255,0,0), -1)
            cv2.imshow('Select the markers on the image', img)

    img = None
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.RIGHT)
        zed.retrieve_objects(objects, obj_runtime_param)

    # Get image as numpy array
    img = image.get_data()
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert to OpenCV BGR format

    cv2.namedWindow('Select the markers on the image')
    cv2.setMouseCallback('Select the markers on the image', on_mouse_click)

    while True:
        cv2.imshow('Select the markers on the image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    image_space_markers = np.array(image_space_markers, dtype=np.float32)
    world_space_markers = np.array(world_space_markers, dtype=np.float32)

    H, status = cv2.findHomography(image_space_markers, world_space_markers, method=cv2.RANSAC)
    print(f"Estimated H matrix:\n {H}") 
   
    # Establish socket connection with laptop receiving centroid coordinates 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((IP, PORT))

    while True:
        
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            ########################################################################
            # Near field

            while not tag_queue.empty():
                tag_data.append(tag_queue.get())
            if len(tag_data) != 0:
                weights = np.linspace(0.2, 1.0, len(tag_data))
                tag_pos = {}
                for tag in tag_data[0].keys():
                    tag_positions = np.array([tag_pos[tag] for tag_pos in tag_data])
                    tag_positions = weights[:, None] * tag_positions
                    tag_pos[tag] = tag_positions.sum(0) / weights.sum()

            ##########################################################################
            
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            zed.retrieve_objects(objects, obj_runtime_param)

            # Get image as numpy array
            img = image.get_data()

            # Draw 2D bounding boxes around all tracked people 
            if objects.is_new:
                min_diff = 100
                curr_zero = None
                for obj in objects.object_list:
                    if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        bbox_2d = obj.bounding_box_2d  # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
                        x_min = int(min([pt[0] for pt in bbox_2d]))
                        y_min = int(min([pt[1] for pt in bbox_2d]))
                        x_max = int(max([pt[0] for pt in bbox_2d]))
                        y_max = int(max([pt[1] for pt in bbox_2d]))

                        bottom_left = (x_min, y_min)
                        bottom_right = (x_max, y_min)
                        center_x = (x_min + x_max) / 2
                        if (center_x < 652):
                            centroid = bottom_right
                        else:
                            centroid = bottom_left

                        cam_homogenous = np.array([centroid[0], centroid[1], 1.0])
                        xyw = H @ cam_homogenous
                        xyw = np.array([xyw[0], xyw[1]], dtype=np.float32)

                        translation = np.array([-0.5, -1.5])
                        scale_x = 1.1
                        scale_y = 1.4

                        xyw += translation
                        xyw[0] /= scale_x
                        xyw[1] /= scale_y

                        world_frame_xy = np.array([xyw[0], xyw[1]])

                        # Check if incoming tag coordinates are associated with this person detection 
                        if (not len(tag_pos.keys())==0):
                            curr_tag_pos = tag_pos[list(tag_pos.keys())[0]]
                            print(curr_tag_pos)
                            bbx_world = np.array(world_frame_xy)
                            tag_world = curr_tag_pos[:2]
                            diff = np.linalg.norm(bbx_world-tag_world)
                            if(diff < 1.0): 
                                print("WORLD", world_frame_xy[0], world_frame_xy[1])
                                obj.id = 0 # assign ID 0 to tagged person 
                  
                        # Only send tagged person (ID 0) coordinates to particle filter
                        if obj.id == 0: 
                            message = json.dumps({'x':float(world_frame_xy[0]), 'y':float(world_frame_xy[1])})
                            sock.send((message.encode('utf-8')))
                            time.sleep(0.01)
                            centroid_file.write(f"{world_frame_xy[0]},{world_frame_xy[1]}\n")

                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), get_color(obj.id), 2)
                        label_text = f"{obj.label} ID {obj.id}"
                        cv2.putText(img, label_text, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(obj.id), 2)


            cv2.imshow("ZED | 2D Bounding Boxes", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    image.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 
