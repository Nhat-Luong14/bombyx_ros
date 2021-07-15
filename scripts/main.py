#!/usr/bin/env python3
import datetime 
import serial
import time
from threading import Thread
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

import config
from savelog import logging, write_json
from utilities import build_log_src_prior, update_log_p_src
from utilities import entropy, get_moves, get_p_src_found, get_p_sample


"""
Serial communication with the arduino. Read the hit-miss log
The function run continuosly in a seperate thread 
Param hit_log: list of recently detection
"""
def read_arduino(hit_log):
    while True:
        if ser2.in_waiting > 0:  # exist message from Arduino in the serial buffer
            line = ser2.readline().decode('utf-8').rstrip() 
            hit_log.append(line)   # put the detection into the queue

        global stop_threads 
        if stop_threads:
            break

"""
Event handle fuction. Save the robot's coordinate when recieve a message from Hector SLAM
Param data: message data form Hector SLAM in topic /slam_out_pose
"""
def callback(data):
    # global pos
    x = data.pose.position.x
    y = data.pose.position.y
    pos.append((x,y))
    rospy.loginfo("Current position: ({},{})".format(x, y))

    
"""
Create a subscriber to topic /slam_out_pose from Hector SLAM and assign 
an event handle fuction when reciving a message.
""" 
def location_listener():
    rospy.init_node('bombyx', anonymous=True)
    rospy.Subscriber("/slam_out_pose", PoseStamped, callback)



def planning(hit_log, pos):
    time.sleep(5)           # wait for raspberry and arduino establish connecting
    log_file = logging()    # create new json file for logging 
    src_found = False
    
    xs = np.linspace(config.x_bounds[0], config.x_bounds[1], config.grid[0])
    ys = np.linspace(config.y_bounds[0], config.y_bounds[1], config.grid[1])    
    
    log_p_src = build_log_src_prior('uniform', xs=xs, ys=ys)    # initialize source distribution
    gas_hit = 0  # 1 for hit and 0 for miss
    s = entropy(log_p_src)
    pos_now = pos.pop()

    for t_ctr, t in enumerate(np.arange(0, config.max_dur, config.dt)):
        # check if source has been found
        if np.linalg.norm(np.array(pos_now) - np.array(config.src_pos)) < config.src_radius:
            src_found = True
            break

        # check if user want to stop the thread
        global stop_threads 
        if stop_threads:
            break

        # pick next move so as to maximally decrease expected entropy
        moves = get_moves(pos_now, step=config.step_size)
        delta_s_expected_list = []

        # estimate expected decrease in p_source entropy for each possible move
        for move in moves:
            # set entropy increase to inf if move is out of bounds
            if not round(config.x_bounds[0], 6) <= round(move[0], 6) <= round(config.x_bounds[1], 6):
                delta_s_expected_list.append(np.inf)
                continue
            elif not round(config.y_bounds[0], 6) <= round(move[1], 6) <= round(config.y_bounds[1], 6):
                delta_s_expected_list.append(np.inf)
                continue

            # get probability of finding source
            p_src_found = get_p_src_found(pos=move, xs=xs, ys=ys, log_p_src=log_p_src)
            p_src_not_found = 1 - p_src_found

            # loop over probability and expected entropy decrease for each sample
            sample_domain = [0, 1]  # miss and hit
            p_samples = np.nan * np.zeros(len(sample_domain))
            delta_s_given_samples = np.nan * np.zeros(len(sample_domain))

            for ctr, h in enumerate(sample_domain):
                # probability of sampling h at pos
                p_sample = get_p_sample(pos=move, xs=xs, ys=ys, h=h, log_p_src=log_p_src)

                # posterior distribution from sampling h at pos
                log_p_src_ = update_log_p_src(pos=pos_now, xs=xs, ys=ys, h=h, log_p_src=log_p_src)

                # decrease in entropy for this move/sample
                s_ = entropy(log_p_src_)
                delta_s_given_sample = s_ - s

                p_samples[ctr] = p_sample
                delta_s_given_samples[ctr] = delta_s_given_sample

            delta_s_src_not_found = p_samples.dot(delta_s_given_samples)    # get expected entropy decrease given source not found
            delta_s_src_found = -s  # get entropy decrease given src found            
            delta_s_expected = (p_src_found * delta_s_src_found) + (p_src_not_found * delta_s_src_not_found) # compute total expected entropy decrease
            delta_s_expected_list.append(delta_s_expected)

        direction = ["stop", "forward", "backward", "left", "right"]
        index = np.argmin(delta_s_expected_list)
        
        ser1.write(str(index).encode())
        while not (ser1.in_waiting > 0):  # exist message from Arduino in the serial buffer
            pass
        print("stopped")    # feedback to raspberry pi
        time.sleep(4)       # wait 4s for gas detection at new location

        gas_hit = 1 if('h' in hit_log) else 0
        pos_now = pos.pop()

        # update source posterior
        log_p_src = update_log_p_src(pos=pos_now, xs=xs, ys=ys, h=gas_hit, log_p_src=log_p_src)
        s_old = s
        s = entropy(log_p_src)

        log_info = {"time":str(datetime.datetime.now()),
            "expected_pos": moves[index],
            "pos": pos_now,
            "move": direction[index],
            "gas_hit": gas_hit,
            "expected_delta_s": delta_s_expected_list[index],
            "delta_s": s - s_old
        }
        write_json(log_info,log_file)
          
    else:
        src_found = False

    if src_found:
        print('Source found!!!!')
    else:
        print('Source not found~')            



if __name__ == '__main__':
    # establish connection to Ardunio
    ser1 = serial.Serial('/dev/ttyACM1', 115200, timeout=1)     #Arduino Uno for Gas sensor 
    ser2 = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)     #Arduino Mega for Motors
    time.sleep(1)
    ser1.flushInput()
    ser2.flushInput()
                    
    pos = deque([(0,0)], maxlen=100)    # use dequeue to share the positon for thread using
    hit_log = deque([], maxlen=50)      # use dequeue to share the log to diffirent threads to read & write
    
    location_listener()              # ros node to sibcribe to SLAM to get position data

    p1 = Thread(target=read_arduino, args=(hit_log,))
    p2 = Thread(target=planning, args=(hit_log,pos,))

    stop_threads = False

    p1.start()
    p2.start()
    rospy.spin()

    stop_threads = True

    p1.join()
    p2.join()