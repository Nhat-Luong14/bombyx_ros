#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

import serial
import time
from threading import Thread
from collections import deque
import numpy as np
import config
from utilities import build_log_src_prior, update_log_p_src
from utilities import entropy, get_moves, get_p_src_found, get_p_sample
# from plume_processing import IdealInfotaxisPlume
# from copy import copy

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
    # pos = (x,y)
    pos.append((x,y))
    # rospy.loginfo("Current position: ({},{})".format(x, y))

    
"""
Create a subscriber to topic /slam_out_pose from Hector SLAM and assign 
an event handle fuction when reciving a message.
""" 
def location_listener():
    rospy.init_node('bombyx', anonymous=True)
    rospy.Subscriber("/slam_out_pose", PoseStamped, callback)


def planning(hit_log, pos):
    time.sleep(5)      # wait for raspberry and arduino establish connecti

    xs = np.linspace(config.x_bounds[0], config.x_bounds[1], config.grid[0])
    ys = np.linspace(config.y_bounds[0], config.y_bounds[1], config.grid[1])    
    
    # initialize source distribution (prior probability)
    log_p_src = build_log_src_prior('uniform', xs=xs, ys=ys)
     
    traj = [pos.pop()]  # position sequence
    h_ = 0              # gas detection. 1 for hit and 0 for miss

    for t_ctr, t in enumerate(np.arange(0, config.max_dur, config.dt)):
        # check if source has been found
        new_pos = pos.pop()
        if np.linalg.norm(np.array(new_pos) - np.array(config.src_pos)) < config.src_radius:
            src_found = True
            break

        # check if user want to stop the thread
        src_found = False
        global stop_threads 
        if stop_threads:
            break
        
        # update source posterior
        log_p_src = update_log_p_src(pos=new_pos, xs=xs, ys=ys, h=h_, log_p_src=log_p_src)
        s = entropy(log_p_src)

        # pick next move so as to maximally decrease expected entropy
        moves = get_moves(new_pos, step=config.step_size)
        delta_s_expecteds = []

        # estimate expected decrease in p_source entropy for each possible move
        for move in moves:
            # set entropy increase to inf if move is out of bounds
            if not round(config.x_bounds[0], 6) <= round(move[0], 6) <= round(config.x_bounds[1], 6):
                delta_s_expecteds.append(np.inf)
                continue
            elif not round(config.y_bounds[0], 6) <= round(move[1], 6) <= round(config.y_bounds[1], 6):
                delta_s_expecteds.append(np.inf)
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
                log_p_src_ = update_log_p_src(pos=new_pos, xs=xs, ys=ys, h=h, log_p_src=log_p_src)

                # decrease in entropy for this move/sample
                s_ = entropy(log_p_src_)
                delta_s_given_sample = s_ - s

                p_samples[ctr] = p_sample
                delta_s_given_samples[ctr] = delta_s_given_sample

            delta_s_src_not_found = p_samples.dot(delta_s_given_samples)    # get expected entropy decrease given source not found
            delta_s_src_found = -s  # get entropy decrease given src found            
            delta_s_expected = (p_src_found * delta_s_src_found) + (p_src_not_found * delta_s_src_not_found) # compute total expected entropy decrease
            delta_s_expecteds.append(delta_s_expected)

        direction = ["stop", "forward", "backward", "left", "right"]
        print(delta_s_expecteds)
        index = np.argmin(delta_s_expecteds)
        print(index)
        print(direction[index])
        # next_pos = moves[index]   # choose move that decreases p_source entropy the most
        traj.append(new_pos)

        
        ser1.write(str(index).encode())
        print("sent cmd")
        while not (ser1.in_waiting > 0):  # exist message from Arduino in the serial buffer
            pass
        print("stopped")   

        time.sleep(4)   # wait 1s after moving for gas detection at new location
        log = list(hit_log)
        h=1 if('h' in log) else 0
        print(h)
        
    else:
        src_found = False

    if src_found:
        print('Source found after {} time steps ({} s)'.format(
            len(traj), len(traj) * config.dt))
    else:
        print('Source not found after {} time steps ({} s)'.format(
            len(traj), len(traj) * config.dt))            


if __name__ == '__main__':
    # establish connection to Ardunio
    ser1 = serial.Serial('/dev/ttyACM0', 115200, timeout=1)     #Arduino Uno for Gas sensor 
    ser2 = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)     #Arduino Mega for Motors
    time.sleep(1)
    ser1.flushInput()
    ser2.flushInput()
                    
    pos = deque([(0,0)], maxlen=100)    # use dequeue to share the positon for thread using
    hit_log = deque([], maxlen=20)      # use dequeue to share the log to diffirent threads to read & write
    
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