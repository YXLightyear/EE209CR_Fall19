#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:49:04 2019

@author: tiantong
"""

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from math import sin, cos, tan, sqrt, atan2
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt

L = 750.0 
W = 500.0 

r = 25.0 #mm
l = 90.0 #mm

def move(x, dt, u):
    '''
    Implement state transition function.
    x: [x, vx, y, vy, theta, omega(angular velocity)], theta is in [-pi, pi].
    u: [wl, wr]
    dt: time interval
    return next state 
    '''
    v = (u[0] + u[1]) * r / 2
    omega = (u[1] - u[0]) * r / l
    ds = v * dt
    dtheta = omega * dt
    return np.array([x[0] + ds * cos(x[4]), \
                     v * cos(x[4]), \
                     x[2] + ds * sin(x[4]), \
                     v * sin(x[4]), \
                     x[4] + dtheta,\
                     omega])

def straight_intersection_point(s, boundary=[750, 500]):
    '''
    calculate the intersection point between straight laser and boundary.
    s: current state, [x, y, theta]
    boundary: [L, W], default [750, 500]mm.
    return ps, the intersection of straight laser between boundaries.
    '''
    x = s[0]
    y = s[2]
    theta = s[4]
    X = boundary[0]
    Y = boundary[1]
    if theta > 0 and theta < np.pi / 2:
        xs = x + (Y - y) / tan(theta)
        ys = y + (X - x) * tan(theta)
        if xs >= 0 and xs <= X:
            return [xs, Y]
        else:
            return [X, ys]        
        
    elif theta > np.pi / 2 and theta < np.pi:
        xs = x + (Y - y) / tan(theta)
        ys = y - x * tan(theta)
        if xs >= 0 and xs <= X:
            return [xs, Y]
        else:
            return [0, ys]
        
    elif theta > -np.pi and theta < -np.pi / 2:
        xs = x - y / tan(theta)
        ys = y - x * tan(theta)
        if xs >= 0 and xs <= X:
            return [xs, 0]
        else:
            return [0, ys]
            
    elif theta > -np.pi / 2 and theta < 0:
        xs = x - y / tan(theta)
        ys = y + (X - x) * tan(theta)
        if xs >= 0 and xs <= X:
            return [xs, 0]
        else:
            return [X, ys]
    elif theta == 0:
        return [X, y]
    elif theta == np.pi / 2:
        return [x, Y]
    elif theta == np.pi or theta == -np.pi:
        return [0, y]
    else:
        return [x, 0]

def right_intersection_point(s, boundary=[750, 500]):
    '''
    calculate the intersection point between right laser and boundary.
    s: current state, [x, y, theta]
    boundary: [L, W], default [750, 500]mm.
    return ps, the intersection of right laser between boundaries.
    '''
    x = s[0]
    y = s[2]
    theta = s[4]
    X = boundary[0]
    Y = boundary[1]
    if theta > 0 and theta < np.pi / 2:
        xr = x + y / tan(np.pi / 2 - theta)
        yr = y - (X - x) * tan(np.pi / 2 - theta)
        if xr >= 0 and xr <= X:
            return [xr, 0]
        else:
            return [X, yr]        
        
    elif theta > np.pi / 2 and theta < np.pi:
        xr = x + (Y - y) / tan(theta - np.pi / 2)
        yr = y +  (X - x) * tan(theta - np.pi / 2)
        if xr >= 0 and xr <= X:
            return [xr, Y]
        else:
            return [X, yr]
        
    elif theta > -np.pi and theta <= -np.pi / 2:
        xr = x + (Y - y) / tan(theta + np.pi / 2)
        yr = y - x * tan(theta + np.pi / 2)
        if xr >= 0 and xr <= X:
            return [xr, Y]
        else:
            return [0, yr]
            
    elif theta > - np.pi / 2 and theta < 0:
        xr = x - y / tan(theta + np.pi / 2)
        yr = y - x * tan(theta + np.pi / 2)
        if xr >= 0 and xr <= X:
            return [xr, 0]
        else:
            return [0, yr]
    elif theta == 0:
        return [x, 0]
    elif theta == np.pi / 2:
        return [X, y]
    elif theta == np.pi or theta == -np.pi:
        return [x, Y]
    else:
        return [0, y]
    
def normalize_angle(x):
    '''
    Normalize the angle difference to [-pi, pi]
    x: actual angle difference, in the range of [-2 pi, 2 pi]
    return the normalized angle.
    '''
    if x >= np.pi:
        x = x - 2 * np.pi
    if x <= -np.pi:
        x = x + 2 * np.pi
    return x

def residual_x(a, b):
    '''
    Compute residual of state.
    '''
    y = a - b
    y[4] = normalize_angle(y[4])
    return y

def residual_z(a, b):
    '''
    Compute residual of measurement.
    '''
    y = a - b
    y[2] = normalize_angle(y[2])
    return y

def Hx(s):
    '''
    Calculate measurement according to current state.
    s: current state, [x, y, theta].
    returns the measurement, [rs, rr, bearing].
    '''
    x = s[0]
#    vx = s[1]
    y = s[2]
#    vy = s[3]
    theta = s[4]
    vtheta = s[5]
    ps = straight_intersection_point(s)
    pr = right_intersection_point(s)
    rs = sqrt((x - ps[0]) ** 2 + (y - ps[1]) ** 2)
    rr = sqrt((x - pr[0]) ** 2 + (y - pr[1]) ** 2)
#    bearing = abs(theta) # absolute bearing
    bearing = normalize_angle(theta)
    omega = vtheta
    return np.array([rs, rr, bearing, omega])

def state_mean(sigmas, Wm):
    '''
    Compute the mean of state sigma points.
    '''
    x = np.zeros(6)
    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 4]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 4]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = np.sum(np.dot(sigmas[:, 2], Wm))
    x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
    x[4] = atan2(sum_sin, sum_cos)
    x[5] = np.sum(np.dot(sigmas[:, 5], Wm))
    return x

def z_mean(sigmas, Wm):
   '''
   Compute the mean of measurement sigma points.
   '''
   z = np.zeros(4)
   sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
   sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
   z[0] = np.sum(np.dot(sigmas[:, 0], Wm))
   z[1] = np.sum(np.dot(sigmas[:, 1], Wm))
   z[2] = atan2(sum_sin, sum_cos)
   z[3] = np.sum(np.dot(sigmas[:, 3], Wm))
   return z

def build_ukf(x, P, std_r, std_b, dt=1.0):
    '''
    Build UKF.
    x: initial state.
    P: initial covariance matrix.
    std_r: standard var. of laser measurement.
    std_b: standard var. of IMU measurement.
    dt: time interval.
    Plus some defined functions as parameters.
    returns ukf.
    ''' 
    # Calculate sigma points.
    points = MerweScaledSigmaPoints(n=6, alpha=0.001, beta=2, kappa=-3, subtract=residual_x)
    ukf = UKF(dim_x=6, dim_z=4, fx=move, hx=Hx, \
              dt=dt, points=points, x_mean_fn=state_mean, \
              z_mean_fn=z_mean, residual_x=residual_x, residual_z=residual_z)
    ukf.x = np.array(x)
    ukf.P = P
    ukf.R = np.diag([std_r ** 2, std_r ** 2, std_b ** 2, std_b ** 2])
    q1 = Q_discrete_white_noise(dim=2, dt=dt, var=1.0)
    q2 = Q_discrete_white_noise(dim=2, dt=dt, var=1.0)
    q3 = Q_discrete_white_noise(dim=2, dt=dt, var=3.05 * pow(10, -4))
    ukf.Q = block_diag(q1, q2, q3)
#    ukf.Q = np.eye(3) * 0.0001
    return ukf

def noisy_sensor(x, std_r, std_b):
    '''
    Generate the noisy reading.
    x: current state, [x, y, theta]
    std_r: laser std.
    std_b: bearing std.
    returns noisy reading of sensor.
    '''
    sensor_out = Hx(x)
    sensor_out[0] += std_r * np.random.randn()
    sensor_out[1] += std_r * np.random.randn()
    sensor_out[2] += std_b * np.random.randn()
    sensor_out[3] += std_b * np.random.randn()
    return np.array(sensor_out)


def noisy_move(x, dt, u, std_an):
    '''
    Generate the noisy state.
    x: current state.
    u: angular vel of left and right wheels, [wl, wr]
    std_an: std. of angular vel. of wheels.
    '''
    vl = (u[0] + std_an * np.random.randn()) * r
    vr = (u[1] + std_an * np.random.randn()) * r
    v = (vl + vr) / 2.0
    omega = (vr - vl) / l
    ds = v * dt
    dtheta = omega * dt
    return np.array([x[0] + ds * cos(x[4]), \
                     v * cos(x[4]), \
                     x[2] + ds * sin(x[4]), \
                     v * sin(x[4]), \
                     x[4] + dtheta,\
                     omega])


#def run_localization(cmds, x, P, std_an, std_r, std_b,\
#                     dt=1.0, ellipse_step=1, step=10):
#    plt.figure()
#    ukf = build_ukf(x, P, std_r, std_b, dt=1.0)
#    sim_pos = ukf.x.copy()
#    
#    np.random.seed(1)
#    traj = []
#    xs = []
#    for i, u in enumerate(cmds):
#        sim_pos = noisy_move(sim_pos, dt/step, u, std_an)
#        traj.append(sim_pos)
#        if i % step == 0:
#            ukf.predict(u=u)
#            if i % ellipse_step == 0:
#                cov = np.array([[ukf.P[0, 0], ukf.P[2, 0]],
#                                [ukf.P[0, 2], ukf.P[2, 2]]])
#                plot_covariance_ellipse((ukf.x[0], ukf.x[2]), cov, std=6, facecolor='k', alpha=0.3)
#            z = noisy_sensor(sim_pos, std_r, std_b)
#            ukf.update(z) 
#            if i % ellipse_step == 0:
#                cov = np.array([[ukf.P[0, 0], ukf.P[2, 0]],
#                                [ukf.P[0, 2], ukf.P[2, 2]]])
#                plot_covariance_ellipse((ukf.x[0], ukf.x[2]), cov, std=6, facecolor='g', alpha=0.8)
#        xs.append(ukf.x.copy())
#    xs = np.array(xs)
#    traj = np.array(traj)
#    plt.plot(traj[:, 0], traj[:, 2], color='b', linewidth=2, label='Actual movement')
#    plt.plot(xs[:, 0], xs[:, 2], 'r--', linewidth=2, label='Estimated movement')
#    plt.legend()
##    plt.legend(handles=[l1,l2],labels=['Actual movement','Estimated movement'],loc='best')
#    plt.title("UKF Robot Localization")
#    plt.axis([0, 750, 0, 500])
#    plt.grid()
#    plt.show()
#    return xs, traj

def run_localization(cmds, x, x_guess, P, std_an, std_r, std_b,\
                     dt=1.0, ellipse_step=1, step=10, show=True):
    if show:
        plt.figure(figsize=(9, 6))
    
    ukf = build_ukf(x_guess, P, std_r, std_b, dt=1.0)
    sim_pos = x
#    ideal_pos = ukf.x.copy()
    
    np.random.seed(1)
#    ideal = []
    traj = []
    xs = []
    for i, u in enumerate(cmds):
#        ideal_pos = move(ideal_pos, dt/step, u)
        sim_pos = noisy_move(sim_pos, dt/step, u, std_an)
        traj.append(sim_pos)
#        ideal.append(ideal_pos)
        if i % step == 0:
            ukf.predict(u=u)
            if i % ellipse_step == 0 and show:
                cov = np.array([[ukf.P[0, 0], ukf.P[2, 0]],
                                [ukf.P[0, 2], ukf.P[2, 2]]])
                plot_covariance_ellipse((ukf.x[0], ukf.x[2]), cov, std=6, facecolor='k', alpha=0.3)
            z = noisy_sensor(sim_pos, std_r, std_b)
            ukf.update(z) 
            if i % ellipse_step == 0 and show:
                cov = np.array([[ukf.P[0, 0], ukf.P[2, 0]],
                                [ukf.P[0, 2], ukf.P[2, 2]]])
                plot_covariance_ellipse((ukf.x[0], ukf.x[2]), cov, std=6, facecolor='g', alpha=0.8)
        xs.append(ukf.x.copy())
    xs = np.array(xs)
    traj = np.array(traj)
#    ideal = np.array(ideal)
    if show:
        plt.plot(traj[:, 0], traj[:, 2], color='b', linewidth=2, label='Actual trajectory')
        plt.plot(xs[:, 0], xs[:, 2], 'r--', linewidth=2, label='Estimated trajectory')
    #    plt.plot(ideal[:, 0], ideal[:, 1], 'y', linewidth=2, label='Ideal trajectory')
        plt.legend()
        plt.title("UKF Robot Localization")
        plt.axis([0, 750, 0, 500])
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.grid()
        plt.show()
    return xs, traj

def sensor_traces(x, cmds, dt, step, std_an, std_r, std_b):
    np.random.seed(1)
    noisy_x = x
    sensor = noisy_sensor(noisy_x, std_r, std_b)
    t = 0
    rs_trace = [sensor[0]]
    rr_trace = [sensor[1]]
    bearing_trace = [sensor[2]]
    anvel_trace = [sensor[3]]
    time = [t]
    
    for u in cmds:
        noisy_x = noisy_move(noisy_x, dt/step, u, std_an)
        sensor = noisy_sensor(noisy_x, std_r, std_b)
        t += dt / step
        rs_trace.append(sensor[0])
        rr_trace.append(sensor[1])
        bearing_trace.append(sensor[2])
        anvel_trace.append(sensor[3])
        time.append(t)
    rs_trace = np.array(rs_trace)
    rr_trace = np.array(rr_trace)
    bearing_trace = np.abs(np.array(bearing_trace))
    anvel_trace = np.abs(np.array(anvel_trace))
    time = np.array(time)
    
    plt.figure(figsize=(9, 6))
    plt.plot(time, bearing_trace, color='b', linewidth=2, label='Output of IMU absolute bearing measurement')
    plt.plot(time, anvel_trace, color='k', linewidth=2, label='Output of IMU angular rate measurement')
#    plt.plot(time, rs_trace, color='g', linewidth=2, label='Output of range sensor in the front of robot')
#    plt.plot(time, rr_trace, 'r', linewidth=2, label='Output of range sensor on the right of robot')
    plt.legend()
#    plt.title("Noisy output of range sensors")
    plt.title("Noisy output of IMU")
    plt.xlabel('t (s)')
    
    plt.ylabel('Distance to boundary (mm)')
#    plt.ylabel('rad or rad/s')
    plt.grid()
    plt.show()
    
def generate_ideal_noisy_traj(x, cmds, dt, step, std_an):
    '''
    Generate ideal traj and noisy traj when given certain control inputs.
    '''
    np.random.seed(1)
    ideal_traj = []
    noisy_traj = []
    ideal_x = x
    noisy_x = x
    for i, u in enumerate(cmds):
       ideal_x = move(ideal_x, dt / step, u)
       ideal_traj.append(ideal_x)
       noisy_x = noisy_move(noisy_x, dt / step, u, std_an)
       noisy_traj.append(noisy_x)
    ideal_traj = np.array(ideal_traj)
    noisy_traj = np.array(noisy_traj)
    plt.figure(figsize=(9, 6))
    plt.plot(ideal_traj[:, 0], ideal_traj[:, 2], color='g', linewidth=2, label='Ideal trajectory')
    plt.plot(noisy_traj[:, 0], noisy_traj[:, 2], 'r--', linewidth=2, label='Noisy trajectory')
    plt.legend()
    plt.title("Ideal trajectory and noisy trajectory")
    plt.axis([0, 750, 0, 500])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.grid()
    plt.show()
    
def accuracy(actual, estimate, show=False):
    '''
    return accuracy of estimated traj.
    '''
    diff = actual - estimate
    diff_x = np.abs(diff[:, 0])
    diff_y = np.abs(diff[:, 2])
    diff_theta = np.abs(diff[:, 4])
    
    diff_vx = np.abs(diff[:, 1])
    diff_vy = np.abs(diff[:, 3])
    diff_o = np.abs(diff[:, 5])
    
    avg_x = np.mean(diff_x)
    avg_y = np.mean(diff_y)
    avg_theta = np.mean(diff_theta)
    std_x = np.std(diff_x)
    std_y = np.std(diff_y)
    std_theta = np.std(diff_theta)
    
    avg_vx = np.mean(diff_vx)
    avg_vy = np.mean(diff_vy)
    avg_o = np.mean(diff_o)
    std_vx = np.std(diff_vx)
    std_vy = np.std(diff_vy)
    std_o = np.std(diff_o)
    if show:
        t = np.linspace(0.1, len(diff_x)*0.1, num=len(diff_x))
        
        plt.figure(figsize=(9, 5))
        
        p1 = plt.subplot(311)
#        plt.plot(t, diff_x, 'r')
        plt.plot(t, diff_x, 'r')
        plt.grid()
        p1.set_ylabel('Errors in x (mm)')
         
        p2 = plt.subplot(312)
        plt.plot(t, diff_y, 'g')
        plt.grid()
        p2.set_ylabel('Errors in y (mm)')
        
        p3 = plt.subplot(313)
        plt.plot(t, diff_theta, 'b')
        plt.grid()
        p3.set_ylabel('Errors in theta (rad)')
        plt.xlabel('t (s)')
        
        plt.show()
    return avg_x, avg_y, avg_theta, std_x, std_y, std_theta, avg_vx, avg_vy, avg_o, std_vx, std_vy, std_o
if __name__ == "__main__":
    std_an = 130 * 2 * np.pi / 60.0 * 0.05         
    std_r = 1200 * 0.03            
    std_b = np.radians(0.1)        
    x = np.array([90, 0, 90, 0, np.pi / 3, 0])
#    x_guess = np.array([90, 0, 90, 0, np.pi / 3, 0])
    x_guess = np.array([150, 10, 90, 10, np.pi / 2, 0])
    show = True
    
#    P = np.diag([0.01, 0.01, 0.01, 0.01, 3.05 * pow(10, -6), 3.05 * pow(10, -6)]) 
    P = np.diag([100, 100, 100, 100, 1.0, 1.0])# initial covariance
    cmds = [np.array([0.6, 0.7])] * 40
    cmds.extend([np.array([0.6, 0.8])])
    cmds.extend([cmds[-1]]*50)
    cmds.extend([np.array([0.8, 0.5])])
    cmds.extend([cmds[-1]]*400)
    cmds.extend([np.array([0.8, 0.3])])
    cmds.extend([cmds[-1]]*150)
    cmds.extend([np.array([0.8, 0.2])])
    cmds.extend([cmds[-1]]*60)
#    cmds = [np.array([1.5, 1.0])] * 100
#    cmds.extend([np.array([1.0, 2.0])])
#    cmds.extend([cmds[-1]]*50)
#    cmds.extend([np.array([1.0, 3.0])])
#    cmds.extend([cmds[-1]]*50)
#    cmds.extend([np.array([5.0, 1.0])])
#    cmds.extend([cmds[-1]]*30)
#    cmds.extend([np.array([5.0, 5.0])])
#    cmds.extend([cmds[-1]]*10)
    xs, traj= run_localization(cmds, x, x_guess, P, std_an, std_r, std_b, ellipse_step=20, step=10, show=show)
#    sensor_traces(x, cmds, 1.0, 10, std_an, std_r, std_b)
#   generate_ideal_noisy_traj(x, cmds, 1.0, 10, std_an)
    avg_x, avg_y, avg_theta, std_x, std_y, std_theta, avg_vx, avg_vy, avg_o, std_vx, std_vy, std_o = \
    accuracy(xs, traj, show=False)