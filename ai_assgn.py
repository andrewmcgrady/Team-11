# -*- coding: utf-8 -*-

'''
This code is based on particle filter research includning, particularly, these two articles:
    * Fox, D., Burgard, W., Dellaert, F., & Thrun, S. (1999). Monte carlo localization: Efficient position 
      estimation for mobile robots. AAAI/IAAI, 1999(343-349), 2-2.
    * F. Dellaert, D. Fox, W. Burgard and S. Thrun, "Monte Carlo localization for mobile robots," 
      Proceedings 1999 IEEE International Conference on Robotics and Automation (Cat. No.99CH36288C), 
      1999, pp. 1322-1328 vol.2, doi: 10.1109/ROBOT.1999.772544.
    
Robots and particles derived from the robot class have these features:
    * Constant noise in movement distance and heading expressed as a percentage 
      of the nominal movement distance and direction, respectively
    * Constant noise in sensing distance to the navigation features as a constant 
      percentage of the domain dimension
    * A copy method to permit avoidance of alaising
    * In-place movement: robot x and y coordinates are updated in place rather than needing to 
      instantiate a new robot
    
Distance to navigation features is computed as the Euclidean distance.

Resampling weights for particles are computed via rank of sum of squared differences in distances to navigation
  features robot versus particles

'''

from math import *
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nav_features  = [[10.0, 10.0], [80.0, 90.0], [20.0, 70.0], [90.0, 20.0]]
domain_size = 100.0

class robot:
    ''' Initialize robot with random location/orientation and movement & sensing noise '''
    def __init__(self, forward_noise, turn_noise, sense_noise):
        self.x = random.random() * domain_size
        self.y = random.random() * domain_size
        self.heading = random.random() * 2.0 * pi
        self.forward_noise = float(forward_noise);
        self.turn_noise    = float(turn_noise);
        self.sense_noise   = float(sense_noise);
    
    ''' Measure distance to navigation features, with noise '''
    def sense(self):
        Z = []
        for i in range(len(nav_features)):
            dist = sqrt((self.x - nav_features[i][0]) ** 2 + (self.y - nav_features[i][1]) ** 2)
            dist += (random.random() - 1) * domain_size * self.sense_noise 
            Z.append(dist)
        return Z
    
    ''' Execute move and change location and orientation in-place '''
    def move(self, forward, delta_heading):
        # Update heading with noise
        self.heading = ((self.heading + float(delta_heading) * (1 + random.random() - 1) * self.turn_noise)) % 2 * pi
        
        # move, and add randomness to the motion command
        dist = float(forward) * (1 + (random.random() - 1) * self.forward_noise) 
        self.x = (self.x + (cos(self.heading) * dist)) % domain_size
        self.y = (self.y + (sin(self.heading) * dist)) % domain_size
    
    ''' Compute sum of squared differences between robot and particle distances to navigation features '''
    def dist_measure(self, robot_meas):
        dist = 0.0;
        me_Z = self.sense()
        for i in range(len(nav_features)):
            dist += (me_Z[i] - robot_meas[i]) ** 2
        return dist
    
    def copy(self):
        myrobot = robot(self.forward_noise, self.turn_noise, self.sense_noise)
        myrobot.x = self.x
        myrobot.y = self.y
        myrobot.heading = self.heading
        return myrobot
        
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.heading))

def compute_prob(particle_weights):
    ranks = np.empty((len(particle_weights),))
    ranks[np.argsort(-np.array(particle_weights))] = np.arange(len(particle_weights))
    ranks += 1
    w = len(particle_weights) * (len(particle_weights) + 1) / 2 
    return ranks / w
        
    
def plot_domain(t, robot, particles):
    fig,ax = plt.subplots()
    ax.scatter([x.x for x in particles], [x.y for x in particles], c='k', alpha=0.15, zorder=0) #
    #ax.plot([myrobot.x], [myrobot.y], marker='o', color='r', markersize=6, zorder=1)
    ax.scatter([robot.x], [robot.y], c='r', zorder=1)
    ax.set_xlim(0,domain_size)
    ax.set_ylim(0,domain_size)
    fig.suptitle('Iteration ' + str(t))
    fig.set_size_inches(6,6)
    plt.show()

''' Parameters '''
num_particles = 2000        # Number of particles in filter
num_iterations = 50          # Number of robot moves to make

move_dist = 5.0  # magnitude of random movement 
move_heading = 0.5 # magnitude of random turn direction
noise_args = (0.05, 0.05, 0.00005)  # noise arguments for movement magnitude, movement direction, and sensing distance to navigation features

''' Create robot to be localized '''
'''   The arguments define, in sequence, the noise in movement magnitude, movement direction, and
      noise is sensing.  The particles used to estimate the robot's position should have the same 
      noise parameters '''
myrobot = robot(*noise_args)

''' Create collection of N particles '''
p = [robot(*noise_args) for i in range(num_particles)]

''' Plot initial position of robot and particles '''
plot_domain(0, myrobot, p)

''' Repeat sense and move T times '''
for t in range(1, num_iterations + 1):
    
    ''' Set move arguments for each iteration for both the robot and particles: (move_magnitude, move_direction) '''
    '''   - We create this tuple so that it can be used to move the robot and each of the particles in the same
            (intended) direction by the same intended amount '''
    move_args = (move_dist*random.random(), move_heading*random.random())
    
    ''' Move robot and sense new position: remember there is noise in movement and position sensing '''
    ''' Note that the asterisk operator is used to unpack the tuple arguments because the move() function 
        needs them individually '''
    myrobot.move(*move_args)
    myrobot_features_dist = myrobot.sense()

    ''' Move all particles by same magnitude and direction as robot moved '''
    ''' Use the variable p for the collection of particles.  It should be a list of "robot" objects. 
        Note that the variable p was originally used for this collection of particles when they were initially 
        defined. '''
        
    p_move_args = move_args
    for j in range(len(p)) :
        p[j].move(*p_move_args)
        
    ''' Compute squared sum of distance differences between robot distances to navigation features and particle distances to navigation features 
        for each particle 
         - Use the variable d to represent a list of distances for each particle 
         - Recall that the variable myrobot_features_dist contains the distances of the robot to the 4 navigation features 
         - The dist_measure() function is useful here '''
    d=[]
    for i in range(len(p)):
        p_dist = p[i].dist_measure(myrobot_features_dist)
        d.append(p_dist)
        
    ''' Then, compute the probability of selection using ranks:  
          - The compute_prob() function does this for you '''
    a = []
    a = compute_prob(d)

    ''' Resample new population of particles from current population 
          - When finished, the new population should be a list of "robot" objects in the variable p 
          - You may use the numpy package here, in particular, the np.cumsum() function if you find it convenient.
          - Be careful of aliasing '''
    cum_prob_p = np.cumsum(a)
    #figure out how to iterate through the index of an array
    binary_list=[]
    
    rand = []
    for i in range(len(cum_prob_p)):
        cum_prob_p[i] = np.random.random()
        rand.append(cum_prob_p[i])

    for i in range(len(cum_prob_p)) and range(len(rand)):
        if cum_prob_p[i-1] < rand[i]:
            if rand[i] <= cum_prob_p[i]:
                binary_list.append(1)
            else:
                binary_list.append(0)
        else:
            binary_list.append(0)
            
    for j in range(len(binary_list)) or range(len(p)):
        if p[j] != 1:
            if binary_list[j]!= 1:
                del p[j]
                p += [robot(*noise_args)for i in range(1)]
            else:
                pass
        else:
            pass
    
      
    ''' Print iteration number and plot robot and particles '''
    print('\nIteration %d' % (t,))
    plot_domain(t, myrobot, p)