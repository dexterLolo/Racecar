# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:23:49 2019

@author: sbtng
"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

xcam = 780
ycam = 100

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
#from scipy.stats import norm

# if using a Jupyter notebook, inlcude:

x_position = np.arange(0, 1240, 1)
y_position = np.arange(0, 270, 1)
anglereturn = np.arange(-1, 2, 1)

#position_farleft = fuzz.zmf(x_position, 0, 350)
position_left = fuzz.zmf(x_position, 0, 480)
position_center = fuzz.gaussmf(x_position, 640.0, 150)
position_rigth = fuzz.smf(x_position,780, 1240)
#position_farrigth = fuzz.smf(x_position,890, 1240)

distance_close = fuzz.zmf(y_position, 0, 100)
distance_center = fuzz.gaussmf(y_position, 130.0, 31.0)
distance_far = fuzz.smf(y_position,157.05, 270)

angle_positive = fuzz.smf(anglereturn, 0.2, 0.75)
angle_none = fuzz.gaussmf(anglereturn, 0, .3)
angle_negative = fuzz.zmf(anglereturn, -0.75, -0.2)
# Visualize these universes and membership functions
#fig, (ax0) = plt.subplots(nrows=1, figsize=(7, 9))

x_position.view

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))


ax0.plot(x_position, position_left, 'r', linewidth=1.5, label='Left')
ax0.plot(x_position, position_center, 'g', linewidth=1.5, label='Center')
ax0.plot(x_position, position_rigth, 'b', linewidth=1.5, label='Rigth')
#ax0.plot(x_position, position_farleft, 'r', linewidth=1.5, label=' Far Center')
#ax0.plot(x_position, position_farrigth, 'r', linewidth=1.5, label='Far Rigth')
ax0.set_title('Position in x')
ax0.legend()

ax1.plot(y_position, distance_close, 'b', linewidth=1.5, label='Close')
ax1.plot(y_position, distance_center, 'g', linewidth=1.5, label='Center')
ax1.plot(y_position, distance_far, 'r', linewidth=1.5, label='Far')
ax1.set_title('Position in y')
ax1.legend()

ax2.plot(anglereturn, angle_positive, 'b', linewidth=1.5, label='Positive/left')
ax2.plot(anglereturn, angle_none, 'g', linewidth=1.5, label='None/center')
ax2.plot(anglereturn, angle_negative, 'r', linewidth=1.5, label='Negative/rigth')
ax2.set_title('Angle movement')
ax2.legend()

for ax in (ax0,ax1,ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()