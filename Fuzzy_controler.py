
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:12:49 2019

@author: sbtng
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

xcam = 708
ycam = 100

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
#from scipy.stats import norm

# if using a Jupyter notebook, inlcude:

x_position = np.arange(0, 1280, 1)
y_position = np.arange(0, 270, 1)
anglereturn = np.arange(-1, 2, 1)

#position_farleft = fuzz.zmf(x_position, 0, 350)
position_left = fuzz.zmf(x_position, 0, 640)
position_center = fuzz.gaussmf(x_position, 640.0, 170.0)
position_rigth = fuzz.smf(x_position,640, 1240)
#position_farrigth = fuzz.smf(x_position,890, 1240)

distance_close = fuzz.zmf(y_position, 0, 100)
distance_center = fuzz.gaussmf(y_position, 130.0, 31.0)
distance_far = fuzz.smf(y_position,157.05, 270)

angle_positive = fuzz.smf(anglereturn, 0.1, 0.5)
angle_none = fuzz.gaussmf(anglereturn, 0, .1)
angle_negative  = fuzz.zmf(anglereturn, -0.5, -0.1)
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
ax2.plot(anglereturn, angle_none, 'g', linewidth=1.5, label='None')
ax2.plot(anglereturn, angle_negative, 'r', linewidth=1.5, label='Negative/rigth')
ax2.set_title('Angle movement')
ax2.legend()

for ax in (ax0,ax1,ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
position_level_left = fuzz.interp_membership(x_position, position_left, xcam)
position_level_center = fuzz.interp_membership(x_position, position_center, xcam)
position_level_rigth = fuzz.interp_membership(x_position, position_rigth, xcam)

'''
serv_level_lo = fuzz.interp_membership(x_serv, serv_lo, 9.8)
serv_level_md = fuzz.interp_membership(x_serv, serv_md, 9.8)
serv_level_hi = fuzz.interp_membership(x_serv, serv_hi, 9.8)
'''
# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
active_rule1 = position_level_left #np.fmax(qual_level_lo, serv_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
tip_activation_lo = np.fmin(active_rule1, angle_positive)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
tip_activation_md = np.fmin(position_level_center, angle_none)

# For rule 3 we connect high service OR high food with high tipping
active_rule3 = position_level_rigth#np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, angle_negative)
tip0 = np.zeros_like(anglereturn)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(anglereturn, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(anglereturn, angle_positive, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(anglereturn, tip0, tip_activation_md, facecolor='g', alpha=0.7)
ax0.plot(anglereturn, angle_none, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(anglereturn, tip0, tip_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(anglereturn, angle_negative, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# Aggregate all three output membership functions together
aggregated = np.fmax(tip_activation_lo,np.fmax(tip_activation_md, tip_activation_hi))

# Calculate defuzzified result
tip = fuzz.defuzz(anglereturn, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(anglereturn, aggregated, tip)  # for plot
print ("tipactivation") 
print (tip_activation)
print ("tip") 
print (tip)
# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(anglereturn, angle_positive, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(anglereturn, angle_none, 'g', linewidth=0.5, linestyle='--')
ax0.plot(anglereturn, angle_negative, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(anglereturn, tip0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

