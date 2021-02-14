import numpy as np

cx = 700

def zmf(x, a, b):
    assert a <= b, 'a <= b is required.'
    y = np.ones(len(x))

    idx = np.logical_and(a <= x, x < (a + b) / 2.)
    y[idx] = 1 - 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 2. * ((x[idx] - b) / (b - a)) ** 2.

    idx = x >= b
    y[idx] = 0

    return y

def smf(x, a, b):
    assert a <= b, 'a <= b is required.'
    y = np.ones(len(x))
    idx = x <= a
    y[idx] = 0

    idx = np.logical_and(a <= x, x <= (a + b) / 2.)
    y[idx] = 2. * ((x[idx] - a) / (b - a)) ** 2.

    idx = np.logical_and((a + b) / 2. <= x, x <= b)
    y[idx] = 1 - 2. * ((x[idx] - b) / (b - a)) ** 2.

    return y

def gaussmf(x, mean, sigma):
    #Hola
    return np.exp(-((x - mean) ** 2.) / float(sigma) ** 2.)
    #Adios

def interp_membership(x, xmf, xx):
    x1 = x[x <= xx][-1]
    x2 = x[x >= xx][0]

    idx1 = np.nonzero(x == x1)[0][0]
    idx2 = np.nonzero(x == x2)[0][0]

    xmf1 = xmf[idx1]
    xmf2 = xmf[idx2]



    if x1 == x2:
        xxmf = xmf[idx1]
    else:
        slope = (xmf2 - xmf1) / float(x2 - x1)
        xxmf = slope * (xx - x1) + xmf1

    return xxmf

def centroid(x, mfx):
    sum_moment_area = 0.0
    sum_area = 0.0

    # If the membership function is a singleton fuzzy set:
    if len(x) == 1:
        return x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float)

    # else return the sum of moment*area/sum of area
    for i in range(1, len(x)):
        x1 = x[i - 1]
        x2 = x[i]
        y1 = mfx[i - 1]
        y2 = mfx[i]

        # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
        if not(y1 == y2 == 0.0 or x1 == x2):
            if y1 == y2:  # rectangle
                moment = 0.5 * (x1 + x2)
                area = (x2 - x1) * y1
            elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                moment = 2.0 / 3.0 * (x2-x1) + x1
                area = 0.5 * (x2 - x1) * y2
            elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                moment = 1.0 / 3.0 * (x2 - x1) + x1
                area = 0.5 * (x2 - x1) * y1
            else:
                moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                area = 0.5 * (x2 - x1) * (y1 + y2)

            sum_moment_area += moment * area
            sum_area += area

    return sum_moment_area / np.fmax(sum_area,
                                     np.finfo(float).eps).astype(float)

def bisector(x, mfx):
    sum_area = 0.0
    accum_area = [0.0] * (len(x) - 1)

    # If the membership function is a singleton fuzzy set:
    if len(x) == 1:
        return x[0]

    # else return the sum of moment*area/sum of area
    for i in range(1, len(x)):
        x1 = x[i - 1]
        x2 = x[i]
        y1 = mfx[i - 1]
        y2 = mfx[i]

        # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
        if not(y1 == y2 == 0. or x1 == x2):
            if y1 == y2:  # rectangle
                area = (x2 - x1) * y1
            elif y1 == 0. and y2 != 0.:  # triangle, height y2
                area = 0.5 * (x2 - x1) * y2
            elif y2 == 0. and y1 != 0.:  # triangle, height y1
                area = 0.5 * (x2 - x1) * y1
            else:
                area = 0.5 * (x2 - x1) * (y1 + y2)
            sum_area += area
            accum_area[i - 1] = sum_area

    # index to the figure which cointains the x point that divide the area of
    # the whole fuzzy set in two
    index = np.nonzero(np.array(accum_area) >= sum_area / 2.)[0][0]

    # subarea will be the area in the left part of the bisection for this set
    if index == 0:
        subarea = 0
    else:
        subarea = accum_area[index - 1]
    x1 = x[index]
    x2 = x[index + 1]
    y1 = mfx[index]
    y2 = mfx[index + 1]

    # We are interested only in the subarea inside the figure in which the
    # bisection is present.
    subarea = sum_area/2. - subarea

    x2minusx1 = x2 - x1
    if y1 == y2:  # rectangle
        u = subarea/y1 + x1
    elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
        root = np.sqrt(2. * subarea * x2minusx1 / y2)
        u = (x1 + root)
    elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
        root = np.sqrt(x2minusx1*x2minusx1 - (2.*subarea*x2minusx1/y1))
        u = (x2 - root)
    else:
        m = (y2-y1) / x2minusx1
        root = np.sqrt(y1*y1 + 2.0*m*subarea)
        u = (x1 - (y1-root) / m)
    return u

def defuzz(x, mfx, mode):
    mode = mode.lower()
    x = x.ravel()
    mfx = mfx.ravel()
    n = len(x)
    assert n == len(mfx), 'Length of x and fuzzy membership function must be \
                      identical.'

    if 'centroid' in mode or 'bisector' in mode:
        zero_truth_degree = mfx.sum() == 0  # Approximation of total area
        assert not zero_truth_degree, 'Total area is zero in defuzzification!'

        if 'centroid' in mode:
            return centroid(x, mfx)

        elif 'bisector' in mode:
            return bisector(x, mfx)

    elif 'mom' in mode:
        return np.mean(x[mfx == mfx.max()])

    elif 'som' in mode:
        tmp = x[mfx == mfx.max()]
        return tmp[tmp == np.abs(tmp).min()][0]

    elif 'lom' in mode:
        tmp = x[mfx == mfx.max()]
        return tmp[tmp == np.abs(tmp).max()][0]

    else:
        raise ValueError('The input for `mode`, %s, was incorrect.' % (mode))

#Funcion del controlador del error
x_position = np.arange(0, 1240,1)
#self.y_position = np.arange(0, 1240,1)
anglereturn = np.arange(-1, 2, 1)

xcam = cx
#ycam = 100

#position_farleft = fuzz.zmf(x_position, 0, 350)
position_left = zmf(x_position, 0, 640)
print (position_left)
position_center = gaussmf(x_position, 640.0, 170)
position_rigth = smf(x_position, 640, 1240)
#position_farrigth = fuzz.smf(x_position,890, 1240)

'''distance_close = zmf(y_position, 0, 100)
distance_center = gaussmf(y_position, 130.0, 31.0)
distance_far = smf(y_position,157.05, 270)'''

angle_positive = smf(anglereturn, 0.1, 0.75)
angle_none = gaussmf(anglereturn, 0, .1)
angle_negative  = zmf(anglereturn, -0.75, -0.1)
# Visualize these universes and membership functions
#fig, (ax0) = plt.subplots(nrows=1, figsize=(7, 9))

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
position_level_left = interp_membership(x_position, position_left, xcam)
position_level_center = interp_membership(x_position, position_center, xcam)
position_level_rigth = interp_membership(x_position, position_rigth, xcam)

# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
active_rule1 = position_level_left #np.fmax(qual_level_lo, serv_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
ang_activation_lo = np.fmin(active_rule1, angle_positive)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
ang_activation_md = np.fmin(position_level_center, angle_none)

# For rule 3 we connect high service OR high food with high tipping
active_rule3 = position_level_rigth#np.fmax(qual_level_hi, serv_level_hi)
ang_activation_hi = np.fmin(active_rule3, angle_negative)
ang0 = np.zeros_like(anglereturn)

# Visualize this

# Aggregate all three output membership functions together
aggregated = np.fmax(ang_activation_lo,np.fmax(ang_activation_md, ang_activation_hi))

# Calculate defuzzified result
angle = defuzz(anglereturn, aggregated, 'centroid')
ang_activation = interp_membership(anglereturn, aggregated, angle)  # for plot
print (cx)
print (angle)