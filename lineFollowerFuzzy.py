import time
import rospy
import cv2
import numpy as np
import skfuzzy as fuzz
from math import atan

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge

#Creamos la clase seguirColor
class lineFollower1():
    #"Constructor" de la clase lineaRecta
    def __init__(self):
        #Suscribiendo
        rospy.Subscriber("ackermann_cmd_mux/output", AckermannDriveStamped,self.ackermanDevolucion)
        #rospy.Subscriber("/scan", LaserScan, self.laserDevolucion)
        rospy.Subscriber("/zed/rgb/image_rect_color",Image, self.camara_callback, queue_size = 1)
        #Publicando 
        self.cmd_pub = rospy.Publisher('ackermann_cmd_mux/input/default', AckermannDriveStamped, queue_size = 1)

        #Puente cv2-ROS
        self.bridge = CvBridge()
        self.velCoeff = 0.0
        self.angleVel = 0.0
        self.steerAngle = 0.0
        
        self.error = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.cxF = 0.0

    #Funcion de devolucion de llamada
    def camara_callback(self,msg):
        self.msg=msg
        frame = self.bridge.imgmsg_to_cv2(msg)
        frameF = self.bridge.imgmsg_to_cv2(msg)

        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.hsvF = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de colores
        lower_yellow = np.array([0,150,130])#([105,100,50])
        upper_yellow = np.array([15,240,255])#([105,100,50])

        self.mask = cv2.inRange(self.hsv,lower_yellow,upper_yellow)
        self.blur = cv2.GaussianBlur(self.mask,(5,5),0)
        self.ret3, self.threshold = cv2.threshold(self.blur,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        self.maskF = cv2.inRange(self.hsvF,lower_yellow,upper_yellow)
        self.blurF = cv2.GaussianBlur(self.maskF,(5,5),0)
        self.ret3, self.thresholdF = cv2.threshold(self.blur,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        rows=self.threshold.shape[0]
        cols=self.threshold.shape[1]

        rowsF=self.thresholdF.shape[0]
        colsF=self.thresholdF.shape[1]

        area1=np.array([[[0,0], [0,500], [1280,500],[1280,0]]],dtype=np.int32)
        cv2.fillPoly(self.threshold,area1,0)

        area1F=np.array([[[0,0], [0,500], [300,500], [550,400], [730,400], [980,500], [1280,500], [1280,0]]],dtype=np.int32)
        area2F=np.array([[[0,500], [0,720], [1280,720],[1280,500]]],dtype=np.int32)
        cv2.fillPoly(self.thresholdF,area1F,0)
        cv2.fillPoly(self.thresholdF,area2F,0)

        contours,hierachy=cv2.findContours(self.threshold.copy(), 1, cv2.CHAIN_APPROX_NONE)
        contoursF,hierachy=cv2.findContours(self.thresholdF.copy(), 1, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            self.cx = int(M['m10']/M['m00'])
            print (self.cx)
            self.cy = int(M['m01']/M['m00'])
            print (self.cy)
            cF = max(contoursF, key=cv2.contourArea)
            MF = cv2.moments(cF)
            self.cxF = int(MF['m10']/MF['m00'])
            #self.cyF = int(MF['m01']/M['m00'])

        else:
            
            self.cx = 640
            #self.cxF = 640
            2
        cv2.imshow('threshold',self.threshold)
        cv2.imshow('thresholdF',self.thresholdF)
        cv2.waitKey(1)

        self.fuzzy()
    
    #Funcion del controlador del error
    def fuzzy(self):
        # Generate universe variables
        #   * Quality and service on subjective ranges [0, 10]
        #   * Tip has a range of [0, 25] in units of percentage points
        #from scipy.stats import norm

        # if using a Jupyter notebook, inlcude:

        x_position = np.arange(0, 1280, 1)
        y_position = np.arange(0, 270, 1)
        anglereturn = np.arange(-1, 2, 1)

        #position_farleft = fuzz.zmf(x_position, 0, 350)
        position_left = fuzz.zmf(x_position, 0, 708)
        position_center = fuzz.gaussmf(x_position, 708.0, 150)
        position_rigth = fuzz.smf(x_position,708, 1240)
        #position_farrigth = fuzz.smf(x_position,890, 1240)

        distance_close = fuzz.zmf(y_position, 0, 100)
        distance_center = fuzz.gaussmf(y_position, 130.0, 31.0)
        distance_far = fuzz.smf(y_position,157.05, 270)

        angle_positive = fuzz.smf(anglereturn, 0.1, 0.75)
        angle_none = fuzz.gaussmf(anglereturn, 0, .1)
        angle_negative  = fuzz.zmf(anglereturn, -0.75, -0.1)
        # Visualize these universes and membership functions
        #fig, (ax0) = plt.subplots(nrows=1, figsize=(7, 9))

        # We need the activation of our fuzzy membership functions at these values.
        # The exact values 6.5 and 9.8 do not exist on our universes...
        # This is what fuzz.interp_membership exists for!
        position_level_left = fuzz.interp_membership(x_position, position_left, self.cx)
        position_level_center = fuzz.interp_membership(x_position, position_center, self.cx)
        position_level_rigth = fuzz.interp_membership(x_position, position_rigth, self.cx)

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
        angle = fuzz.defuzz(anglereturn, aggregated, 'centroid')
        ang_activation = fuzz.interp_membership(anglereturn, aggregated, angle)  # for plot
        #print ("tipactivation") 
        #print (tip_activation)
        print ("angle") 
        print (angle)
        # Visualize this
             
    def ackermanDevolucion(self,msg):
        #Velocidad del carrito
        msg.drive.speed = self.velCoeff
        # Angulo de Direccion
        msg.drive.steering_angle = self.steerAngle
        #Velocidad del angulo de Direccion
        msg.drive.steering_angle_velocity = self.angleVel
        #Se publican las velocidades
        self.cmd_pub.publish(msg)

#Funcion principal del interprete "main"
if __name__ == "__main__":
    #Inicializamos el nodo linearecta
    rospy.init_node("lineFollower1")
    #Instanciamos la clase lineaRecta
    node = lineFollower1()

rospy.spin()
#Se mantiene escuchando a los topicos eternament
