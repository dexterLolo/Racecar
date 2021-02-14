import cv2
import rospy
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge

class cone_follower():
	def __init__(self):
        #Suscribiendo
        rospy.Subscriber("ackermann_cmd_mux/output", AckermannDriveStamped,self.ackermanDevolucion)
        #rospy.Subscriber("/scan", LaserScan, self.laserDevolucion)
        rospy.Subscriber("/zed/rgb/image_rect_color",Image, self.camara_callback, queue_size = 1)
        #Publicando 
        self.cmd_pub = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/default', AckermannDriveStamped, queue_size = 1) 

        #Puente cv2-ROSº
        self.bridge = CvBridge()
        self.velCoeff = 0.0
        self.angleVel = 0.0
        self.steerAngle = 0.0
        
        self.cx = 0.0
        self.cy = 0.0

        self.x_position = np.arange(0.0, 1240.0,1.0)
        #self.y_position = np.arange(0, 1240,1)
        self.anglereturn = np.arange(-1, 2, 1)

        self.xcam = 900
        self.pts1 = np.float32([[0,550],[1280,550],[1280,720],[0,720]])
        

        self.sec = 0.0

    def camara_callback(self,msg):

        self.msg=msg
        img = self.bridge.imgmsg_to_cv2(msg)
        #print img.shape[0]
        #print img.shape[1]
        change = cv2.getPerspectiveTransform(self.pts1)
        frame = cv2.warpPerspective(img,change,(1280,220))

########    Cone Color Detection    ###############################################################################

        self.hsvY = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #defining the Range of yellow color
        #yellow_lower=np.array([26,100,100])
        #yellow_upper=np.array([35,255,255])

        #Orange
        orange_lower=np.array([05,163,188])
        orange_upper=np.array([12,255,255])


        #finding the range of orange
        orange = cv2.inRange(self.hsvY,orange_lower,orange_upper)

        self.blurO = cv2.GaussianBlur(orange,(5,5),0)
        
        rowsO=self.blurO.shape[0]
        colsO=self.blurO.shape[1
]

        _,contoursO,hierachy=cv2.findContours(self.blurO.copy(), 1, cv2.CHAIN_APPROX_NONE)

        if len(contoursO) >= 600:
            c = max(contoursY, key=cv2.contourArea)
            M = cv2.moments(c)
            self.cx = int(M['m10']/M['m00'])
            print "Cone"

        else
        	self.velCoeff = 1.5
            self.angleVel = 0.0
            self.steerAngle = 0.0
            

        self.sec = self.sec + 1.0
        #print 'seconds'
        #print self.sec/12.0

        cv2.imshow('thresholdYO',self.blurO)
        cv2.waitKey(1)
        #cv2.imshow('thresholdR',self.blurR)
        #cv2.waitKey(1)
        #cv2.imshow('thresholdB',self.blurB)
        #cv2.waitKey(1)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(1)

        self.Fuzzy()

    def Fuzzy(self):
        self.xcam = self.cx
        #if self.cx == 0.0:
           #self.xcam = -1 * self.xcam
        #else:
           #self.xcam = self.cx
        #print self.cx
        #print self.xcam
        #ycam = 100
        #position_farleft = fuzz.zmf(x_position, 0, 350)
        position_left = fuzz.zmf(self.x_position, 0.0, 740.0)
        position_center = fuzz.gaussmf(self.x_position, 640.0, 90)
        position_rigth = fuzz.smf(self.x_position, 540, 1240)
        #position_farrigth = fuzz.smf(x_position,890, 1240)

        '''distance_close = zmf(y_position, 0, 100)
        distance_center = gaussmf(y_position, 130.0, 31.0)
        distance_far = smf(y_position,157.05, 270)'''
        
        angle_positive = fuzz.smf(self.anglereturn, 0.1, 0.75)
        angle_none = fuzz.gaussmf(self.anglereturn, 0, .1)
        angle_negative  = fuzz.zmf(self.anglereturn, -0.75, -0.1)
        # Visualize these universes and membership functions
        #fig, (ax0) = plt.subplots(nrows=1, figsize=(7, 9))

        # We need the activation of our fuzzy membership functions at these values.
        # The exact values 6.5 and 9.8 do not exist on our universes...
        # This is what fuzz.interp_membership exists for!
        position_level_left = interp_membership(self.x_position, position_left, self.xcam)
        position_level_center = interp_membership(self.x_position, position_center, self.xcam)
        position_level_rigth = interp_membership(self.x_position, position_rigth, self.xcam)

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
        ang0 = np.zeros_like(self.anglereturn)

        # Visualize this

        # Aggregate all three output membership functions together
        aggregated = np.fmax(ang_activation_lo,np.fmax(ang_activation_md, ang_activation_hi))

        # Calculate defuzzified result
        angle = defuzz(self.anglereturn, aggregated, 'centroid')
        ang_activation = interp_membership(self.anglereturn, aggregated, angle)  # for plot
        

        self.steerAngle = angle * -1
     
        if self.steerAngle < 0.10 and self.steerAngle > -0.10:
            self.velCoeff = 1.5
            self.angleVel = 0.2
            self.steerAngle = 0.0
        else:
            self.velCoeff = 0.7
            self.angleVel = 0.2

        #Imprime error
        # print "Error: ", self.error
        print "Steer Angle: ", self.steerAngle
        self.ackermanDevolucion(AckermannDriveStamped())
        #Se regresan los valores
        return self.velCoeff
        return self.angleVel
        return self.steerAngle

    def ackermanDevolucion(self,msg):
        #Velocidad del carrito
        msg.drive.speed = 0.70 - abs(np.rad2deg(self.steerAngle)/100)
        # Angulo de Direccion
        msg.drive.steering_angle = self.steerAngle
        #Velocidad del angulo de Direccion
        msg.drive.steering_angle_velocity = self.angleVel
        #Se publican las velocidades
        self.cmd_pub.publish(msg)

#Funcion principal del interprete "main"
if __name__ == "__main__":
    #Inicializamos el nodo linearecta
    rospy.init_node("cone_follower")
    #Instanciamos la clase lineaRecta
    node = cone_follower()

rospy.spin()
#Se mantiene escuchando a los topicos eternamen

        