import time
import rospy
import cv2
import numpy as np
from math import atan

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import Switch4 

class street_ligth():
    #"Constructor" de la clase lineaRecta
    def __init__(self):
        #Suscribiendo
        rospy.Subscriber("ackermann_cmd_mux/output", AckermannDriveStamped,self.ackermanDevolucion)
        #rospy.Subscriber("/scan", LaserScan, self.laserDevolucion)
        rospy.Subscriber("/zed/rgb/image_rect_color",Image, self.camara_callback, queue_size = 1)
        #Publicando 
        self.cmd_pub = rospy.Publisher('ackermann_cmd_mux/input/default', AckermannDriveStamped, queue_size = 1)

        #Puente cv2-ROSº
        self.bridge = CvBridge()

        self.pts1 = np.float32([[0,550],[1280,550],[1280,720],[0,720]])
        self.pts2 = np.float32([[0,0], [1280,0],[1280,220],[0,220]])

        self.sec = 0.0
    def camara_callback(self,msg):

        self.msg=msg
        img = self.bridge.imgmsg_to_cv2(msg)
        #print img.shape[0]
        #print img.shape[1]
        change = cv2.getPerspectiveTransform(self.pts1,self.pts2)
        frame = cv2.warpPerspective(img,change,(1280,220))

########    Yellow Color Detection    ###############################################################################

        self.hsvY = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #defining the Range of yellow color
        #yellow_lower=np.array([26,100,100])
        #yellow_upper=np.array([35,255,255])

        #Purple
        yellow_lower=np.array([130,0,10])
        yellow_upper=np.array([185,255,255])


        #finding the range of yellow
        yellow=cv2.inRange(self.hsvY,yellow_lower,yellow_upper)

        self.blurY = cv2.GaussianBlur(yellow,(5,5),0)
        
        rowsY=self.blurY.shape[0]
        colsY=self.blurY.shape[1]


########    Red Color Detection    ###############################################################################

        self.hsvR = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #definig the range of red color
        red_lower=np.array([0,90,70])
        red_upper=np.array([11,255,155])

        #finding the range of red
        red=cv2.inRange(self.hsvR, red_lower, red_upper)

        self.blurR = cv2.GaussianBlur(red,(5,5),0)

        rowsR=self.blurR.shape[0]
        colsR=self.blurR.shape[1]

########    Green Color Detection    ###############################################################################

        self.hsvG = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #defining the Range of Green color
        green_lower=np.array([95,255,90])
        green_upper=np.array([136,255,90])
    
        #finding the range of green
        green=cv2.inRange(self.hsvB,green_lower,green_upper)

        self.blurG = cv2.GaussianBlur(green,(5,5),0)

        rowsG=self.blurG.shape[0]
        colsG=self.blurG.shape[1]

########    Contours comparison    ###############################################################################
        
        _,contoursY,hierachy=cv2.findContours(self.blurY.copy(), 1, cv2.CHAIN_APPROX_NONE)
        _,contoursR,hierachy=cv2.findContours(self.blurR.copy(), 1, cv2.CHAIN_APPROX_NONE)
        _,contoursG,hierachy=cv2.findContours(self.blurG.copy(), 1, cv2.CHAIN_APPROX_NONE)


        if len(contoursG) > 0:
            c = max(contoursG, key=cv2.contourArea)
            M = cv2.moments(c)

            self.Switch4()######################################

            print "Green"
        elif len(contoursY) > 0:
            c = max(contoursY, key=cv2.contourArea)
            M = cv2.moments(c)
            
            print "Yellow"
        elif len(contoursR) > 0:
            c = max(contoursR, key=cv2.contourArea)
            M = cv2.moments(c)
            
            print "Red"

        self.sec = self.sec + 1.0
        #print 'seconds'
        #print self.sec/12.0

        cv2.imshow('thresholdY',self.blurY)
        cv2.waitKey(1)
        #cv2.imshow('thresholdR',self.blurR)
        #cv2.waitKey(1)
        #cv2.imshow('thresholdB',self.blurB)
        #cv2.waitKey(1)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(1)
if __name__ == "__main__":
    #Inicializamos el nodo linearecta
    rospy.init_node("street_ligth")
    #Instanciamos la clase lineaRecta
    node = street_ligth()


#Se mantiene escuchando a los topicos eternament
