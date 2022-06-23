#! /usr/bin/env python


#coseno 3 el mero mero
#Se importan librerias que se usaran

import time
import rospy
import numpy as np
#Se importan topics
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
 
#Creamos la clase lineaRecta

class wallFollower():
    #"Constructor" de la clase lineaRecta
    def __init__(self):
        #
        rospy.Subscriber('/drive',AckermannDriveStamped,self.ackermanDevolucion)
        #
        self.cmd_pub = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/default', AckermannDriveStamped,queue_size = 1)
        #'ackermann_cmd_mux/input/default'
        #Se crea suscriptor de LaserScan
        scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback) 
        #rospy.subscriber(/scan(sensor_msgs/LaserScan))
        self.idealDis = 0.6
        self.kp = 1.0
        # self.ki = 0.0
        self.kd = 0.4
        self.mode = "Left" 

        self.velCoeff = 0.0
        self.angleVel = 0.0
        self.steerAngle = 0.0

        self.prev_error = 0
        self.prev_time = 0

    #Metodo para sacar promedio de las lista de distancias
    def avr(self,arr,a,b):
        aux = b - a + 1
        total = 0
        for i in range(a,b+1):
            total += arr[i]
        return total/aux

    #Metodo de deteccion de obstaculos
    def obs(self,arr):
        aux = 0
        for i in range(9,11):
            if arr[i] <= 0.15:
                # print "Hola ", i
                return 0
            elif arr[i] <= 0.5:
                # print "Hola ", i
                return 1
            elif arr[i] <= 0.6:
                # print i
                aux = 2
            else:
                aux = 3
        return aux

    #Metodo del controlador
    def PID(self):
        #Condicionales para determinar la direccion del angulo de movimiento
        if self.mode == "Left":
            dir = 1
            self.error = self.averageL - self.idealDis
            self.diagonal2 = self.hipo2L
            # self.diagonal1 = self.hipo1L
        #Condicional para cambiar signo del angulo a negativo
        elif self.mode == "Right":
            dir = -1
            self.error = self.averageR  - self.idealDis
            self.diagonal2 = self.hipo2R
            # self.diagonal1 = self.hipo1R

        self.errorF = self.diagonal2 * np.cos(np.deg2rad(35)) - self.idealDis
        # self.errorP = self.diagonal1 * np.cos(np.deg2rad(-30)) -  self.idealDis
        #Se obtiene la hora actua;
        #self.current_time = time.time()
        #Proportional
        self.pControl = self.kp * self.error
        #Integral
        #self.iControl = self.ki * self.errorP
        #Derivative
        self.dControl = self.kd * self.errorF
        #Los valores actuales de errores y tiempo se vielven previos
        # self.prev_time = self.current_time
        # self.prev_error = self.error
 
        if self.obstacle == 0:
            self.velCoeff = 0.0
            self.angleVel = 0.0
            self.steerAngle = (self.pControl + self.dControl) * dir
        elif self.obstacle == 1:
            self.velCoeff = -0.6
            self.angleVel = 0.2
            self.steerAngle = (self.pControl + self.dControl) *-1 * dir
        elif self.obstacle == 2:
            self.velCoeff = 4.0
            self.angleVel = 0.2
            self.steerAngle = dir * -0.8
        else:
            self.velCoeff = 4.0
            self.angleVel = 0.2
            self.steerAngle = (self.pControl + self.dControl) * dir

        #Imprime error
        # print "Error: ", self.error
        # print "Steer Angle: ", self.steerAngle

        self.ackermanDevolucion(AckermannDriveStamped())
        #Se regresan los valores
        return self.velCoeff
        return self.angleVel
        return self.steerAngle

    def scan_callback(self,msg):

        self.my_list = list()
        #Loop para introducir promedios de 100 datos a la lista
        #Lista tiene longitud 10
        for i in range(20):
            self.my_list.append(np.mean(msg.ranges[(40+(50*i)):(40-1+(50*(i+1)))]))
        #Distancia promedio de izquierda a derecha
        self.averageR = self.avr(self.my_list,1,4)
        self.averageL = self.avr(self.my_list,15,18)
        #Metodo del obstaculo
        self.obstacle = self.obs(self.my_list)
 
        self.hipo2R = msg.ranges[320]
        # self.hipo1R = msg.ranges[60]
 
        self.hipo2L = msg.ranges[760]
        # self.hipo1L = msg.ranges[1020]      

        # print "Distance : ", msg.ranges[140]
        # print "averageR : ", self.averageR
        # print "averageL : ", self.averageL
        # print "obstacle : ", self.obstacle
        #Se crea division
        # print "---------"
        #Se borra la lista

        del self.my_list[:]

        self.PID()

    def ackermanDevolucion(self,msg):
        #Velocidad del carrito
        msg.drive.speed = self.velCoeff
        #print self.velCoeff
        # Angulo de Direccion
        msg.drive.steering_angle = self.steerAngle
        #Velocidad del angulo de Direccion
        msg.drive.steering_angle_velocity = self.angleVel
        #Se publican las velocidades
        self.cmd_pub.publish(msg)

#Funcion principal del interprete "main"
if __name__ == "__main__":
    #Inicializamos el nodo linearecta
    rospy.init_node("wallFollower")
    #Instanciamos la clase lineaRecta
    node = wallFollower()
rospy.spin()

#Se mantiene escuchando a los topicos eternament
