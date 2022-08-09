#!Anaconda/anaconda/python
#coding: utf-8



import dlib                     
import numpy as np              
import cv2                      

import time

#------alram set
import time
import RPi.GPIO as GPIO
  
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
pin_alram=20 # pin 20 to connect alram
  
GPIO.setup(pin_alram, GPIO.OUT) 
GPIO.output(pin_alram, GPIO.LOW)  #no alram
#-----alram set

line_cuxi=1

threshold_set=3# 


class face_emotion():

    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)
        
        self.cnt = 0


    def learning_face(self):
        
        # eye 
        state_cur_eye=0 
        state_last_eye=0

        begin_eye=0
        timeTotal_eye=0
        timeTotalbigin_eye=0
        timeTotalend_eye=0

        timeClose_eye=0
        time_start_eye=0
        time_end_eye=0

        coutClose_eye=0
        
        
        #mouth
        state_cur_mouth=0
        state_last_mouth=0
        
        begin_mouth=0
        timeTotal_mouth=0
        timeTotalbigin_mouth=0
        timeTotalend_mouth=0
        
        timeClose_mouth=0
        time_start_mouth=0
        time_end_mouth=0
        
        coutClose_mouth=0
        

        line_brow_x = []
        line_brow_y = []

        while(self.cap.isOpened()):

           
            flag, im_rd = self.cap.read()

            k = cv2.waitKey(1)

            
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            faces = self.detector(img_gray, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX

            
            if(len(faces)!=0):

                for i in range(len(faces)):
                    
                    for k, d in enumerate(faces):
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                        
                        self.face_width = d.right() - d.left()

                        shape = self.predictor(im_rd, d)
                        
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                       
                        mouth_w = abs(shape.part(54).x + shape.part(64).x- shape.part(48).x-shape.part(60).x)/2  
                        mouth_h = abs(shape.part(66).y + shape.part(67).y + shape.part(65).y - shape.part(62).y-shape.part(63).y -shape.part(61).y)  /3 
                        mouth_higth= mouth_h/mouth_w
                    

                        brow_sum = 0  
                        frown_sum = 0  
                        for j in range(17, 21):
                            brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                            frown_sum += shape.part(j + 5).x - shape.part(j).x
                            line_brow_x.append(shape.part(j).x)
                            line_brow_y.append(shape.part(j).y)

                        tempx = np.array(line_brow_x)
                        tempy = np.array(line_brow_y)
                        z1 = np.polyfit(tempx, tempy, 1)  
                        self.brow_k = -round(z1[0], 3)  

                        brow_hight = (brow_sum / 10) / self.face_width  
                        brow_width = (frown_sum / 5) / self.face_width  

                        
                        
                        
                        eye_h_right= abs(((shape.part(41).y +shape.part(40).y )-(shape.part(37).y +shape.part(38).y ))/2)
                        eye_w_right= abs(shape.part(39).x -shape.part(36).x)
                        
                                           
                        eye_h_left= abs(((shape.part(44).y +shape.part(43).y )-(shape.part(47).y +shape.part(46).y ))/2)
                        eye_w_left= abs(shape.part(45).x -shape.part(42).x)
                        
                        eye_right=eye_h_right/eye_w_right
                        eye_left=eye_h_left/eye_w_left
                        
                        
                        
                        eye_hight=(eye_right+eye_left)/2
                        
                        
                        
                        
                        eye_hight_thread=0.3
                        cv2.putText(im_rd, "ET: "+str(round(eye_hight,2))+"/"+str(eye_hight_thread), (10,440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), line_cuxi, 3)
                        
                        
                       

                        if eye_hight >= eye_hight_thread:
                            
                            state_cur_eye=1
                            time_start_eye=time.time()
                            
                            if begin_eye==0:
                                cv2.putText(im_rd, "eye: open", (10,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 3)
                                pass
                            elif begin_eye==1:
                                
                                timeTotalend_eye=time.time()
                                timeTotal_eye=timeTotalend_eye-timeTotalbigin_eye
                                if timeTotal_eye>60:
                                    
                                    if coutClose_eye>threshold_set: 
                                        
                                        cv2.putText(im_rd, "eye:sleep", (10,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 3)
                                        print("60s close ",coutClose_eye, "> ",threshold_set,"sleep")
                                  
                                        state_last_eye=2
                                         
                                    else:
                                        cv2.putText(im_rd, "eye: nature", (10,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 3)
                                        print("60s close  <",coutClose_eye,"nature")
                                        state_last_eye=1
                                       
                                        
                                    begin_eye=0
                                else:
                                    cv2.putText(im_rd, "eye: open", (10,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 3)
                                    
                                    
                                    
                        else: 
                            
                            if state_cur_eye == 1: 
                                time_start_eye = time.time() 
                                state_cur_eye=12  
                            
                            elif state_cur_eye==12:
                                
                                time_end_eye= time.time()
                            
                                timeClose_eye=time_end_eye-time_start_eye
                            
                                if timeClose_eye>=threshold_set: 
                                    state_cur_eye=2 
                                    coutClose_eye=coutClose_eye+1 
                                    
                            elif state_cur_eye==2:
                                
                                if begin_eye==0:
                                    begin_eye=1
                                    coutClose_eye=1
                                    timeTotalbigin_eye=time.time()
                                elif begin_eye==1:
                                    
                                    timeTotalend_eye=time.time()
                                    timeTotal_eye=timeTotalend_eye-timeTotalbigin_eye
                                    
                                    pass
                                
                                time_end_eye= time.time() 
                                timeClose_eye=time_end_eye-time_start_eye
                                
                                if timeClose_eye>20:
                                    state_cur_eye=hreshold_set 
                                    cv2.putText(im_rd, "eye: sleep", (10,   20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), line_cuxi, 3)
                                    print("close eye time more than 10s ,sleep")
                                    
                                    
                                    state_last_eye=2
                                else:
                                    cv2.putText(im_rd, "closing eye", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), line_cuxi, 3)
                            
                            
                            
                            
                            
                        if state_last_eye==1:
                            cv2.putText(im_rd, "LastState:nature", (10,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 2)
                            GPIO.output(pin_alram, GPIO.LOW)  #no alram
                        elif state_last_eye==2:
                            cv2.putText(im_rd, "LastState:sleep", (10,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 2)
                            GPIO.output(pin_alram, GPIO.HIGH)  # alram
                        elif state_last_eye==0:
                            cv2.putText(im_rd, "LastState:wait...", (10,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 2)
                        
                            
                            
                            
                        cv2.putText(im_rd, "EyeCloseCout:"+str(coutClose_eye), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), line_cuxi, 3)
                        cv2.putText(im_rd, "TimeClose_eye:"+str(round(timeClose_eye,1)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), line_cuxi, 3)
                        cv2.putText(im_rd, "TimeTotal_eye:"+str(round(timeTotal_eye,1)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), line_cuxi, 3)
                        

                        mouth_hight_thread = 0.3
                        cv2.putText(im_rd, "MT: "+str(round(mouth_higth,2))+"/"+str(mouth_hight_thread), (440,440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), line_cuxi, 3)
                        

                        
                        if round(mouth_higth >= mouth_hight_thread):
                            cv2.putText(im_rd, "mouth:open", (440, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), line_cuxi, 4)
                            
                            if state_cur_mouth==1:
                                time_start_mouth=time.time()
                                state_cur_mouth=12
                                
                            elif state_cur_mouth==12:
                                time_end_mouth=time.time()
                                timeClose_mouth=time_end_mouth-time_start_mouth
                                
                                if timeClose_mouth>threshold_set:
                                    state_cur_mouth=2
                                    coutClose_mouth=coutClose_mouth+1
                                    
                            elif state_cur_mouth==2:
                                if begin_mouth==0:
                                    coutClose_mouth=1
                                    begin_mouth=1
                                    timeTotalbigin_mouth=time.time()
                                    
                                elif begin_mouth==1:
                                    timeTotalend_mouth=time.time()
                                    timeTotal_mouth=timeTotalend_mouth-timeTotalbigin_mouth
                                    
                                    if timeTotal_mouth>60:
                                        begin_mouth=0
                                        if coutClose_mouth>threshold_set:
                                            print(" mouth cout",coutClose_mouth,">",threshold_set,"sleep ")
                                            state_last_mouth=2
                                        else:
                                            print(" mouth cout",coutClose_mouth,"<",threshold_set,"nature ")
                                            state_last_mouth=1
                                        

                                pass
                      

                        else:
                            cv2.putText(im_rd, "mouth:close", (440, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), line_cuxi, 4)
                            state_cur_mouth=1
                            
                            if begin_mouth==0:
                                
                                pass
                            
                            elif begin_mouth==1:
                                timeTotalend_mouth=time.time()
                                timeTotal_mouth=timeTotalend_mouth-timeTotalbigin_mouth
                                if timeTotal_mouth>60:
                                    begin_mouth=0
                                  
                                    if coutClose_mouth>threshold_set:
                                        
                                        print(" mouth cout",coutClose_mouth,">",threshold_set,"sleep ")
                                        state_last_mouth=2
                                    else:
                                        print(" mouth cout",coutClose_mouth,"<",threshold_set,"nature ")
                                        state_last_mouth=1
                                
                                    
                                
                        cv2.putText(im_rd, "mouthCloseCout:"+str(coutClose_mouth), (440, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), line_cuxi, 4)
                        cv2.putText(im_rd, "TimeClose_mouth:"+str(round(timeClose_mouth,1)), (440, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), line_cuxi, 4)
                        cv2.putText(im_rd, "TimeTotal_mouth:"+str(round(timeTotal_mouth,1)), (440, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), line_cuxi, 4)
                        
                        if state_last_mouth==1:
                            cv2.putText(im_rd, "LastState:nature", (440,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 2)
                            GPIO.output(pin_alram, GPIO.LOW)  #no alram
                        elif state_last_mouth==2:
                            cv2.putText(im_rd, "LastState:sleep", (440,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 2)
                            GPIO.output(pin_alram, GPIO.HIGH)  # alram
                        elif state_last_mouth==0:
                            cv2.putText(im_rd, "LastState:wait...", (440,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 0), line_cuxi, 2)
                         

                
                
            else:
                cv2.putText(im_rd, "No Face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

  
            cv2.namedWindow("camera",0)
            cv2.imshow("camera", im_rd)
            
            k=cv2.waitKey(1)
            if (k == ord('s')):
                self.cnt+=1
                cv2.imwrite("screenshoot"+str(self.cnt)+".jpg", im_rd)
            elif(k == ord('q')):
                break

        self.cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()
