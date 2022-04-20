#import libraries

import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pipeline
import globals
import sys

globals.init()


##Working on an input video
#cap=cv2.VideoCapture("project_video.mp4")

#src_path=input("Enter the source video path: ")
#dstn_path=input("Enter the destination video path: ")
#isDebug=input("Enter 0 for no debbuged output video, 1 for debugged output video: ")
cap=cv2.VideoCapture(sys.argv[1])

fps=cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
frameSize = (int(width), int(height))



out = cv2.VideoWriter(sys.argv[2],0x7634706d , fps, frameSize)

i=0
while(cap.isOpened()):
    i=i+1
    ret,frame=cap.read()
    if ret==1:
        #Call the pipeline in a single the captured frame from the video
        out_frame=pipeline.lane_finding_pipeline(frame,sys.argv[3])
        out.write(out_frame)
        print("Producing output video, ",int((i/frame_count)*100),"% completed.", end='\r')

        #cv2.imshow("output",out_frame)
        #if cv2.waitKey(1)&0xFF==ord('q'):
        #    break
    else:
        print("Video has been Produced")
        break

cap.release()
out.release()
cv2.destroyAllWindows()



#video_output = 'harder_challenge_video_output.mp4'
#clip1 = VideoFileClip("harder_challenge_video.mp4")
#output_clip = clip1.fl_image(lane_finding_pipeline)
#output_clip.write_videofile(video_output, audio=False)
