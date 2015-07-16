# RaspiProject

Hi !

You need:
- a Raspberry Pi 2 (maybe a B+ but not tried)
- a RaspiCam

You need to:
- Install OpenCV and Python (http://www.pyimagesearch.com/2015/02/23/install-opencv-and-python-on-your-raspberry-pi-2-and-b/) 
- Install MJPG to directly stream the RaspiCam (https://github.com/jacksonliam/mjpg-streamer) -> Put it into camstream
- Install MJPG to use it with OpenCV (http://downloads.sourceforge.net/project/mjpg-streamer/mjpg-streamer_r94-1_i386.deb?r=&ts=1437065215&use_mirror=vorboss) -> Put it into camstream

If all installations are done and directory have their right names, you can try to launch :
- livestream.sh to livestream your RaspiCam
- livestream-motiondetection.sh to analyse and stream your RaspiCam (you have to kill python manually after use)

(see in shell scripts for the right directory name)

Have fun !

