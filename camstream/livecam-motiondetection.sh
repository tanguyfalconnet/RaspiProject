cd ../motiondetection
source ~/.profile
workon cv
python pi_surveillance.py --conf conf.json &
cd ../camstream/mjpg-streamer-file/mjpg-streamer/
export LD_LIBRARY_PATH=.
./mjpg_streamer -o "output_http.so -w ./www" -i "input_file.so -f /home/pi/OpenCV/motiondetection -n output.jpg"
cd ../..
