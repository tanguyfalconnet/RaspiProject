cd mjpg-streamer-raspi/mjpg-streamer-experimental/
export LD_LIBRARY_PATH=.
./mjpg_streamer -o "output_http.so -w ./www" -i "input_raspicam.so -x 1280 -y 720 -fps 15 -ex night"
cd ../..
