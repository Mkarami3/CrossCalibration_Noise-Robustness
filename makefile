all:
	g++ Calibration.cpp -o calib.out `pkg-config --cflags --libs opencv`


	
