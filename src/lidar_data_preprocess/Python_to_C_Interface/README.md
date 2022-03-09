# README

## Requirements
- Python 2.7
- matplotlib Python library (optional)
- numpy

The LiDAR data pre-processing pipeline is contained in LidarPreprocess.c.

## To compile LidarPreprocess.c:
make

## To run SampleProgram.py:
python SampleProgram.py

## Comments:
- After running make, a shared object named LidarPreprocess.so will be created.
- The shared object is called by the code in SampleProgram.py.
- Change the filepath to LidarPreprocess.so in SampleProgram.py.
- Change the filepath to the .bin Velodyne LiDAR files in LidarPreprocess.c.
