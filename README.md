Autonomous-RL-robot

Hardware:
    JetsonNano
    ArduinoUno
    Internet Camera with mic
    2 dc motors
Software:
    Deep Learning RL Agent acting in gym-like env

Rostopic list for checking avaiable topics
rosrun gazebo_ros gazebo world.xml
rviz -> after opening you can add by topic

Instruction for starting:
1. compile *.so from gazebo_plugins cmake CMakeLists.txt ./build and then make
2. move libmove_plugin.so ros/noetic/lib
