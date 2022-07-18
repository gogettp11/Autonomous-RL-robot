Autonomous-RL-robot

Hardware:
    JetsonNano
    ArduinoUno
    Internet Camera with mic
    2 dc motors
Software:
    Deep Learning RL Agent acting in gym-like env

Rostopic list for checking avaiable topics
rosrun rviz rviz -> after opening you can add by topic and check what robot see

Instruction for starting:
1. compile *.so from gazebo_plugins cmake CMakeLists.txt ./build and then make
2. move libmove_plugin.so to ros/noetic/lib
3. start 'roscore' and wait
4. start 'rosrun gazebo_ros gazebo training_world.xml' and wait
5. start the main script
