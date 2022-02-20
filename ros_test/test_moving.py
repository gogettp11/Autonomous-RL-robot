import rospy
#msgs
from gazebo_msgs.srv import ApplyBodyWrench
#position
from geometry_msgs.msg import *
import asyncio

async def wrap(fun, *args):
    fun(*args)

async def main():
    publisher = rospy.Publisher('/%s/cmd_vel' % topic, Twist, queue_size=8)
    msg = Twist()
    msg.linear.x = 1.0
    msg.angular.z = 0.5
    publisher.publish(msg)

asyncio.run(main())