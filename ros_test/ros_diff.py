import rospy
from geometry_msgs.msg import Twist

def talker():
    pub = rospy.Publisher('/gz/cmd_vel', Twist, queue_size=10)
    t = Twist()
    t.angular.x = 2000
    t.angular.y = 200
    t.angular.z = 200
    t.linear.x = 200
    t.linear.y = 200
    t.linear.z = 200
    rospy.init_node('talker', anonymous=True)
    pub.publish(t)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass