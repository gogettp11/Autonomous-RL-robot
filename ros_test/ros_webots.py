import rospy
from std_msgs.msg import Float64
import random
def cb(data):
    print(f"dane: {data.data}")
def talker():
    pub = rospy.Publisher('motor', Float64, queue_size=10)
    subscriber = rospy.Subscriber('sensor', Float64, callback= cb)
    rospy.init_node('python_controller', anonymous=True)
    a = -10.0
    rate = rospy.Rate(2) # 10hz
    while not rospy.is_shutdown():
        a = random.uniform(-10,10)
        pub.publish(a)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass