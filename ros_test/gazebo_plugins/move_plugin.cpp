#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/ByteMultiArray.h"

namespace gazebo
{
  /// \brief A plugin to control a Velodyne sensor.
  class VelodynePlugin : public ModelPlugin
  { 
    private:
    physics::ModelPtr model;
    physics::JointPtr joint_left;
    physics::JointPtr joint_right;
    physics::Joint_V joints;
    /// \brief A node used for transport
    transport::NodePtr node;
    /// \brief A subscriber to a named topic.
    transport::SubscriberPtr sub;
    common::PID pid;
    /// \brief A node use for ROS transport
    std::unique_ptr<ros::NodeHandle> rosNode;
    /// \brief A ROS subscriber
    ros::Subscriber rosSub;
    /// \brief A ROS callbackqueue that helps process messages
    ros::CallbackQueue rosQueue;
    /// \brief A thread the keeps running the rosQueue
    std::thread rosQueueThread;

    /// \brief Constructor
    public: VelodynePlugin() {}

    /// \brief The load function is called by Gazebo when the plugin is
    /// inserted into simulation
    /// \param[in] _model A pointer to the model that this plugin is
    /// attached to.
    /// \param[in] _sdf A pointer to the plugin's SDF element.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      // Just output a message for now
      std::cerr << "\nThe velodyne plugin is attach to model[" <<
        _model->GetName() << "]\n";
      
      // Safety check
      if (_model->GetJointCount() < 2)
      {
        std::cerr << "Invalid joint count, Velodyne plugin not loaded\n";
        return;
      }
      this->model = _model;

      // Create the node
      this->node = transport::NodePtr(new transport::Node());

      this->node->Init(this->model->GetWorld()->Name());

      // Create a topic name  
      std::string topicName = "~/vel_cmd";

      // Subscribe to the topic, and register a callback
      this->sub = this->node->Subscribe(topicName,
      &VelodynePlugin::OnMsg, this);

      this->joint_left = _model->GetJoints()[1]; //left
      this->joint_right = _model->GetJoints()[2]; //right

      // Setup a P-controller, with a gain of 0.1.
      this->pid = common::PID(0.1, 0, 0);

      // Apply the P-controller to the joint.
      this->model->GetJointController()->SetVelocityPID(
          this->joint_left->GetScopedName(), this->pid);
    
      // Apply the P-controller to the joint.
      this->model->GetJointController()->SetVelocityPID(
          this->joint_right->GetScopedName(), this->pid);
          
            // Initialize ros, if it has not already bee initialized.
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "gazebo_client",
            ros::init_options::NoSigintHandler);
      }

      // Create our ROS node. This acts in a similar manner to
      // the Gazebo node
      this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

      // Create a named topic, and subscribe to it.
      ros::SubscribeOptions so =
        ros::SubscribeOptions::create<std_msgs::ByteMultiArray>(
            "/vel_cmd",
            1,
            boost::bind(&VelodynePlugin::OnRosMsg, this, _1),
            ros::VoidPtr(), &this->rosQueue);
      this->rosSub = this->rosNode->subscribe(so);

      // Spin up the queue helper thread.
      this->rosQueueThread =
        std::thread(std::bind(&VelodynePlugin::QueueThread, this));
      }

      public: void SetVelocity(const double &left, const double &right)
      { 
        std::cerr << "SetVelocity: " << left << " " << right << "\n";

        // Set the joint's target velocity.
        this->model->GetJointController()->SetVelocityTarget(
        this->joint_left->GetScopedName(), left);

        // Set the joint's target velocity.
        this->model->GetJointController()->SetVelocityTarget(
        this->joint_right->GetScopedName(), right);
      }
      /// \brief Handle incoming message
      private: void OnMsg(ConstVector3dPtr &_msg)
      {
        this->SetVelocity(_msg->x(), _msg->y());
      }
      public: void OnRosMsg(const std_msgs::ByteMultiArrayConstPtr &_msg)
      {
        this->SetVelocity(_msg->data[0], _msg->data[1]);
      }

      /// \brief ROS helper function that processes messages
      private: void QueueThread()
      {
        static const double timeout = 0.01;
        while (this->rosNode->ok())
        {
          this->rosQueue.callAvailable(ros::WallDuration(timeout));
        }
      }
  };
  GZ_REGISTER_MODEL_PLUGIN(VelodynePlugin)
}