#include <AFMotor.h>

AF_DCMotor motor_left(2);
AF_DCMotor motor_right(1);

struct MotorHandler{
    AF_DCMotor motor_left(2);
    AF_DCMotor motor_right(1);

    void goRight(int miliseconds){
      this->motor_left.setSpeed(200)
      this->motor_right.setSpeed(100)
      this->motor_left.run(FORWARD);
      this->motor_right.run(FORWARD);
      delay(miliseconds);
      this->motor_left.run(RELEASE);
      this->motor_right.run(RELEASE);
    }
    void goForward(int seconds){
      this->motor_left.setSpeed(200);
      this->motor_right.setSpeed(200);
      this->motor_right.run(FORWARD);
      this->motor_left.run(FORWARD);
      delay(miliseconds);
      this->motor_left.run(RELEASE);
      this->motor_right.run(RELEASE);
    }
    void goLeft(int miliseconds){
      this->motor_left.setSpeed(100);
      this->motor_right.setSpeed(200);
      this->motor_right.run(FORWARD);
      this->motor_left.run(FORWARD);
      delay(miliseconds);
      this->motor_left.run(RELEASE);
      this->motor_right.run(RELEASE);
    }
}steer;

void setup() {
  Serial.begin(9600);
  Serial.write("it works!");
}

// the loop function runs over and over again forever
void loop() {
  if(Serial.available()>0){

    char direction = (char)(Serial.read()); // possible directions: L - left, R - right, F - forward
    Serial.write("%c ", direction);
    
    if(direction=='L'){
      steer.goLeft(1000);
    }else if(direction=='R'){
      steer.goRight(1000);
    }else if(direction=='F'){
      steer.goForward(1000);
    }else{ return; } // got some rubbish message
      delay(1000);
  }
}