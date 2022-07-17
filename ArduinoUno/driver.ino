#include <AFMotor.h>

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
    void goForward(int miliseconds){
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

    // read data
    String incoming_data = Serial.readString(); // data format: Direction [1 byte char], Movement Lenghts [number converted to string]
    char direction = incoming_data[0];
    incoming_data.remove(0);
    int miliseconds = incoming_data.toInt();

    if(miliseconds == 0)
      miliseconds = 1000;
    
    // move
    if(direction=='L'){
      steer.goLeft(miliseconds);
    }else if(direction=='R'){
      steer.goRight(miliseconds);
    }else if(direction=='F'){
      steer.goForward(miliseconds);
    }else{ return; } // got some rubbish message
    Serial.write("%c", direction);
  }
}