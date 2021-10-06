#include <AFMotor.h>

AF_DCMotor motor(2);

void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  Serial.write("it works!");
}

// the loop function runs over and over again forever
void loop() {
  // if(Serial.available()>0){
  //     motor.setSpeed(200);
  //     motor.run(FORWARD);
  //     delay(2000);
  //     motor.run(RELEASE);
  //     delay(2000);
  // }
  if(Serial.available()>0){
    char left_right = (char)(Serial.read());
    motor.setSpeed(200);
    if(left_right=='l'){
      Serial.write("FORWARD\n");
      motor.run(FORWARD);
    }else if(left_right=='r'){
      Serial.write("BACKWARD\n");
      motor.run(BACKWARD);
    }else
      Serial.write("WRONG SIGN\n"+left_right);
      delay(2000);
      motor.run(RELEASE);
  }
}
// if(Serial.available()>0){
//     char left_right = (char)(Serial.read());
//       motor.setSpeed(200);
//       if(left_right=='l'){
//         Serial.write("FORWARD\n");
//         motor.run(FORWARD);
//       }else if(left_right=='r'){
//         Serial.write("BACKWARD\n");
//         motor.run(BACKWARD);
//       }
//       delay(2000);
//       motor.run(RELEASE);
//   }