#include <SoftwareSerial.h>
#include <PZEM004TV1.h>
#include <AltSoftSerial.h> 

//GSM Module
//SoftwareSerial gprsSerial(7, 8);

// PZEM004T Power Monitor
#define RX_PIN 12 // Arduino RX pin connected to PZEM TX
#define TX_PIN 13  // Arduino TX pin connected to PZEM RX
PZEM004TV1 pzem(RX_PIN, TX_PIN);

// GSM using AltSoftSerial (Pins 8, 9)
AltSoftSerial gprsSerial;  


int relay_1 = 4;
int relay_2 = 5;
int relay_3 = 6;
int relay_4 = 3;

void setup() {
 /* Debugging serial */
 Serial.begin(9600);

 //GSM module
 gprsSerial.begin(9600);// if it is not working change it to 9600

 pinMode(relay_1, OUTPUT); //1 RELAY shut down
 pinMode(relay_2, OUTPUT); //2 RELAY C1
 pinMode(relay_3, OUTPUT); //3 RELAY C2
 pinMode(relay_4, OUTPUT); //4 RELAY C3
}

void loop() {

 

//Read the data from the sensor

  float voltage = pzem.readVoltage();
  float current = pzem.readCurrent();
  float power = pzem.readPower();
  float energy = pzem.readEnergy();
  float frequency = pzem.readFrequency();
  float powerFactor = pzem.readPowerFactor();
  double activePower=voltage*current*powerFactor;
  float reactivePower = sin(acos(powerFactor)) * voltage * current;

  if (voltage > 244 || voltage< 216) { // Protection
  digitalWrite(relay_1, LOW);

 }

  if (current > 13) { // Protection
  digitalWrite(relay_1, LOW);

 }

  if (powerFactor < 0.9) {
  if (reactivePower < 19.8) {

  digitalWrite(relay_2, HIGH);
  digitalWrite(relay_3, HIGH);
  digitalWrite(relay_4, HIGH); 

 } else if (reactivePower >= 19.8 && reactivePower < 41.3) {
  digitalWrite(relay_2, LOW);
  digitalWrite(relay_3, HIGH);
  digitalWrite(relay_4, HIGH);

 } else if (reactivePower >= 41.3 && reactivePower < 83) {
  digitalWrite(relay_2, HIGH);
  digitalWrite(relay_3, LOW);
  digitalWrite(relay_4, HIGH);

 } else if (reactivePower >= 83 && reactivePower < 102.4) {
  digitalWrite(relay_2, LOW);
  digitalWrite(relay_3, LOW);
  digitalWrite(relay_4, HIGH);

 } else if (reactivePower >= 102.4 && reactivePower < 164) {
  digitalWrite(relay_2, HIGH);
  digitalWrite(relay_3, HIGH);
  digitalWrite(relay_4, LOW); 

 } else if (reactivePower >= 164 && reactivePower < 184) {
  digitalWrite(relay_2, LOW);
  digitalWrite(relay_3, HIGH);
  digitalWrite(relay_4, LOW); 

 } else if (reactivePower >= 184 && reactivePower < 205.3) {
  digitalWrite(relay_2, HIGH);
  digitalWrite(relay_3, LOW);
  digitalWrite(relay_4, LOW);

 } else {//reactivePower > 205.3 
  digitalWrite(relay_2, LOW);
  digitalWrite(relay_3, LOW);
  digitalWrite(relay_4, LOW);
 }

 }
 // Check if the data is valid
 if (isnan(voltage) || voltage == 0 || isnan(current)) {  
  Serial.println("Error reading ");
  check(); 

 } else {

 // Print the values to the Serial console
 
 
 Serial.print("Voltage: "); Serial.print(voltage); Serial.println("V");
 Serial.print("Current: "); Serial.print(current); Serial.println("A");
 Serial.print("Energy: "); Serial.print(energy, 3); Serial.println("Wh");
 Serial.print("Frequency: "); Serial.print(frequency, 1); Serial.println("Hz");
 Serial.print("reactivePower: "); Serial.print(reactivePower); Serial.println("VAR");
 Serial.print("activePower: "); Serial.print(activePower); Serial.println("W");
 Serial.print("PowerFactor: "); Serial.println(powerFactor);
 
 }

 Serial.println();
 delay(1000);


 if (gprsSerial.available()) {
 Serial.write(gprsSerial.read());
 }

 gprsSerial.println("AT");
 delay(1000);
 gprsSerial.println("AT+CPIN?");
 delay(1000);
gprsSerial.println("AT+CREG?");
 delay(1000);
 gprsSerial.println("AT+CGATT?");
 delay(1000);
 gprsSerial.println("AT+CIPSHUT");
 delay(1000);
 gprsSerial.println("AT+CIPSTATUS");
 delay(1000);
 gprsSerial.println("AT+CIPMUX=0");
 delay(1000);

 //ShowSerialData();

 gprsSerial.println("AT+CSTT=\"APN\",\"mobitel\"");//start task and setting the APN,
 delay(1000);

 //ShowSerialData();

 gprsSerial.println("AT+CIICR");//bring up wireless connection
 delay(1000);

// ShowSerialData();

 gprsSerial.println("AT+CIFSR");//get local IP adress
 delay(1000);

 //ShowSerialData();

 gprsSerial.println("AT+CIPSPRT=0");
 delay(1000);

 //ShowSerialData();

 gprsSerial.println("AT+CIPSTART=\"TCP\",\"api.thingspeak.com\",\"80\"");//start up the connection
 delay(1000);

 //ShowSerialData();

 gprsSerial.println("AT+CIPSEND");//begin send data to remote server
 delay(1000);
 //ShowSerialData();
//api key=paste your Thingspeak write API key
 String str = "GET https://api.thingspeak.com/update?api_key=****************&field1=" + 
String(voltage) + "&field2=" + String(current) + "&field3=" + String(power) + "&field4=" + 
String(energy) + "&field5=" + String(frequency) + "&field6=" + String(powerFactor);
 Serial.println(str);

 gprsSerial.println(str);//begin send data to remote server
 delay(1000);
 //ShowSerialData();

 gprsSerial.println((char)26);//sending
 delay(1000);//waitting for reply, important! the time is base on the condition of internet
 gprsSerial.println();

 //ShowSerialData();

 gprsSerial.println("AT+CIPSHUT");//close the connection
 delay(1000);
}

void check() {
 gprsSerial.println("AT+CMGF=1"); //Set the GSM Module in Text Mode
 delay(1000);
 gprsSerial.println("AT+CMGS=\" * \"\r"); // Replace it with your mobile number here (*)
 delay(1000);
 gprsSerial.println("warning alert:Incoming Supply Failure"); // The SMS text you want to send
 delay(1000);
 gprsSerial.println((char)26); // ASCII code of CTRL+Z
 delay(1000);
 delay(5000);
}
void ShowSerialData()
{
 while (gprsSerial.available() != 0)
 Serial.write(gprsSerial.read());
 delay(1000);
}

