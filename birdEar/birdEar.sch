EESchema Schematic File Version 4
LIBS:birdEar-cache
EELAYER 30 0
EELAYER END
$Descr USLetter 11000 8500
encoding utf-8
Sheet 1 2
Title "Bird Ear"
Date "2019-09-25"
Rev "A"
Comp "Ryan Walker, Jin Seo"
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Device:Microphone MK1
U 1 1 5D8C2F06
P 850 3700
F 0 "MK1" H 980 3746 50  0000 L CNN
F 1 "CMA-4544PF-W" H 980 3655 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical" V 850 3800 50  0001 C CNN
F 3 "~" V 850 3800 50  0001 C CNN
	1    850  3700
	1    0    0    -1  
$EndComp
$Comp
L Device:Battery_Cell BT1
U 1 1 5D8C35FD
P 1200 1600
F 0 "BT1" H 1318 1696 50  0000 L CNN
F 1 "Battery_Cell" H 1318 1605 50  0000 L CNN
F 2 "Battery:BatteryHolder_Keystone_1042_1x18650" V 1200 1660 50  0001 C CNN
F 3 "~" V 1200 1660 50  0001 C CNN
F 4 "1042P" H 1200 1600 50  0001 C CNN "Part Number"
	1    1200 1600
	1    0    0    -1  
$EndComp
$Sheet
S 6450 3500 1300 1000
U 5D8C399D
F0 "MCU" 50
F1 "MCU.sch" 50
F2 "Audio_In" I L 6450 3600 50 
F3 "Loud" I L 6450 3700 50 
F4 "I2C_SCL" I R 7750 3600 50 
F5 "I2C_SDA" I R 7750 3700 50 
F6 "PB1" I L 6450 3850 50 
F7 "PB2" I L 6450 3950 50 
$EndSheet
$Comp
L Regulator_Linear:AZ1117-3.3 U2
U 1 1 5D8C3D0D
P 2650 1350
F 0 "U2" H 2650 1592 50  0000 C CNN
F 1 "AZ1117EH-3.3TRG1" H 2650 1501 50  0000 C CNN
F 2 "Package_TO_SOT_SMD:SOT-223-3_TabPin2" H 2650 1600 50  0001 C CIN
F 3 "https://www.diodes.com/assets/Datasheets/AZ1117.pdf" H 2650 1350 50  0001 C CNN
	1    2650 1350
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C3
U 1 1 5D8C4221
P 2050 1550
F 0 "C3" H 2150 1600 50  0000 L CNN
F 1 "1uF" H 2142 1505 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 2050 1550 50  0001 C CNN
F 3 "~" H 2050 1550 50  0001 C CNN
	1    2050 1550
	1    0    0    -1  
$EndComp
Wire Wire Line
	2050 1350 2350 1350
Wire Wire Line
	2050 1750 2650 1750
Connection ~ 2650 1750
Wire Wire Line
	2650 1750 2650 1800
Wire Wire Line
	2050 1450 2050 1350
Wire Wire Line
	2050 1750 2050 1650
$Comp
L Device:C_Small C4
U 1 1 5D8C447C
P 3250 1550
F 0 "C4" H 3350 1600 50  0000 L CNN
F 1 "1uF" H 3342 1505 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 3250 1550 50  0001 C CNN
F 3 "~" H 3250 1550 50  0001 C CNN
	1    3250 1550
	1    0    0    -1  
$EndComp
Wire Wire Line
	2950 1350 3250 1350
Wire Wire Line
	3250 1350 3250 1450
Wire Wire Line
	3250 1650 3250 1750
Wire Wire Line
	3250 1750 2650 1750
Wire Wire Line
	2650 1650 2650 1750
Connection ~ 2050 1350
Wire Wire Line
	1200 1400 1200 1350
Wire Wire Line
	1200 1700 1200 1750
Wire Wire Line
	1200 1750 2050 1750
Connection ~ 2050 1750
$Comp
L power:+3.3V #PWR012
U 1 1 5D8C4A64
P 3250 1250
F 0 "#PWR012" H 3250 1100 50  0001 C CNN
F 1 "+3.3V" H 3265 1423 50  0000 C CNN
F 2 "" H 3250 1250 50  0001 C CNN
F 3 "" H 3250 1250 50  0001 C CNN
	1    3250 1250
	1    0    0    -1  
$EndComp
Wire Wire Line
	3250 1250 3250 1350
Connection ~ 3250 1350
Text Notes 600  2050 0    50   ~ 0
18650 Cell. Ensure battery has \nundervoltage shutoff, place \nmultiple battaries in parallel for more power.
Wire Wire Line
	850  3900 850  4050
$Comp
L power:GND #PWR01
U 1 1 5D8C4D2E
P 850 4050
F 0 "#PWR01" H 850 3800 50  0001 C CNN
F 1 "GND" H 855 3877 50  0000 C CNN
F 2 "" H 850 4050 50  0001 C CNN
F 3 "" H 850 4050 50  0001 C CNN
	1    850  4050
	1    0    0    -1  
$EndComp
Wire Wire Line
	850  3500 850  3300
$Comp
L Amplifier_Operational:LM324 U1
U 1 1 5D8C521D
P 2850 3200
F 0 "U1" H 2850 2833 50  0000 C CNN
F 1 "LM324" H 2850 2924 50  0000 C CNN
F 2 "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm" H 2800 3300 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 2900 3400 50  0001 C CNN
F 4 "LM324DR" H 2850 3200 50  0001 C CNN "Part Number"
	1    2850 3200
	1    0    0    1   
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 2 1 5D8C52C4
P 2950 4950
F 0 "U1" H 2950 5317 50  0000 C CNN
F 1 "LM324" H 2950 5226 50  0000 C CNN
F 2 "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm" H 2900 5050 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 3000 5150 50  0001 C CNN
F 4 "LM324DR" H 2950 4950 50  0001 C CNN "Part Number"
	2    2950 4950
	1    0    0    -1  
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 4 1 5D8C5409
P 2900 7200
F 0 "U1" H 2900 7567 50  0000 C CNN
F 1 "LM324" H 2900 7476 50  0000 C CNN
F 2 "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm" H 2850 7300 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 2950 7400 50  0001 C CNN
F 4 "LM324DR" H 2900 7200 50  0001 C CNN "Part Number"
	4    2900 7200
	1    0    0    -1  
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 5 1 5D8C54A9
P 900 7150
F 0 "U1" H 500 7200 50  0000 L CNN
F 1 "LM324" H 500 7100 50  0000 L CNN
F 2 "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm" H 850 7250 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 950 7350 50  0001 C CNN
	5    900  7150
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR03
U 1 1 5D8C59B1
P 800 7600
F 0 "#PWR03" H 800 7350 50  0001 C CNN
F 1 "GND" H 805 7427 50  0000 C CNN
F 2 "" H 800 7600 50  0001 C CNN
F 3 "" H 800 7600 50  0001 C CNN
	1    800  7600
	1    0    0    -1  
$EndComp
$Comp
L power:+3.3V #PWR02
U 1 1 5D8C5AA2
P 800 6700
F 0 "#PWR02" H 800 6550 50  0001 C CNN
F 1 "+3.3V" H 815 6873 50  0000 C CNN
F 2 "" H 800 6700 50  0001 C CNN
F 3 "" H 800 6700 50  0001 C CNN
	1    800  6700
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R1
U 1 1 5D8C65A2
P 1100 3100
F 0 "R1" H 1168 3146 50  0000 L CNN
F 1 "2.2k" H 1168 3055 50  0000 L CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 1100 3100 50  0001 C CNN
F 3 "~" H 1100 3100 50  0001 C CNN
	1    1100 3100
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C2
U 1 1 5D8C666E
P 1300 3300
F 0 "C2" H 1400 3350 50  0000 L CNN
F 1 "1uF" H 1392 3255 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 1300 3300 50  0001 C CNN
F 3 "~" H 1300 3300 50  0001 C CNN
	1    1300 3300
	0    1    1    0   
$EndComp
$Comp
L power:+3.3V #PWR04
U 1 1 5D8C70F2
P 1100 2900
F 0 "#PWR04" H 1100 2750 50  0001 C CNN
F 1 "+3.3V" H 1115 3073 50  0000 C CNN
F 2 "" H 1100 2900 50  0001 C CNN
F 3 "" H 1100 2900 50  0001 C CNN
	1    1100 2900
	1    0    0    -1  
$EndComp
Wire Wire Line
	850  3300 1100 3300
Wire Wire Line
	1100 2900 1100 3000
Wire Wire Line
	1100 3200 1100 3300
Connection ~ 1100 3300
Wire Wire Line
	1100 3300 1200 3300
$Comp
L Device:R_Small_US R2
U 1 1 5D8C8826
P 1850 3050
F 0 "R2" H 1918 3096 50  0000 L CNN
F 1 "1M" H 1918 3005 50  0000 L CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 1850 3050 50  0001 C CNN
F 3 "~" H 1850 3050 50  0001 C CNN
	1    1850 3050
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R3
U 1 1 5D8C88C0
P 1850 3550
F 0 "R3" H 1918 3596 50  0000 L CNN
F 1 "1M" H 1918 3505 50  0000 L CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 1850 3550 50  0001 C CNN
F 3 "~" H 1850 3550 50  0001 C CNN
	1    1850 3550
	1    0    0    -1  
$EndComp
Wire Wire Line
	1850 3300 1850 3150
Wire Wire Line
	1400 3300 1850 3300
Wire Wire Line
	1850 3300 1850 3450
Connection ~ 1850 3300
$Comp
L power:GND #PWR07
U 1 1 5D8C93C4
P 1850 3750
F 0 "#PWR07" H 1850 3500 50  0001 C CNN
F 1 "GND" H 1855 3577 50  0000 C CNN
F 2 "" H 1850 3750 50  0001 C CNN
F 3 "" H 1850 3750 50  0001 C CNN
	1    1850 3750
	1    0    0    -1  
$EndComp
Wire Wire Line
	1850 3750 1850 3650
Wire Wire Line
	1850 2850 1850 2950
$Comp
L power:+3.3V #PWR06
U 1 1 5D8C9BE2
P 1850 2850
F 0 "#PWR06" H 1850 2700 50  0001 C CNN
F 1 "+3.3V" H 1865 3023 50  0000 C CNN
F 2 "" H 1850 2850 50  0001 C CNN
F 3 "" H 1850 2850 50  0001 C CNN
	1    1850 2850
	1    0    0    -1  
$EndComp
Wire Wire Line
	1850 3300 2550 3300
$Comp
L Device:C_Small C1
U 1 1 5D8CAEE3
P 1100 6900
F 0 "C1" H 1200 6950 50  0000 L CNN
F 1 "1uF" H 1192 6855 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 1100 6900 50  0001 C CNN
F 3 "~" H 1100 6900 50  0001 C CNN
	1    1100 6900
	1    0    0    -1  
$EndComp
Wire Wire Line
	1100 6800 1100 6750
Wire Wire Line
	1100 6750 800  6750
Wire Wire Line
	800  6750 800  6850
Connection ~ 800  6750
Wire Wire Line
	800  6750 800  6700
$Comp
L power:GND #PWR05
U 1 1 5D8CC801
P 1100 7000
F 0 "#PWR05" H 1100 6750 50  0001 C CNN
F 1 "GND" H 1105 6827 50  0000 C CNN
F 2 "" H 1100 7000 50  0001 C CNN
F 3 "" H 1100 7000 50  0001 C CNN
	1    1100 7000
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R4
U 1 1 5D8CCF3A
P 2300 2500
F 0 "R4" V 2095 2500 50  0000 C CNN
F 1 "100k" V 2186 2500 50  0000 C CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 2300 2500 50  0001 C CNN
F 3 "~" H 2300 2500 50  0001 C CNN
	1    2300 2500
	0    1    1    0   
$EndComp
Wire Wire Line
	2550 3100 2500 3100
Wire Wire Line
	2500 3100 2500 2500
Wire Wire Line
	2500 2500 2400 2500
Wire Wire Line
	2500 2500 2800 2500
Connection ~ 2500 2500
Wire Wire Line
	2200 2500 2100 2500
Wire Wire Line
	2100 2500 2100 2600
$Comp
L power:GND #PWR010
U 1 1 5D8CEF8F
P 2100 2600
F 0 "#PWR010" H 2100 2350 50  0001 C CNN
F 1 "GND" H 2105 2427 50  0000 C CNN
F 2 "" H 2100 2600 50  0001 C CNN
F 3 "" H 2100 2600 50  0001 C CNN
	1    2100 2600
	1    0    0    -1  
$EndComp
Wire Wire Line
	3450 2500 3450 3200
Wire Wire Line
	3450 3200 3150 3200
Connection ~ 3450 3200
$Comp
L Device:R_POT_TRIM_US RV2
U 1 1 5D8D08F7
P 3000 2300
F 0 "RV2" V 2900 2150 50  0000 C CNN
F 1 "1M" V 2900 2300 50  0000 C CNN
F 2 "Potentiometer_THT:Potentiometer_Vishay_T73YP_Vertical" H 3000 2300 50  0001 C CNN
F 3 "~" H 3000 2300 50  0001 C CNN
F 4 "3362P-1-105LF" V 3000 2300 50  0001 C CNN "Part Number"
	1    3000 2300
	0    -1   1    0   
$EndComp
Wire Wire Line
	2850 2300 2800 2300
Wire Wire Line
	2800 2300 2800 2500
Wire Wire Line
	3000 2450 3000 2500
Wire Wire Line
	3000 2500 3450 2500
Text Notes 2650 2700 0    50   ~ 0
Gain: 0:10
Text GLabel 3700 4950 2    50   Output ~ 0
Loud
$Comp
L Device:R_POT_TRIM_US RV1
U 1 1 5D8D3430
P 2050 5050
F 0 "RV1" V 2100 5200 50  0000 C CNN
F 1 "1M" V 1950 5050 50  0000 C CNN
F 2 "Potentiometer_THT:Potentiometer_Vishay_T73YP_Vertical" H 2050 5050 50  0001 C CNN
F 3 "~" H 2050 5050 50  0001 C CNN
F 4 "3362P-1-105LF" V 2050 5050 50  0001 C CNN "Part Number"
	1    2050 5050
	1    0    0    1   
$EndComp
$Comp
L power:+3.3V #PWR08
U 1 1 5D8D36A6
P 2050 4750
F 0 "#PWR08" H 2050 4600 50  0001 C CNN
F 1 "+3.3V" H 2065 4923 50  0000 C CNN
F 2 "" H 2050 4750 50  0001 C CNN
F 3 "" H 2050 4750 50  0001 C CNN
	1    2050 4750
	1    0    0    -1  
$EndComp
Wire Wire Line
	2050 4900 2050 4750
Wire Wire Line
	2050 5200 2050 5350
$Comp
L power:GND #PWR09
U 1 1 5D8D49B3
P 2050 5350
F 0 "#PWR09" H 2050 5100 50  0001 C CNN
F 1 "GND" H 2055 5177 50  0000 C CNN
F 2 "" H 2050 5350 50  0001 C CNN
F 3 "" H 2050 5350 50  0001 C CNN
	1    2050 5350
	1    0    0    -1  
$EndComp
Wire Wire Line
	2200 5050 2650 5050
Wire Wire Line
	2300 4700 2300 4850
Wire Wire Line
	2300 4850 2650 4850
Wire Wire Line
	3250 4950 3600 4950
Text Notes 3050 6600 0    50   ~ 0
Spares
Wire Wire Line
	3600 4950 3600 3700
Wire Wire Line
	3600 3700 6450 3700
Connection ~ 3600 4950
Wire Wire Line
	3600 4950 3700 4950
Text Notes 6750 4450 0    197  ~ 39
STM32
Text Label 2000 3300 0    50   ~ 0
Raw_Audio
Text Label 2300 4700 0    50   ~ 0
Raw_Audio
$Comp
L Sensor_Temperature:Si7050-A20 U4
U 1 1 5D99023B
P 8800 3700
F 0 "U4" H 9144 3746 50  0000 L CNN
F 1 "Si7050-A20" H 9144 3655 50  0000 L CNN
F 2 "Package_DFN_QFN:DFN-6-1EP_3x3mm_P1mm_EP1.5x2.4mm" H 8800 3300 50  0001 C CNN
F 3 "https://www.silabs.com/documents/public/data-sheets/Si7050-1-3-4-5-A20.pdf" H 8600 4000 50  0001 C CNN
	1    8800 3700
	1    0    0    -1  
$EndComp
Wire Wire Line
	8400 3600 7950 3600
Wire Wire Line
	7750 3700 8200 3700
$Comp
L Device:R_Small_US R13
U 1 1 5D9C740B
P 7950 3400
F 0 "R13" H 8018 3446 50  0000 L CNN
F 1 "1M" H 8018 3355 50  0000 L CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 7950 3400 50  0001 C CNN
F 3 "~" H 7950 3400 50  0001 C CNN
	1    7950 3400
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R14
U 1 1 5D9C7D5A
P 8200 3400
F 0 "R14" H 8268 3446 50  0000 L CNN
F 1 "1M" H 8268 3355 50  0000 L CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 8200 3400 50  0001 C CNN
F 3 "~" H 8200 3400 50  0001 C CNN
	1    8200 3400
	1    0    0    -1  
$EndComp
Wire Wire Line
	7950 3500 7950 3600
Connection ~ 7950 3600
Wire Wire Line
	7950 3600 7750 3600
Wire Wire Line
	8200 3500 8200 3700
Connection ~ 8200 3700
Wire Wire Line
	8200 3700 8400 3700
Wire Wire Line
	7950 3300 7950 3050
Wire Wire Line
	7950 3050 8200 3050
Connection ~ 8800 3050
Wire Wire Line
	8800 3050 8800 2950
Wire Wire Line
	8200 3300 8200 3050
Connection ~ 8200 3050
Wire Wire Line
	8200 3050 8800 3050
$Comp
L power:+3.3V #PWR036
U 1 1 5D9D1244
P 8800 2950
F 0 "#PWR036" H 8800 2800 50  0001 C CNN
F 1 "+3.3V" H 8815 3123 50  0000 C CNN
F 2 "" H 8800 2950 50  0001 C CNN
F 3 "" H 8800 2950 50  0001 C CNN
	1    8800 2950
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C24
U 1 1 5D9D286B
P 9200 3200
F 0 "C24" H 9300 3250 50  0000 L CNN
F 1 "1uF" H 9292 3155 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 9200 3200 50  0001 C CNN
F 3 "~" H 9200 3200 50  0001 C CNN
	1    9200 3200
	1    0    0    -1  
$EndComp
Wire Wire Line
	8800 3400 8800 3050
$Comp
L power:GND #PWR038
U 1 1 5D9D6E18
P 9200 3350
F 0 "#PWR038" H 9200 3100 50  0001 C CNN
F 1 "GND" H 9205 3177 50  0000 C CNN
F 2 "" H 9200 3350 50  0001 C CNN
F 3 "" H 9200 3350 50  0001 C CNN
	1    9200 3350
	1    0    0    -1  
$EndComp
Wire Wire Line
	8800 3050 9200 3050
Wire Wire Line
	9200 3050 9200 3100
Wire Wire Line
	9200 3350 9200 3300
$Comp
L power:GND #PWR037
U 1 1 5D9DC977
P 8800 4050
F 0 "#PWR037" H 8800 3800 50  0001 C CNN
F 1 "GND" H 8805 3877 50  0000 C CNN
F 2 "" H 8800 4050 50  0001 C CNN
F 3 "" H 8800 4050 50  0001 C CNN
	1    8800 4050
	1    0    0    -1  
$EndComp
Wire Wire Line
	8800 4050 8800 4000
Wire Wire Line
	5900 5350 5900 5650
$Comp
L power:GND #PWR?
U 1 1 5D9F0862
P 5900 5650
AR Path="/5D8C399D/5D9F0862" Ref="#PWR?"  Part="1" 
AR Path="/5D9F0862" Ref="#PWR034"  Part="1" 
F 0 "#PWR034" H 5900 5400 50  0001 C CNN
F 1 "GND" H 5905 5477 50  0000 C CNN
F 2 "" H 5900 5650 50  0001 C CNN
F 3 "" H 5900 5650 50  0001 C CNN
	1    5900 5650
	1    0    0    -1  
$EndComp
Text Label 4900 5350 0    50   ~ 0
PB1
Wire Wire Line
	7100 5350 7100 5650
$Comp
L power:GND #PWR?
U 1 1 5D9F087C
P 7100 5650
AR Path="/5D8C399D/5D9F087C" Ref="#PWR?"  Part="1" 
AR Path="/5D9F087C" Ref="#PWR035"  Part="1" 
F 0 "#PWR035" H 7100 5400 50  0001 C CNN
F 1 "GND" H 7105 5477 50  0000 C CNN
F 2 "" H 7100 5650 50  0001 C CNN
F 3 "" H 7100 5650 50  0001 C CNN
	1    7100 5650
	1    0    0    -1  
$EndComp
Text Label 6100 5350 0    50   ~ 0
PB2
Text Label 6150 3850 0    50   ~ 0
PB1
Text Label 6150 3950 0    50   ~ 0
PB2
Wire Wire Line
	6150 3850 6450 3850
Wire Wire Line
	6450 3950 6150 3950
$Comp
L Sensor_Optical:LTR-303ALS-01 U5
U 1 1 5DA14906
P 8800 1900
F 0 "U5" H 9244 1946 50  0000 L CNN
F 1 "LTR-303ALS-01" H 9150 1650 50  0000 L CNN
F 2 "OptoDevice:Lite-On_LTR-303ALS-01" H 8800 2350 50  0001 C CNN
F 3 "http://optoelectronics.liteon.com/upload/download/DS86-2013-0004/LTR-303ALS-01_DS_V1.pdf" H 8500 2250 50  0001 C CNN
	1    8800 1900
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0101
U 1 1 5DA15987
P 8800 2400
F 0 "#PWR0101" H 8800 2150 50  0001 C CNN
F 1 "GND" H 8805 2227 50  0000 C CNN
F 2 "" H 8800 2400 50  0001 C CNN
F 3 "" H 8800 2400 50  0001 C CNN
	1    8800 2400
	1    0    0    -1  
$EndComp
Wire Wire Line
	8800 2400 8800 2300
Connection ~ 8800 1150
Wire Wire Line
	8800 1150 8800 1050
$Comp
L power:+3.3V #PWR0102
U 1 1 5DA1B21F
P 8800 1050
F 0 "#PWR0102" H 8800 900 50  0001 C CNN
F 1 "+3.3V" H 8815 1223 50  0000 C CNN
F 2 "" H 8800 1050 50  0001 C CNN
F 3 "" H 8800 1050 50  0001 C CNN
	1    8800 1050
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C25
U 1 1 5DA1B225
P 9200 1300
F 0 "C25" H 9300 1350 50  0000 L CNN
F 1 "1uF" H 9292 1255 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 9200 1300 50  0001 C CNN
F 3 "~" H 9200 1300 50  0001 C CNN
	1    9200 1300
	1    0    0    -1  
$EndComp
Wire Wire Line
	8800 1500 8800 1150
Wire Wire Line
	8800 1150 9200 1150
Wire Wire Line
	9200 1150 9200 1200
$Comp
L power:GND #PWR0103
U 1 1 5DA1DB45
P 9200 1550
F 0 "#PWR0103" H 9200 1300 50  0001 C CNN
F 1 "GND" H 9205 1377 50  0000 C CNN
F 2 "" H 9200 1550 50  0001 C CNN
F 3 "" H 9200 1550 50  0001 C CNN
	1    9200 1550
	1    0    0    -1  
$EndComp
Wire Wire Line
	9200 1400 9200 1550
Wire Wire Line
	8400 1800 8200 1800
Text Label 8200 1800 0    50   ~ 0
SDA
Text Label 8200 2000 0    50   ~ 0
SCL
Wire Wire Line
	8200 2000 8400 2000
Text Label 8250 3600 0    50   ~ 0
SDA
Text Label 8250 3700 0    50   ~ 0
SCL
Text Notes 8950 4050 0    79   ~ 0
Temp Sensor
Text Notes 8950 2350 0    79   ~ 0
Light Sensor
$Comp
L Connector:Conn_01x02_Male J4
U 1 1 5DA67096
P 1700 800
F 0 "J4" V 1762 844 50  0000 L CNN
F 1 "Conn_01x02_Male" V 1853 844 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical" H 1700 800 50  0001 C CNN
F 3 "~" H 1700 800 50  0001 C CNN
	1    1700 800 
	0    1    1    0   
$EndComp
Wire Wire Line
	1600 1350 1600 1000
Wire Wire Line
	1200 1350 1600 1350
Wire Wire Line
	1700 1000 1700 1350
Wire Wire Line
	1700 1350 2050 1350
Text Notes 1450 750  0    50   ~ 0
On Switch 
$Comp
L Switch:SW_Push SW2
U 1 1 5D9A7F57
P 5400 5350
F 0 "SW2" H 5400 5635 50  0000 C CNN
F 1 "KSA0A311 LFTR" H 5400 5544 50  0000 C CNN
F 2 "Button_Switch_THT:SW_Tactile_Straight_KSA0Axx1LFTR" H 5400 5550 50  0001 C CNN
F 3 "" H 5400 5550 50  0001 C CNN
	1    5400 5350
	1    0    0    -1  
$EndComp
$Comp
L Switch:SW_Push SW3
U 1 1 5D9A8159
P 6600 5350
F 0 "SW3" H 6600 5635 50  0000 C CNN
F 1 "KSA0A311 LFTR" H 6600 5544 50  0000 C CNN
F 2 "Button_Switch_THT:SW_Tactile_Straight_KSA0Axx1LFTR" H 6600 5550 50  0001 C CNN
F 3 "" H 6600 5550 50  0001 C CNN
	1    6600 5350
	1    0    0    -1  
$EndComp
Wire Wire Line
	5600 5350 5900 5350
Wire Wire Line
	4900 5350 5200 5350
Wire Wire Line
	6100 5350 6400 5350
Wire Wire Line
	6800 5350 7100 5350
Wire Wire Line
	800  7450 800  7600
$Comp
L Connector:Conn_01x03_Male J6
U 1 1 5DBAAF06
P 2900 7750
F 0 "J6" V 3053 7890 50  0000 L CNN
F 1 "Conn_01x03_Male" V 2962 7890 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 2900 7750 50  0001 C CNN
F 3 "~" H 2900 7750 50  0001 C CNN
	1    2900 7750
	0    1    -1   0   
$EndComp
Wire Wire Line
	3000 7550 3000 7450
Wire Wire Line
	3000 7450 3300 7450
Wire Wire Line
	3300 7450 3300 7200
Wire Wire Line
	3300 7200 3200 7200
Wire Wire Line
	2800 7550 2500 7550
Wire Wire Line
	2500 7550 2500 7100
Wire Wire Line
	2500 7100 2600 7100
Wire Wire Line
	2600 7300 2600 7450
Wire Wire Line
	2600 7450 2900 7450
Wire Wire Line
	2900 7450 2900 7550
$Comp
L Connector:Conn_01x02_Male J7
U 1 1 5D9EC8ED
P 3850 750
F 0 "J7" V 3912 794 50  0000 L CNN
F 1 "Conn_01x02_Male" V 4003 794 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical" H 3850 750 50  0001 C CNN
F 3 "~" H 3850 750 50  0001 C CNN
	1    3850 750 
	0    1    1    0   
$EndComp
Wire Wire Line
	3750 950  3750 1050
Wire Wire Line
	3850 1050 3850 950 
Wire Wire Line
	3750 1050 3800 1050
$Comp
L power:GND #PWR011
U 1 1 5D8C40B1
P 2650 1800
F 0 "#PWR011" H 2650 1550 50  0001 C CNN
F 1 "GND" H 2655 1627 50  0000 C CNN
F 2 "" H 2650 1800 50  0001 C CNN
F 3 "" H 2650 1800 50  0001 C CNN
	1    2650 1800
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR0106
U 1 1 5D9F52EC
P 3800 1100
F 0 "#PWR0106" H 3800 850 50  0001 C CNN
F 1 "GND" H 3805 927 50  0000 C CNN
F 2 "" H 3800 1100 50  0001 C CNN
F 3 "" H 3800 1100 50  0001 C CNN
	1    3800 1100
	1    0    0    -1  
$EndComp
Wire Wire Line
	3800 1100 3800 1050
Connection ~ 3800 1050
Wire Wire Line
	3800 1050 3850 1050
Wire Wire Line
	3450 3200 3850 3200
$Comp
L Amplifier_Operational:LM324 U1
U 3 1 5D8C5364
P 4150 3100
F 0 "U1" H 4150 3467 50  0000 C CNN
F 1 "LM324" H 4150 3376 50  0000 C CNN
F 2 "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm" H 4100 3200 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 4200 3300 50  0001 C CNN
F 4 "LM324DR" H 4150 3100 50  0001 C CNN "Part Number"
	3    4150 3100
	1    0    0    1   
$EndComp
Wire Wire Line
	3850 3000 3750 3000
Wire Wire Line
	3750 3000 3750 2800
Wire Wire Line
	3750 2800 4600 2800
Wire Wire Line
	4600 2800 4600 3100
Wire Wire Line
	4600 3100 4450 3100
Connection ~ 4600 3100
$Comp
L Device:R_Small_US R15
U 1 1 5DA13BF6
P 5200 3100
F 0 "R15" V 4995 3100 50  0000 C CNN
F 1 "16k" V 5086 3100 50  0000 C CNN
F 2 "Resistor_SMD:R_0805_2012Metric" H 5200 3100 50  0001 C CNN
F 3 "~" H 5200 3100 50  0001 C CNN
	1    5200 3100
	0    1    1    0   
$EndComp
$Comp
L Device:C_Small C23
U 1 1 5DA147CB
P 5450 3300
F 0 "C23" H 5550 3350 50  0000 L CNN
F 1 "1nF" H 5542 3255 50  0000 L CNN
F 2 "Capacitor_SMD:C_0805_2012Metric" H 5450 3300 50  0001 C CNN
F 3 "~" H 5450 3300 50  0001 C CNN
	1    5450 3300
	-1   0    0    1   
$EndComp
Wire Wire Line
	5300 3100 5450 3100
Wire Wire Line
	5450 3100 5450 3200
Wire Wire Line
	4600 3100 5100 3100
Text Notes 4900 2800 0    39   ~ 0
Anti Aliasing Filter, fo = 10khz
Wire Notes Line
	4850 3650 5900 3650
Wire Notes Line
	4850 2700 5900 2700
$Comp
L power:GND #PWR032
U 1 1 5DA25816
P 5450 3400
F 0 "#PWR032" H 5450 3150 50  0001 C CNN
F 1 "GND" H 5455 3227 50  0000 C CNN
F 2 "" H 5450 3400 50  0001 C CNN
F 3 "" H 5450 3400 50  0001 C CNN
	1    5450 3400
	1    0    0    -1  
$EndComp
Wire Wire Line
	5450 3100 6250 3100
Wire Wire Line
	6250 3100 6250 3600
Wire Wire Line
	6250 3600 6450 3600
Connection ~ 5450 3100
Wire Notes Line
	4850 2700 4850 3650
Wire Notes Line
	5900 2700 5900 3650
Text Notes 2200 5550 0    50   ~ 0
Comparator is to wake up the \nmicrocontroller during a loud\nevent. This will help with \npower savings
$EndSCHEMATC
