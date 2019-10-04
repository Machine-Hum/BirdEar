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
F 2 "" V 850 3800 50  0001 C CNN
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
F 2 "" V 1200 1660 50  0001 C CNN
F 3 "~" V 1200 1660 50  0001 C CNN
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
$EndSheet
$Comp
L Regulator_Linear:AZ1117-3.3 U2
U 1 1 5D8C3D0D
P 2650 1350
F 0 "U2" H 2650 1592 50  0000 C CNN
F 1 "AZ1117EH-3.3TRG1" H 2650 1501 50  0000 C CNN
F 2 "" H 2650 1600 50  0001 C CIN
F 3 "https://www.diodes.com/assets/Datasheets/AZ1117.pdf" H 2650 1350 50  0001 C CNN
	1    2650 1350
	1    0    0    -1  
$EndComp
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
L Device:C_Small C3
U 1 1 5D8C4221
P 2050 1550
F 0 "C3" H 2150 1600 50  0000 L CNN
F 1 "1uF" H 2142 1505 50  0000 L CNN
F 2 "" H 2050 1550 50  0001 C CNN
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
F 2 "" H 3250 1550 50  0001 C CNN
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
Wire Wire Line
	1200 1350 2050 1350
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
F 2 "" H 2800 3300 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 2900 3400 50  0001 C CNN
	1    2850 3200
	1    0    0    1   
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 2 1 5D8C52C4
P 2950 4950
F 0 "U1" H 2950 5317 50  0000 C CNN
F 1 "LM324" H 2950 5226 50  0000 C CNN
F 2 "" H 2900 5050 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 3000 5150 50  0001 C CNN
	2    2950 4950
	1    0    0    -1  
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 3 1 5D8C5364
P 2950 6250
F 0 "U1" H 2950 6617 50  0000 C CNN
F 1 "LM324" H 2950 6526 50  0000 C CNN
F 2 "" H 2900 6350 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 3000 6450 50  0001 C CNN
	3    2950 6250
	1    0    0    -1  
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 4 1 5D8C5409
P 2900 7050
F 0 "U1" H 2900 7417 50  0000 C CNN
F 1 "LM324" H 2900 7326 50  0000 C CNN
F 2 "" H 2850 7150 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 2950 7250 50  0001 C CNN
	4    2900 7050
	1    0    0    -1  
$EndComp
$Comp
L Amplifier_Operational:LM324 U1
U 5 1 5D8C54A9
P 950 5400
F 0 "U1" H 550 5450 50  0000 L CNN
F 1 "LM324" H 550 5350 50  0000 L CNN
F 2 "" H 900 5500 50  0001 C CNN
F 3 "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf" H 1000 5600 50  0001 C CNN
	5    950  5400
	1    0    0    -1  
$EndComp
Wire Wire Line
	850  5700 850  5850
$Comp
L power:GND #PWR03
U 1 1 5D8C59B1
P 850 5850
F 0 "#PWR03" H 850 5600 50  0001 C CNN
F 1 "GND" H 855 5677 50  0000 C CNN
F 2 "" H 850 5850 50  0001 C CNN
F 3 "" H 850 5850 50  0001 C CNN
	1    850  5850
	1    0    0    -1  
$EndComp
$Comp
L power:+3.3V #PWR02
U 1 1 5D8C5AA2
P 850 4950
F 0 "#PWR02" H 850 4800 50  0001 C CNN
F 1 "+3.3V" H 865 5123 50  0000 C CNN
F 2 "" H 850 4950 50  0001 C CNN
F 3 "" H 850 4950 50  0001 C CNN
	1    850  4950
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R1
U 1 1 5D8C65A2
P 1100 3100
F 0 "R1" H 1168 3146 50  0000 L CNN
F 1 "2.2k" H 1168 3055 50  0000 L CNN
F 2 "" H 1100 3100 50  0001 C CNN
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
F 2 "" H 1300 3300 50  0001 C CNN
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
F 2 "" H 1850 3050 50  0001 C CNN
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
F 2 "" H 1850 3550 50  0001 C CNN
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
P 1150 5150
F 0 "C1" H 1250 5200 50  0000 L CNN
F 1 "1uF" H 1242 5105 50  0000 L CNN
F 2 "" H 1150 5150 50  0001 C CNN
F 3 "~" H 1150 5150 50  0001 C CNN
	1    1150 5150
	1    0    0    -1  
$EndComp
Wire Wire Line
	1150 5050 1150 5000
Wire Wire Line
	1150 5000 850  5000
Wire Wire Line
	850  5000 850  5100
Connection ~ 850  5000
Wire Wire Line
	850  5000 850  4950
$Comp
L power:GND #PWR05
U 1 1 5D8CC801
P 1150 5250
F 0 "#PWR05" H 1150 5000 50  0001 C CNN
F 1 "GND" H 1155 5077 50  0000 C CNN
F 2 "" H 1150 5250 50  0001 C CNN
F 3 "" H 1150 5250 50  0001 C CNN
	1    1150 5250
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R4
U 1 1 5D8CCF3A
P 2300 2500
F 0 "R4" V 2095 2500 50  0000 C CNN
F 1 "100k" V 2186 2500 50  0000 C CNN
F 2 "" H 2300 2500 50  0001 C CNN
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
Wire Wire Line
	3450 3200 3600 3200
Connection ~ 3450 3200
$Comp
L Device:R_POT_TRIM_US RV2
U 1 1 5D8D08F7
P 3000 2300
F 0 "RV2" V 3050 2450 50  0000 C CNN
F 1 "1M" V 2900 2300 50  0000 C CNN
F 2 "" H 3000 2300 50  0001 C CNN
F 3 "~" H 3000 2300 50  0001 C CNN
	1    3000 2300
	0    1    1    0   
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
Text GLabel 3850 3200 2    50   Output ~ 0
Audio
Text GLabel 3700 4950 2    50   Output ~ 0
Loud
$Comp
L Device:R_POT_TRIM_US RV1
U 1 1 5D8D3430
P 2050 5050
F 0 "RV1" V 2100 5200 50  0000 C CNN
F 1 "1M" V 1950 5050 50  0000 C CNN
F 2 "" H 2050 5050 50  0001 C CNN
F 3 "~" H 2050 5050 50  0001 C CNN
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
Text Label 3600 3200 0    50   ~ 0
Audio
Text Label 2300 4700 0    50   ~ 0
Audio
Wire Wire Line
	2300 4700 2300 4850
Wire Wire Line
	2300 4850 2650 4850
Wire Wire Line
	3250 4950 3600 4950
Text Notes 2200 5350 0    50   ~ 0
Comparator is to wake up the microcontroller.\nThis will help with power savings
Text Notes 3050 6600 0    50   ~ 0
Spares
Wire Wire Line
	3600 4950 3600 3700
Wire Wire Line
	3600 3700 6450 3700
Connection ~ 3600 4950
Wire Wire Line
	3600 4950 3700 4950
Wire Wire Line
	3600 3200 3600 3600
Wire Wire Line
	3600 3600 6450 3600
Connection ~ 3600 3200
Wire Wire Line
	3600 3200 3850 3200
Text Notes 6750 4450 0    197  ~ 39
STM32
$EndSCHEMATC
