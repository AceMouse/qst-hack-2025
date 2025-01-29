This folder contains the data and scripts needed to build a classifier for photon-number resolving detection.

- **"PSNSPD_data.py" is the main script needed for generating training data.**

- **"QST-Hack-2025 - DTU Physics - Case - Photon Counting.pdf" contains the problem description incluing the target performance of the classifier**

The other files are there to make this script work. Understanding them is not necessary for building the classifier.
- The "8-pixel x photon.csv" files define single and double photon detections, which can be used as reference to ideal photon detection.
- "psnspd.py" defines a class for generating signals with noise from "noise.py" and randomised shifts for each photon. 

"PSNSPD_data.py" works by calling the class defined in "psnspd.py" and parsing an integer to the script returns the desired number of simulated detection signals, as a .csv file, as well as the number of photons corresponding to each signal. 

To use the script, start by installing the required packages in the directory.\
For Shell (on Linux): "python3 -m venv venv && pip3 install -r requirements.txt".

**To generate N signals, run the script like this: "python3 PSNSPD_data.py --signal_no N".\
If no input is given, the default is 100 signals.**

# 2025-01-29 update
The computational complexity of the detector algorithm, must be within the capabilities of the AMD ZU9EG fpga, which is capable of 630 GMAC (giga multiply-and-accumulate) operations, where 1 MAC is one 24-bit multiplication plus one 24-bit addition. The 630 GMAC is calculated as 2520 dsp-engines * 1 MAC/dsp-engine * 250 MHz = 630 GMAC.

In the final implementation the algorithm will run on a AMD ZU9EG fpga, which receives 12-bit samples from a 1 Gsamp/s ADC (analog-to-digital converter).
![PXL_20250129_084637736](https://github.com/user-attachments/assets/087d6a9b-52a3-4f43-9b78-37dbcd2a2aa9)

I (Axel) am located in room 251 (though often in room 249) building 307. Welcome to come by if you have questions.
