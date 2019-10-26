# emotionai
Neural networks for face and emotion detection

## To set up the application:
1. `git clone` this repository
2. `python3 -m venv venv` to create a new virtual environment
3. `. venv/bin/activate` to activate the virtual environment (deactivate the virtual environment with `deactivate`)
4. `pip3 install -r requirements.txt` to install/update all requirements

## To run the application:
1. `. venv/bin/activate` to activate the virtual environment if not already activated
2. `python3 detect2.py` to run the app

# Installation Process
# Prerequitistes
Must have the following running on your machine:<br>
python, brew, pip, 
# Installation
Activate virtual environment
Use the package manager pip to install the following necessary python libraries
pip install numpy scipy matplotlib scikit-image scikit-learn ipython pandas
 or save the file requirments.txt and run
<br>**pip install -r requirements.txt**

Exiting out of virtual environment

Install Command line Tools<br>
**sudo xcode-select --install**
<br>

If you run into the error, <br>
**xcode-select: error: command line tools are already installed, use "Software Update" to install updates**
<br>

Run the following two lines:<br>
**sudo rm -rf /Library/Developer/CommandLineTools<br>
xcode-select --install**
<br><br>Once Apple Command Line Tools are installed, install opencv 
<br> **brew install opencv**
<br>Run **brew cellar** and you should see opencv as one of the opencv library listed

Check if your opencv installation worked
Run python
- import cv2 <br>
- cv2.__version__<br>

If installing Opencv through brew doesnt work, attempt running<br>
***pip2 install opencv-python***


# Usage
