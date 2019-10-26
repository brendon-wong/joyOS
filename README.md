# emotionai
Neural networks for face and emotion detection

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
