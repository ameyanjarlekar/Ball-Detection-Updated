## Ball Detection code ##

* All the required code along with weights for supervised detection are in [src](./src)
* The launch files for ROS are in [launch](./launch)

### Work points

- [x] Capture stream reliably
- [x] Calculate mask on images for initial detection of ball, ensuring that if ball is present it isn't missed, even if garbage is also detected
- [x] Calculate windows around every contour in mask to get multiple windows
- [x] Get weights for supervised code from URC image database
- [x] Feed windows to supervised code to get the window with best probability of ball
- [ ] Get and synchronise the direction data with images being computed, while adjusting rover direction
- [ ] Specify what constitutes the label of images (direction data and window position)
- [ ] Use the window with best probability of ball to adjust rover motion

### Unsupervised

* Images are captured through a USB cam as live feed
* Mask is calculated based on the pre-set HSV values for the ball
* Multiple rectangular windows are calculated around each contour of the mask
* Numpy array of the windows is fed to the supervised code, which returns the window with max probability of ball
* Finally we have the label of the window with max probability of ball, for each frame captured
* What constitutes this label and how to use this data in the final function(s) is yet to be finalized

### Supervised

For a direct access :
1) To get output of direct livestream images from the video of UBS_cam use the code "turn_usb", you need the supporting codes "utils.py", "basictest.py" , "detect_ball.py" and the file "new_model_num_coluoured.h5" for executing the code.The input is consistent with the rover having a 360 degree turn with images taken at particular angle intervals with the output being the angle at which the probability of the ball being placed is the maximum.
2) The keras model, given the input images are in greyscale format are stored in  	new_model_num3_greyoutput.h5 while for coloured
are stored in  	new_model_num_coluoured.h5. 
3) Start the USB_cam on your laptop by first starting the roscore service in your corresponding workspace (will mostly be Rover_basestation) . Then run the code "turn_usb".
 
 Detailed explaination of the code :
1) Given a folder, the images could be converted into corresponding RGB, greyscale values through basic openCV functions in roverrover.py
2) These are then fed into a CNN network developed using keras whose code is found in "try.py".
3) The CNN Network was trained using a dataset available of a tennis match with around 0.3M examples with hyperparameters of the   CNN being tuned using the paper https://github.com/ameyanjarlekar/MRT/blob/master/Reno_Convolutional_Neural_Networks_CVPR_2018_paper.pdf provided in the repo . 
4) The network was further retrained using a 16,000 odd augmented dataset collected from the image frames of the past videos available of the competition with just the feedforward network weights modified during this process.
 
 Working implementation of the code :
1) The plan is to make the rover collect images during a 360 degree rotation of itself. This image data is then fed into an    unsupervised algorithm (https://github.com/iitbmartian/Autonomous_Subsystem/tree/master/Ball_Detection/Unsupervised/src) which primarily detects green pastures and creates cropped image windows over these pastures.
2) The files "utils.py","detect_ball.py" create cropped images from the original image at places where a green pasture is observed and convert them into corresponding RGB/greyscale numpy arrays of size 50*50.
3) These images are then sent to the supervised algo which provides the image with the max probability of getting a ball.
4) All this is integrated in a single code "turn.py" which outputs the corresponding value of degree with the maximum probability along with the maximum probability.
5) After the desired angle is detected, the rover moves in a forward direction and finds out if it needs to deviate in its path. During the forward movement, the rover detects a single image and using functions in "detect_ball.py" it finds out possible multiwindows which contain green to yellow pastures. Then, using "forward_usb.py" it detects the multiwindow having maximum probability of getting the ball and thus the required deviation angle is found out.

### Requirements

* ROS kinetic
* USB cam (the port for the USB cam is currently hardcoded in the code, 'resource' variable, and will need to be changed
* URC like lighting and background conditions

### Running

After setting up the code run in the workspace root -
```
source devel/setup.bash
roslaunch Ball_Detection turn_usb.launch
(or)
roslaunch Ball_Detection forward_usb.launch
```
Don't forget to first, in workspace root -
```
chmod chmod +x -R .
```

### Contribution ###

Feel free to contribute to this project by either raising issues or handing in pull requests.
