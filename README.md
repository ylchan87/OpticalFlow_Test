# About

Sample to show optical flow estimation using various algos available in OpenCV(4.5.2):

* "DIS" at working point medium, fast, ultrafast
* "Farneback"
* "DenseRLOF"
* "DualTVL1"
* "PCAFlow"
* "DeepFlow"

Tested on OpenCV4.5.2 and python 3.7

Modified from:
official openCV samples
https://github.com/opencv/opencv/blob/master/samples/python/dis_opt_flow.py

# Usage

`python opt_flow_test.py [<video_source>] [<mode>]`

where

* `<video_source>` : filename, or 0 for Camera0, 1 for Camera1 etc.  (default:0)
* `<mode>`        : 'testAll' to run all algo and save (default:'interact')

With 'interactive' mode, controls at pop up wndow are:
* t   - toggle temporal propagation of flow vectors
* 1-8 - switch to use different algo
* ESC - exit

# Examples

Run and interact with camera0 (usually the webcam at notebook)

`python opt_flow_test.py`

Use a file as source and interact

`python opt_flow_test.py input.mp4`

Run all algo on the file and save result as video

`python opt_flow_test.py input.mp4 testAll`

# Demo Output

Test on a linux notebook with i7-8550U (a 2017 CPU for non-gaming notebook), input resolution 690x540

Result approx. FPS:
| Algo | FPS |
| ---- | ----|
|"DIS_Medium"   |  30|
|"DIS_Fast"     | 120|
|"DIS_UltraFast"| 200|
|"Farneback"    |   4.5|
|"DenseRLOF"    |   2|
|"DualTVL1"     |   0.2|
|"PCAFlow"      |  10|
|"DeepFlow"     |   1|

Result video:
https://www.youtube.com/watch?v=ytqnOpcZFek
