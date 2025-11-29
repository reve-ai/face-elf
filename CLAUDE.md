This repository implements a scientific project focusing on real-time face detection and identification.

The project will include several pieces:

In the first phase:

1. It is a python project, running on an NVIDIA DGX Spark system (GB10 GPU) with CUDA available
2. It opens the first available webcam and reads it for input images
3. It runs a nvidia face detection model to find the center and bounding box of all faces in the image
4. It displays the captured image together with a bounding box around each face in real time

In the second phase:

5. It allows entering a mode where it detects a central face in the image
6. It allows capturing this face, more than once, into a sequence of captured snapshots shown across the side of the screen
7. Each captured face image is cropped to only the bounding box, and scaled to a normalized size of 384 pixels along the longest axis
8. It allows saving all the captured images in a directory called "known-faces/<some-name>" where the user gives the some-name
9. It then allows exiting back to the real-time face detection-and-display mode

In the third phase:

10. It calculates a facial embedding for each subdirectory of images in known-faces/
11. For each face it finds in the real-time video image stream, it normalizes the face in the box to the same size as above, calculates the embedding, and searches whether there is any face that matches the embedding close enough. If so, the best match is mapped to that box in the display, and the "<some-name>" of that directory is displayed as annotation text over the face in the image.
12. If it finds a face it does not recognize, it saves a cropped, scaled box of that face into the directory new-faces/(timestamp) where (timestamp) is of the form YYYY-MM-DD-HH-MM, and each face is captured at most once every second.

