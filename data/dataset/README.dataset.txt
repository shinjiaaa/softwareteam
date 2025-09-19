# Drone Crash Avoidance > v7
https://universe.roboflow.com/tylervisimoai/drone-crash-avoidance

Provided by a Roboflow user
License: CC BY 4.0

# Project Overview

## Project Goal
The goal of this project is to train a drone in the ability to sense and avoid nearby hazardous objects. The images being labelled within this project are being used to train a YOLO model to detect nearby objects that we most likely want to avoid. 

## Labelling Instructions
First of all, you don't have to label each and every image. The images are taken from various drone crash compilations on YouTube. The videos were split using timestamps and thus, they may not always be frame perfect, so feel free to cut out the first or last frame(s) if they do not match the rest of the video. Additonally, the footage isn't always fantastic, so if an image is blurred or distorted beyond reason (perhaps the frame where the drone actually hits something and loses control), just mark it as null and leave it be.

It may occasionally be helpful to watch the video corresponding to a set of images. The images are named by the title of YouTube video and then the timestamp (start and end in seconds) where they were taken from. The YouTube videos will be listed below.

When labelling, focus on nearby, hazardous objects the drone may crash into, such as trees, street lamps, telephone wires, or other drones. Additionally, label any object that the drone appears to be tracking, such as a moving vehicle or person. Try not to get too caught up in objects that are in the background.

For now, we'd like to keep the list of classes relatively small, since the main goal is not really to classify well, just to detect. Ultimately, as long as the drone senses a hazard within a reasonable distance (as in, it has the proper amount of time to avoid the hazard), the actual classification doesn't really matter.

### Current Class List
1. Tree
2. Pole
3. Wire
4. Vehicle
5. Person
6. Drone
7. Building
8. Ground (used for crashing into a dirt hill, for example)

## Data Sources Used
Links to Drone Crash Compilations
(2) Extreme Drone Crashes - Compilation 2015 - YouTube  
(2) Drone Fail 2019 Compilation, Mavic Pro, Inspire 2, Parrot Anafi, Phantom 4 - YouTube 