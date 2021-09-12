from cv2 import putText, VideoCapture, FONT_HERSHEY_PLAIN, rectangle, imshow, waitKey, destroyAllWindows,resize
from cv2.dnn import readNet, blobFromImage, NMSBoxes
from numpy import random, argmax, load, full, concatenate, clip, newaxis
import cv2
import argparse
import glob

# Load Yolo
net = readNet("yolov3.weights", "yolov3.cfg")
with open("coco", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = random.uniform(0, 255, size=(len(classes), 3))

#calling the caffe framework model for coloring the image
#colorization.prototxt file contains the configurations of all layers in a txt format, uses Google's protocol bufffer thus named ass prototxt.
#colorization.caffeemodel is trained configurations parameters of the Blob array of every combination/network.
color_net = cv2.dnn.readNetFromCaffe("colorization.prototxt", "colorization.caffemodel") 
pts = load("points.npy") #this file containes the values in numpy format.
# add the cluster centers as 1x1 convolutions to the model
class8 = color_net.getLayerId("class8_ab")
conv8 = color_net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
color_net.getLayer(class8).blobs = [pts.astype("float32")]
color_net.getLayer(conv8).blobs = [full([1, 313], 2.606, dtype="float32")]

# load the input video from disk, scale the pixel intensities to the
# range [0, 1], and then convert the image from the BGR to Lab color
# space
vs = cv2.VideoCapture('small.mpeg')
fps = 0
img_array=[]
while True:
    while True:
        _, frame = vs.read()
        fps += 1
        if (fps < 19):
            continue
        else:
            fps = 0
            break
    #image = cv2.imread('')
    scaled = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and then
    # perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 49
    color_net.setInput(cv2.dnn.blobFromImage(L))
    ab = color_net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (416, 416))

    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    L = cv2.split(lab)[0]
    L = cv2.resize(L, (416, 416))
    colorized = concatenate((L[:, :, newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = clip(colorized, 0, 1)

    #the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned
    # 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")

    #YOLO started
    height, width, channels = colorized.shape
    # Detecting objects`
    blob = blobFromImage(colorized, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id =argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = NMSBoxes(boxes, confidences, 0, 0)
    ln = len(boxes)
    for i in range(ln):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            font = FONT_HERSHEY_PLAIN
            rectangle(colorized, (x, y), (x + w, y + h), color, 2)
            putText(colorized, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 1)
    imshow("image",colorized)
    img_array.append(colorized)
    if waitKey(1) & 0xFF == ord('q'):
        break

#5 can be changed according to frame rate of the input video
out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5,(416,416))
for i in range(len(img_array)):
    out.write(img_array[i])
print(len(img_array))
out.release()
vs.release()
destroyAllWindows()




    
