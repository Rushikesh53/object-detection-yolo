import cv2
import matplotlib.pyplot as plt
# % matplotlib inline
# a="exp"+3
image = cv2.imread("yolov5/runs/detect/exp/bus.jpg")
height, width = image.shape[:2]
resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.axis("off")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.show()


# python detect.py --source data/images/img.png --weights yolov5s.pt --conf 0.4
