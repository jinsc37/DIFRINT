import cv2
import os

def frame2vid(src, vidDir):
	images = [img for img in os.listdir(src) if img.endswith(".png")]
	images.sort()
	frame = cv2.imread(os.path.join(src, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(vidDir, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

	for image in images:
		video.write(cv2.imread(os.path.join(src, image)))

	cv2.destroyAllWindows()
	video.release()

if __name__ == '__main__':
	frame2vid(src='./output/OurStabReg/01/', vidDir='./output/OurStabReg/01.avi')