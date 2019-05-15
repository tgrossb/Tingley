#!/usr/bin/env python3

import sys
import cv2
import pickle
from rect import Rect
from params import FitParams, DrawParams
from note import Note

def main():
	# First thing is to read the source image and grayscale it
	pickleFile = sys.argv[1]

	with open(pickleFile, 'rb') as file:
		rawImg, scaleFactor, rangedNotes = pickle.load(file)

	if scaleFactor < 1.0:
		img = cv2.resize(rawImg, None, fx = scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_CUBIC)
	else:
		img = rawImg

	lniName = "Labeled notes"
	noteImg = img.copy()
	drawParams = DrawParams()
	for note in rangedNotes:
		rect = Rect(note.rect.x, note.rect.y, note.rect.w, note.rect.h, str(note.noteLine) + " " + str(note.duration))
		rect.draw(noteImg, drawParams, lniName, allInfo = False)
	cv2.namedWindow(lniName)
	cv2.setMouseCallback(lniName, Rect.onClick, lniName)
	cv2.imshow(lniName, noteImg)

	print("\t\tPress space to continue...")
	while True:
		if cv2.waitKey(30) == 32:
			break

	cv2.destroyAllWindows()




if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print()
		sys.exit(0)
