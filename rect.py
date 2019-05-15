import cv2
import math
from params import DrawParams

class Rect:
	allRects = []
	currentLabeled = None

	def __init__(self, x, y, w, h, label = None):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.label = label if label is not None else "-"
		self.middle = (self.x + self.w/2, self.y + self.h/2)
		self.area = self.w * self.h
		self.window = None

	def overlap(self, other):
		ow = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
		oh = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
		oa = ow * oh
		return oa / self.area

	def distance(self, other):
		dx = self.middle[0] - other.middle[0]
		dy = self.middle[1] - other.middle[1]
		return math.sqrt(dx*dx + dy*dy)

	def merge(self, other):
		x = min(self.x, other.x)
		y = min(self.y, other.y)
		w = max(self.x + self.w, other.x + other.w) - x
		h = max(self.y + self.h, other.y + other.h) - y
		label = self.label + " " + other.label
		return Rect(x, y, w, h, label)

	def draw(self, img, drawParams, windowName, allInfo = True):
		color, textColor, weight = drawParams

		end = (self.x + int(self.w), self.y + int(self.h))
		cv2.rectangle(img, (self.x, self.y), end, color, weight)

		if self.window is None or not self.window == windowName:
			self.window = windowName
			Rect.allRects.append((self, img, color, textColor, windowName, allInfo))


	def drawLabel(self, img, color, textColor, allInfo, font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 1):
		if self.label is not None:
			if allInfo:
				sub = "(" + str(self.x) + ", " + str(self.y) + ") + <" + str(self.w) + ", " + str(self.h) + "> => (" + str(self.middle[0]) + ", " + str(self.middle[1]) + ")"
				subWidth, subHeight = cv2.getTextSize(sub, font, fontScale = fontScale, thickness = thickness)[0]
				end = (self.x + subWidth + 4, self.y - subHeight - 4)
				cv2.rectangle(img, (self.x, self.y), end, color, cv2.FILLED)
				cv2.putText(img, sub, (self.x + 2, self.y - 2), font, fontScale, textColor, thickness)

			labelWidth, labelHeight = cv2.getTextSize(self.label, font, fontScale = fontScale, thickness = thickness)[0]
			end = (self.x + labelWidth + 4, self.y - (subHeight if allInfo else -4) - labelHeight - 8)
			cv2.rectangle(img, (self.x, self.y - (subHeight if allInfo else -4) - 4), end, color, cv2.FILLED)
			cv2.putText(img, self.label, (self.x + 2, self.y - (subHeight if allInfo else -4) - 6), font, fontScale, textColor, thickness)



	def __eq__(self, other):
		return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h and self.label == other.label


	@staticmethod
	def onClick(event, x, y, flags, clickedWindow):
#		print("Click on " + clickedWindow)
		if not event == cv2.EVENT_LBUTTONDOWN:
			return
		if Rect.currentLabeled is not None:
			img, windowName = Rect.currentLabeled
			cv2.imshow(windowName, img)
#		print("Clicked on window '" + clickedWindow + "' (len: " + str(len(Rect.allRects)) + ")")
		for rect, img, color, textColor, windowName, allInfo in Rect.allRects:
			if windowName == clickedWindow:
#				print("Correct window (" + clickedWindow + ")")
				if rect.x < x and (rect.x + rect.w) > x and rect.y < y and (rect.y + rect.h) > y:
#					print("Rectangle click")
					labeledImg = img.copy()
					rect.drawLabel(labeledImg, color, textColor, allInfo)
					cv2.imshow(clickedWindow, labeledImg)
					Rect.currentLabeled = (img, windowName)
