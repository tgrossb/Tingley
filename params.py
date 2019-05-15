import cv2

class FitParams:
	SPEED_UP = False
	def __init__(self, start = 0.01, stop = 1.5, step = 0.01, templateThreshold = 0.7, overlapThreshold = 0.5, fitFunction = cv2.TM_CCOEFF_NORMED):
		self.start = start
		self.stop = stop if not FitParams.SPEED_UP else 0.5
		self.step = step
		self.templateThreshold = templateThreshold
		self.overlapThreshold = overlapThreshold
		self.fitFunction = fitFunction

	def getTemplateFitParams(self):
		return (self.start, self.stop, self.step, self.templateThreshold, self.fitFunction)

	def __iter__(self):
		yield from [self.start, self.stop, self.templateThreshold, self.overlapThreshold, self.fitFunction]


class DrawParams:
#	def __init__(self, color = (128, 64, 64), textColor = (48, 155, 191), weight = 1):
	def __init__(self, color = (0, 0, 255), textColor = (255, 255, 255), weight = 1):
		self.color = color
		self.textColor = textColor
		self.weight = weight

	def __iter__(self):
		yield from [self.color, self.textColor, self.weight]
