from rect import Rect

class Note:
	# The line is in terms of half lines off of the base line
	# For example, an open D (D3) would be 0, E3 => 1, E2 => -6
	def __init__(self, noteLine, duration, sharp = False, flat = False, rect = None):
		# Use the line to get the number of whole steps from A4
		# From the (l = -1, h = -21), as l is incremented, increment h in the pattern [2, 2, 1, 2, 2, 2, 1]

		#               A  0
		#             -G-  2
		#             F    4
		#           -E-    5
		#           D      7
		#         -C-      9
		#         B       10
		# -------A--------12-----
		#       G         14
		# -----F----------16-----
		#     E           17
		# ---D------------19-----
		#   C             21
		# -B--------------22---
		# A               24
		# ---------------------
		#
		# ---------------------
		#

		halfSubPattern = [2, 2, 1, 2, 2, 2, 1]
		patternLoc = 0

		# Use the definition of -8 note lines equals 24
		# For each line above -8, add by the halfSubPattern
		self.midi = 24
		lineRoller = -8
		while lineRoller < noteLine:
			lineRoller += 1
			self.midi += halfSubPattern[patternLoc]
			patternLoc = (patternLoc + 1) % 7
		if sharp:
			self.midi += 1
		if flat:
			self.midi -= 1

		patternLoc = 0
		lineRoller = -1
		halfOffset = -21
		if noteLine < -1:
			while lineRoller > noteLine:
				halfOffset -= halfSubPattern[6-patternLoc]
				patternLoc = (patternLoc + 1) % 7
				lineRoller -= 1
		elif noteLine > -1:
			while lineRoller < noteLine:
				halfOffset += halfSubPattern[patternLoc]
				patternLoc = (patternLoc + 1) % 7
				lineRoller += 1

		# If there is a sharp or flat increment or decrement (respectively) the halfOffset variable by 1
		if sharp:
			halfOffset += 1
		if flat:
			halfOffset -= 1

		# Now that we have the half steps from A4, the frequency of the note is 440Hz * (2^(1/12))^halfOffset
		self.pitch = 440.0 * ((2**(1/12.0)) ** halfOffset)
		self.noteLine = noteLine
		self.duration = duration
		self.rect = rect

	@staticmethod
	def calculateNoteLine(noteRect, staffCenterY, staffSep):
		# Base the calculation off of the relative center positions
		# Essentially, figure out how many note boxes the center of the note is from the center of the staff
		# Ex. if (staffCenter - noteCenter) = 2.5 * noteHeight, the note is 2.5 full lines up, of 5 note lines up

#		print("Center: " + str(staffCenterY) + "  h:" + str(noteRect.h))
#		for c in range(0, 10):
#			min = (staffCenterY - c*noteRect.h/2.0) - (noteRect.h/4.0)
#			max = (staffCenterY - c*noteRect.h/2.0) + (noteRect.h/4.0)
#			print("Bucket %d: %f -> %f" % (c, max, min))
#		print()

		noteCenter = noteRect.middle[1]
		return int(round(float(staffCenterY - noteCenter)/(staffSep/2.0)))
