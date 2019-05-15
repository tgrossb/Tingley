#!/usr/bin/env python3

from screeninfo import get_monitors
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Queue, Manager, Process
import heapq
import pickle
from rect import Rect
from params import FitParams, DrawParams
from plotter import Plotter
from note import Note

# The file can be assumed to be in ./templates for all template files
staffTmpFiles = ["staff1.png", "staff2.png"] #, "staff5.png"]
sharpTmpFiles = ["sharp.png"]
flatTmpFiles = ["flatLine.png", "flatSpace.png"]
quarterTmpFiles = ["quarterSpace1.png", "quarterSpace2.png", "quarterLine1.png", "quarterLine2.png"]
halfTmpFiles = ["halfSpace1.png", "halfSpace2.png", "halfSpace3.png", "halfLine1.png", "halfLine2.png"]
wholeTmpFiles = ["wholeSpace.png", "wholeNoteLine.png", "wholeLine.png", "wholeNoteSpace.png"]

staffTmps = [cv2.imread("templates/" + fileName, 0) for fileName in staffTmpFiles]
sharpTmps = [cv2.imread("templates/" + fileName, 0) for fileName in sharpTmpFiles]
flatTmps = [cv2.imread("templates/" + fileName, 0) for fileName in flatTmpFiles]
quarterTmps = [cv2.imread("templates/" + fileName, 0) for fileName in quarterTmpFiles]
halfTmps = [cv2.imread("templates/" + fileName, 0) for fileName in halfTmpFiles]
wholeTmps = [cv2.imread("templates/" + fileName, 0) for fileName in wholeTmpFiles]

# Trackbar names
START_NAME = "Scale start (%)"
STOP_NAME = "Scale stop (%)"
STEP_NAME = "Scale step (%)"
TMP_THRESH_NAME = "Template threshold (%)"
BOX_THRESH_NAME = "Overlap threshold (/100)"

# (start, stop, step, templateThreshold, mergeThreshold, fitFunction
staffFitParams = FitParams(start = 0.3, templateThreshold = 0.75, overlapThreshold = 0.01)
sharpFitParams = FitParams(start = 0.3, templateThreshold = 0.75, overlapThreshold = 0.5)
flatFitParams = FitParams(start = 0.3, templateThreshold = 0.75, overlapThreshold = 0.5)
quarterFitParams = FitParams(start = 0.3, templateThreshold = 0.7, overlapThreshold = 0.5)
halfFitParams = FitParams(start = 0.3, templateThreshold = 0.7, overlapThreshold = 0.5)
wholeFitParams = FitParams(start = 0.3, templateThreshold = 0.7, overlapThreshold = 0.5)


# Applies a list of templates to an image at various scales
# Finds the optimal scale that maximizes the number of hits across the image
# The start and step can be given, but both will be minimized by default
# Additionally, allows for a message to be printed and updated for progress
def matchTemplate(grayscale, templates, templateFitParams = None, msg = None, plotter = None):
	if templateFitParams is None:
		templateFitParams = FitParams()

	start, stop, step, templateThreshold, fitFunction = templateFitParams.getTemplateFitParams()

	# Figure out how many scales will be checked
	scales = np.arange(start, stop, step)
	done = 0
	total = len(scales) * len(templates)
	printProgress(msg, done, total)

	if plotter is not None:
		subplot, dataQ = plotter


	x, y = [], []
	bestScale = 1.0
	bestHits = []
	bestHitCount = -1
	for scale in scales:
		hitScore = 0
		hitCount = 0
		hits = []
		for template in templates:
			try:
				template = cv2.resize(template, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
			except Exception as e:
				# This is fairly expected
				# If the size of the image is too small, resize will throw an exception
				done += 1
				printProgress(msg, done, total, post = "Scale: %fx    Hits: %d    Max hits: %d @ %fx" % (scale, -1, bestHitCount, bestScale))
				continue

			# Vertify that the template isnt too big for the image
			if template.shape[0] >= grayscale.shape[0] or template.shape[1] >= grayscale.shape[1]:
				done += 1
				printProgress(msg, done, total, post = "Scale: %fx    Hits: %d    Max hits: %d @ %fx" % (scale, -1, bestHitCount, bestScale))
				continue

			res = cv2.matchTemplate(grayscale, template, fitFunction)
			matches = np.where(res >= templateThreshold)
			hitVals = []
			for c in range(len(matches[0])):
				hitVals.append(res[matches[0][c]][matches[1][c]])
			hitCount += len(hitVals)
			hits += [(hitVals, matches)]

			done += 1
			printProgress(msg, done, total, post = "Scale: %fx    Hits: %d    Max hits: %d @ %fx" % (scale, hitCount, bestHitCount, bestScale))

		x.append(hitCount)
		y.append(scale)
		if plotter is not None:
			dataQ.put((subplot, Plotter.DATA, (y, x)))

		if hitCount > bestHitCount:
			bestHitCount = hitCount
			bestHitScore = hitScore
			bestHits = hits
			bestScale = scale
			if plotter is not None:
				dataQ.put((subplot, Plotter.Y_MAX, bestHitCount+1))

	printProgress(msg, total, total, post = "Max hits: %d @ %fx" % (bestHitCount, bestScale), end = '\n')
	return bestScale, bestHits


# Converts the hit locations of the templates into bounding boxes
# Bases the size of these bounding boxes of the template used to generate the hit
def getTemplateMatches(grayscale, templates, templateFitParams = None, templateMsg = None, locationMsg = "Calculating bounding boxes...", plotter = None):
	scale, locations = matchTemplate(grayscale, templates, templateFitParams = templateFitParams, msg = templateMsg, plotter = plotter)

	if templateMsg is None:
		locationMsg = None

	imgLocations = []
	tmpCount = len(templates)
	printProgress(locationMsg, 0, tmpCount)
	for c in range(tmpCount):
		h, w = templates[c].shape
		w *= scale
		h *= scale
		hitVals, hits = locations[c]
		for c in range(len(hitVals)):
			imgLocations.append(Rect(hits[1][c], hits[0][c], w, h, str(hitVals[c])))
		# No point in printing the progress in each subloop as this would actually just
		# add time with how fast the subloop will run
		printProgress(locationMsg, c+1, tmpCount)
	printProgress(locationMsg, tmpCount, tmpCount, end = '\n')
	return imgLocations, scale


# Recursively expands a single rect to as large as it can be, limited by the overlap threshold
def mergeSingleRect(rect, rects, threshold):
	merged = False

	# Iterate backwards to avoid having to handle changing indices
	for c in range(len(rects)-1, -1, -1):
		# If the overlap is significant enough, merge the two rects
		if rect.overlap(rects[c]) > threshold or rects[c].overlap(rect) > threshold:
			rect = rect.merge(rects.pop(c))

			# Instead of going full recursive and checking the new rect against all
			# other again, let this loop finish and check with the new rect at the end
			# Worst case scenario, this costs a few iterations, but best case it saves
			# a lot of iterations
			merged = True

		# If we have reached a rectangle that is far enough from the focus rectange to have no overlap
		# then we can break out of the loop - it aint happenin
		elif rects[c].distance(rect) > rect.w/2 + rects[c].w/2:
			break

	# If we still haven't gotten a merge, we've gone as far as we can with this rectangle (base case)
	if not merged:
		return rect

	# Otherwise, we have a shot at going back through with the new and improved rectangle
	# Try that out recursively
	return mergeSingleRect(rect, rects, threshold)


# Expands all rects to as large as they can be, minimizing the rect count, limited by the overlap threshold
def mergeRects(rects, threshold, msg = None):
	if msg is not None:
		print(msg, end = '')
	initSize = len(rects)

	mergedRects = []

	while len(rects) > 0:
		# Get the first rect and sort all others by their distance from this rect
		# The list must be reversed because it must be iterated over backwards for indexing reasons
		rect = rects.pop(0)
		rects.sort(key = lambda r: r.distance(rect), reverse = True)

		# Now, recursively merge the focus rect with those close by
		# Each time the rect merges with another, it gets larger so all previously failed
		# merge checks must be done again
		bigRect = mergeSingleRect(rect, rects, threshold)
		mergedRects.append(bigRect)

	diff = initSize - len(mergedRects)
	reductionPercent = 0 if initSize == 0 else (100 * float(diff)/initSize)
	if msg is not None:
		print("  Finished - Reduced template match count by %d%% (%d -> %d, down by %d matches)" % (round(reductionPercent), initSize, len(mergedRects), diff))

	return mergedRects


def printProgress(msg, done, total, post = "", end = ''):
	if msg is not None:
		print("\r\033[K%s %d%% (%d/%d)    %s" % (msg, round(100*float(done)/total), done, total, post), end = end)


# Removes rectangles that don't have enough rectangles in its row
# Computes the number of rectangles per row and removes all rectangles
# from rows that are outliers to this count (<1.5 standard deviations from the mean)
def cleanUpLonesomeRows(staffs, outlierMultiplier = 1.5, log = False):
	if log:
		print("Cleaning up template matches using outlier removal...", end = '')

	rows = [rect.y for rect in staffs]
	rowCounts = np.array([0 for c in range(max(rows)+1)])
	for row in rows:
		rowCounts[row] += 1
	q2 = np.percentile(rowCounts, 25)
	q4 = np.percentile(rowCounts, 75)
	iqr = (q4-q2) * outlierMultiplier
	goodRowCounts = set(rowCounts[rowCounts >= (q2-iqr)])
	cleanStaffs = [rect for rect in staffs if rowCounts[rect.y] in goodRowCounts]

	diff = len(staffs) - len(cleanStaffs)
	reductionPercent = 0 if len(staffs) == 0 else (100 * float(diff)/len(staffs))
	if log:
		print("  Finished - Reduced template match count by %d%% (%d -> %d, down by %d matches)" % (round(reductionPercent), len(staffs), len(cleanStaffs), diff))

	return cleanStaffs


# Paints a list of rectangles onto the source window with the given drawing parameters
def drawRects(img, rects, windowName, drawParams = None):
	if drawParams is None:
		drawParams = DrawParams()

	rectsImg = img.copy()
	for rect in rects:
		rect.draw(rectsImg, drawParams, windowName)
	return rectsImg


# A package method of the getTemplateMatches and mergeRects methods, as well as the drawing of the results
# However, this does not show the image
# Assume that the display name is the same as the window name unless otherwise indicated
def matchAndMerge(img, grayscale, templates, windowName, fitParams = None, drawParams = None, displayName = None, plotter = None, sublog = None):
	if displayName is None:
		displayName = windowName
	if fitParams is None:
		fitParams = FitParams()

	printProcessStatus(sublog, "template_matching", 0)
	rects, scale = getTemplateMatches(grayscale, templates, templateFitParams = fitParams, templateMsg = None, plotter = plotter)
	initSize = len(rects)
	printProcessStatus(sublog, "template_matching", 1, post = "(%d matches found)" % initSize)

	printProcessStatus(sublog, "rect_merging", 0)
	rects = mergeRects(rects, fitParams.overlapThreshold, msg = None)
	mergedSize = len(rects)
	printProcessStatus(sublog, "rect_merging", 1, initSize, mergedSize)

#	printProcessStatus(sublog, "rect_drawing", 0)
#	rectsImg = drawRects(img, rects, windowName, drawParams = drawParams)
#	printProcessStatus(sublog, "rect_drawing", 1)

	return rects


# A package method that handles the template matching, overlapping rectange merging, and displaying of an image
def matchMergeAndDraw(img, grayscale, templates, windowName, fitParams = None, drawParams = None, displayName = None, plotter = None):
	rects = matchAndMerge(img, grayscale, templates, windowName, fitParams = fitParams, drawParams = drawParams, displayName = displayName, plotter = plotter)
	rectsImg = drawRects(img, rects, windowName, drawParams = drawParams)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, Rect.onClick, windowName)
#	scaledImage = cv2.resize(rectsImg, None, fx = displayScaleFactor, fy = displayScaleFactor, cv2.interCubic)
	cv2.imshow(windowName, rectsImg)
	return rects


# A central callback for trackbars on a window
# This method is called whenever a single trackbar is updated, but it
# pulls information from all trackbars to match and merge
def updateWindow(img, grayscale, templates, windowName):
	# Get all of the trackbars from the window
	start = cv2.getTrackbarPos(START_NAME, windowName)/100.0
	stop = cv2.getTrackbarPos(STOP_NAME, windowName)/100.0
	step = cv2.getTrackbarPos(STEP_NAME, windowName)/100.0
	templateThreshold = cv2.getTrackbarPos(TMP_THRESH_NAME, windowName)/100.0
	overlapThreshold = cv2.getTrackbarPos(OVERLAP_THRESH_NAME, windowName)/100.0

	rects = matchAndMerge(img, grayscale, templates, windowName, params = FitParams(start, stop, step, templateThreshold, overlapThreshold))
	rectsImg = drawRects(img, rects, windowName, drawParams = None)
	cv2.imshow(windowName, rectsImg)


def trackMatchMergeAndDraw(img, grayscale, windowName, templates, initStart = 25, initStop = 150, initStep = 1, initTmpThresh = 70, initBoxThresh = 50):
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, Rect.onClick, windowName)

	cv2.createTrackbar(START_NAME, windowName, initStart, 150, lambda x: updateWindow(img, grayscale, templates, windowName))
	cv2.createTrackbar(STOP_NAME, windowName, initStop, 150, lambda x: updateWindow(img, grayscale, templates, windowName))
	cv2.createTrackbar(STEP_NAME, windowName, initStep, 100, lambda x: updateWindow(img, grayscale, templates, windowName))
	cv2.createTrackbar(TMP_THRESH_NAME, windowName, initTmpThresh, 100, lambda x: updateWindow(img, grayscale, templates, windowName))
	cv2.createTrackbar(BOX_THRESH_NAME, windowName, initBoxThresh, 200, lambda x: updateWindow(img, grayscale, templates, windowName))
	updateWindow(img, grayscale, templates, windowName)


def printProcessStatus(sublog, task, status, initSize = None, finalSize = None, units = "matches", post = None):
	if sublog is None:
		return

	end = (" " + post + "\n") if post is not None else "\n"

	indicator = "-"
	if status == 0:
		status = "begin"
	if status == 1:
		status = "success"
		indicator = u'\u2713'

	# Pad the begining to 20 characters because of some insider knowledge about the sublog name lengths
	beginString = "[%s] LOG/%s:" % (indicator, sublog)
	beginString = beginString.ljust(20)

	if initSize is not None and finalSize is not None:
		reductionPercent = 100 * float(initSize - finalSize) / initSize if initSize > 0 else 0
		print("%sTask '%s' has achieved '%s' status (-%d%%, %d -> %d %s)" % (beginString, task, status, round(reductionPercent), initSize, finalSize, units), end = end)
	else:
		print("%sTask '%s' has achieved '%s' status" % (beginString, task, status), end = end)
	sys.stdout.flush()


def processFeature(bundle):
	name, img, grayscale, fitParams, index, dataQ = bundle
	plotter = (index, dataQ)
	if name == "Staffs":
		printProcessStatus("STAFFS", "template_matching", 0)
		staffs, scale = getTemplateMatches(grayscale, staffTmps, templateFitParams = fitParams, templateMsg = None, plotter = plotter)
		initSize = len(staffs)
		printProcessStatus("STAFFS", "template_matching", 1, post = "(%d matches found)" % initSize)

		# Remove staff boxes that are on wierd rows
		printProcessStatus("STAFFS", "match_cleaning", 0)
		staffs = cleanUpLonesomeRows(staffs)
		cleanedSize = len(staffs)
		printProcessStatus("STAFFS", "match_cleaning", 1, initSize, cleanedSize)

		# Merge staff boxes
		printProcessStatus("STAFFS", "rect_merging", 0)
		staffs = mergeRects(staffs, 0.1, msg = None)
		mergedSize = len(staffs)
		printProcessStatus("STAFFS", "rect_merging", 1, cleanedSize, mergedSize)

		# Now that the staff boxes have been cleaned up, they should be grouped row wise
		# into staff lines.  Use this to merge a very wide rectangle for each
		printProcessStatus("STAFFS", "secondary_merging", 0)
		staffs = mergeRects([Rect(0, rect.y, img.shape[1], rect.h, rect.label) for rect in staffs], 0.1, msg = None)
		printProcessStatus("STAFFS", "secondary_merging", 1, mergedSize, len(staffs))

#		printProcessStatus("STAFFS", "rect_drawing", 0)
#		staffsImg = drawRects(img, staffs, "Staffs", drawParams = DrawParams((0, 0, 255), (0, 0, 0), 2))
#		printProcessStatus("STAFFS", "rect_drawing", 1)

		resultBundle = staffs


	elif name == "Sharps":
		resultBundle = matchAndMerge(img, grayscale, sharpTmps, name, fitParams = fitParams, plotter = plotter, sublog = name.upper())
	elif name == "Flats":
		resultBundle = matchAndMerge(img, grayscale, flatTmps, name, fitParams = fitParams, plotter = plotter, sublog = name.upper())

	elif name == "Quarters":
		resultBundle = matchAndMerge(img, grayscale, quarterTmps, name, fitParams = fitParams, plotter = plotter, sublog = name.upper())
	elif name == "Halfs":
		resultBundle = matchAndMerge(img, grayscale, halfTmps, name, fitParams = fitParams, plotter = plotter, sublog = name.upper())
	elif name == "Wholes":
		resultBundle = matchAndMerge(img, grayscale, wholeTmps, name, fitParams = fitParams, plotter = plotter, sublog = name.upper())

	return (name, resultBundle)


def identifyNotes(features, staffSeps):
	# Figure out the average distance between staff boxes
	staffs = features[processNames[0]]
	staffs.sort(key = lambda staff: staff.y)
	averageStaffDistance = 0
	for c in range(len(staffs)-1):
		averageStaffDistance += staffs[c+1].y - (staffs[c].h + staffs[c].y)
	averageStaffDistance /= (len(staffs)-1) if len(staffs) > 1 else 1

	# Now sort markings into their proper staff box so they can be sorted by x value within this
	# Make sure to sort each sublist (within a staff) by x value as well
	# Also, merge on the staff lines with a min heap merge
	staffGroupedFeatures = []
	for staffRect in staffs:
		categoricalLineRects = []
		for featureType in features.keys():
			if featureType == processNames[0]:
				continue
			boxedList = [(rect, featureType) for rect in features[featureType] if abs(rect.middle[1] - staffRect.middle[1]) < (staffRect.h + averageStaffDistance/2)]
			boxedList.sort(key = lambda bundle: bundle[0].x)
			categoricalLineRects.append(boxedList)
		mergedLineRects = heapq.merge(*categoricalLineRects, key = lambda bundle: bundle[0].x)
		staffGroupedFeatures.append(mergedLineRects)

	# Now, go staff line by staff line and turn each into a note
	notes = []
	staffGroupedNotes = []
	for c in range(len(staffGroupedFeatures)):
		lineFeatures = staffGroupedFeatures[c]
		staffNotes = []

		keySharps = []
		keyFlats = []
		firstNoteFound = False
		previousAccidental = [False, False]
		for rect, featureType in lineFeatures:
			noteLine = Note.calculateNoteLine(rect, staffSeps[c][0], staffSeps[c][1])

			# Until the first note, add sharps and flats to the key rather than to notes
			if not firstNoteFound:
				# Check for sharps
				if featureType == processNames[1]:
					keySharps.append(noteLine)
					continue
				# Check for flats
				elif featureType == processNames[2]:
					keyFlats.append(noteLine)
					continue
				# Otherwise, its the first note - flip the flag
				else:
					firstNoteFound = False

			# Now, if the first note has been found, handle each feature type
			# Accidentals should change the previousAccidentals variable and other notes should be added as notes
			# Also, when a note is added, it should take the previous accidentals with it (reset them)
			duration = -1
			if featureType == processNames[1]:
				previousAccidental[0] = True
			elif featureType == processNames[2]:
				previousAccidental[1] = True
			elif featureType == processNames[3]:
				duration = 4
			elif featureType == processNames[4]:
				duration = 2
			elif featureType == processNames[5]:
				duration = 1
			else:
				print("Something else, slipped in?? (" + featureType + ")")

			if duration > 0:
				if not previousAccidental[0] and not previousAccidental[1]:
					if noteLine in keySharps:
						previousAccidental[0] = True
					elif noteLine in keyFlats:
						previousAccidental[1] = True
				staffNotes.append(Note(noteLine, duration, *previousAccidental, rect = rect))
				previousAccidental = [False, False]
		staffGroupedNotes.append(staffNotes)
		notes += staffNotes

	return notes, staffGroupedNotes


staffSetMarks = None
def staffCenterDetecting(event, x, y, flags, bundle):
	global staffSetMarks

	if not event == cv2.EVENT_LBUTTONDOWN:
		return

	features, img, windowName = bundle
	staffRects = features[processNames[0]]
	if staffSetMarks is None:
		staffSetMarks = [[None, None, 0] for c in range(len(staffRects))]
	doneDrawParams = DrawParams(color = (0, 255, 0), weight = 2)
	halfDoneDrawParams = DrawParams(color = (0, 204, 204), weight = 2)
	notDoneDrawParams = DrawParams(weight = 2)

	staffsImg = img.copy()
	for c in range(len(staffRects)):
		rect = staffRects[c]
		if rect.x < x and (rect.x + rect.w) > x and rect.y < y and (rect.y + rect.h) > y:
			nextWriteIndex = staffSetMarks[c][2]
			staffSetMarks[c][nextWriteIndex] = y
			staffSetMarks[c][2] = (nextWriteIndex+1)%2
		if staffSetMarks[c][0] is not None and staffSetMarks[c][1] is not None:
			rect.draw(staffsImg, doneDrawParams, windowName)
			cv2.rectangle(staffsImg, (0, staffSetMarks[c][0]), (img.shape[1], staffSetMarks[c][0]), (255, 0, 0), cv2.FILLED)
			cv2.rectangle(staffsImg, (0, staffSetMarks[c][1]), (img.shape[1], staffSetMarks[c][1]), (0, 0, 255), cv2.FILLED)
		elif staffSetMarks[c][0] is None and staffSetMarks[c][1] is None:
			rect.draw(staffsImg, notDoneDrawParams, windowName)
		else:
			rect.draw(staffsImg, halfDoneDrawParams, windowName)
			if staffSetMarks[c][0] is not None:
				cv2.rectangle(staffsImg, (0, staffSetMarks[c][0]), (img.shape[1], staffSetMarks[c][0]), (255, 0, 0), cv2.FILLED)
			else:
				cv2.rectangle(staffsImg, (0, staffSetMarks[c][1]), (img.shape[1], staffSetMarks[c][1]), (0, 0, 255), cv2.FILLED)


	cv2.imshow(windowName, staffsImg)


processNames = ["Staffs", "Sharps", "Flats", "Quarters", "Halfs", "Wholes"]
def main():
	# First thing is to read the source image and grayscale it
	imgFile = sys.argv[1]
	rawImg = cv2.imread(imgFile)

	factor = 1.0
	monitor = get_monitors()[0]
	screenH, screenW = monitor.height, monitor.width
	yRatio = rawImg.shape[0] / float(screenH)
	xRatio = rawImg.shape[1] / float(screenW)
	if yRatio > 0.9 and yRatio >= xRatio:
		factor = (0.9 * screenH) / rawImg.shape[0]
	elif xRatio > 0.9 and xRatio > yRatio:
		factor = (0.9 * screenW) / rawImg.shape[1]
	if factor < 1.0:
		print("Warning: This image will be reduced to %d%% scale in a way that will effect the acuracy of results.  Consider using a smaller image." % (factor*100.0))
		img = cv2.resize(rawImg, None, fx = factor, fy = factor, interpolation = cv2.INTER_CUBIC)
	else:
		img = rawImg

	grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	coreCount = multiprocessing.cpu_count()
	usingCores = 7 if coreCount > 7 else coreCount
	print("[X] LOG/CPU:        Using %d/%d cores%s" % (usingCores, coreCount, "!\n" if coreCount == usingCores else "\n"))

	printProcessStatus("PROG", "feature_processing", 0)

	manager = Manager()
	dataQ = manager.Queue()
	fitParams = [staffFitParams, sharpFitParams, flatFitParams, quarterFitParams, halfFitParams, wholeFitParams]
	plotterProcess = Process(target = Plotter.graphingWorker, args = (2, 3, dataQ, processNames, [(fitParam.start, fitParam.stop) for fitParam in fitParams]))
	plotterProcess.start()

	pool = Pool(usingCores-1)
	results = pool.map(processFeature, [(processNames[c], img, grayscale, fitParams[c], c, dataQ) for c in range(len(processNames))])


	printProcessStatus("PROG", "feature_processing", 1, post = "- displaying images")

	features = {}
	for name, rects in results:
		rectsImg = drawRects(img, rects, name)
		cv2.namedWindow(name)
		cv2.setMouseCallback(name, Rect.onClick, name)
		cv2.imshow(name, rectsImg)
		features[name] = rects

	print("\t\tPress space to continue...")
	while True:
		if cv2.waitKey(30) == 32:
			break

	dataQ.put((0, Plotter.CLOSE, 0))
	plotterProcess.join()

	cv2.destroyAllWindows()

	# Use the distances between the centers of the staff boxes to attempt to correct for single problem staff centers
	# Identify the outliers and replace them with the median
	name = "Please click on the center line and another successive lines for each staff set (click on centers)"
	staffRects = features[processNames[0]]
	staffsImg = img.copy()
	for rect in staffRects:
		rect.draw(staffsImg, DrawParams(weight = 2), name)
	cv2.namedWindow(name)
	cv2.imshow(name, staffsImg)
	cv2.setMouseCallback(name, staffCenterDetecting, (features, img, name))


	print("\t\tPress space to continue...")
	while True:
		if cv2.waitKey(30) == 32 and staffSetMarks is not None:
			done = True
			for sub in staffSetMarks:
				if None in sub:
					done = False
					break

			if done:
				break
	cv2.destroyAllWindows()


	# Identify and sort the notes, and sort the seps by their centers
	staffSeps = [(staffSetMark[0], abs(staffSetMark[1] - staffSetMark[0])) for staffSetMark in staffSetMarks]
	staffSeps.sort(key = lambda x: x[0])

	notes, staffGroupedNotes = identifyNotes(features, staffSeps)

	# Go through the notes and create one more image with all of them labeled with their note lines
	# Also, restrict note ranges to [-8, 15] note lines from the center
	rangedNotes = []
	labeledNotesImg = img.copy()
	lniName = "Labeled notes"
	drawParams = DrawParams()
	for note in notes:
		if note.noteLine >= -8 and note.noteLine <= 15:
			rangedNotes.append(note)
			note.rect.label = str(note.noteLine) + " " + str(note.duration)
			note.rect.draw(labeledNotesImg, drawParams, lniName, allInfo = False)
	cv2.namedWindow(lniName)
	cv2.setMouseCallback(lniName, Rect.onClick, lniName)
	cv2.imshow(lniName, labeledNotesImg)

#	fig, axarr = plt.subplots(2, 4)
#	for c in range(len(staffGroupedNotes)):
#		noteYCounts = {}
#		# Get the average height as well
#		averageHeight = 0
#		count = 0
#		for note in staffGroupedNotes[c]:
#			# You know who you are
#			if note.noteLine == 30:
#				continue
#			y = note.rect.y
#			if y in noteYCounts:
#				noteYCounts[y] += 1
#			else:
#				noteYCounts[y] = 1
#			averageHeight += note.rect.h
#			count += 1
#		averageHeight /= float(count)
#		# Group everything within += averageHeight/4
#		noteYs = sorted(noteYCounts)
#		yFreqs = list(noteYCounts.values())
#		diffs = np.array([(noteYs[d+1] - noteYs[d]) for d in range(len(noteYs)-1)])
#		combines = []
#		while True:
#			minIndex = np.argmin(diffs)
#			if diffs[minIndex] > (averageHeight / 4.0):
#				break
#			combines.append(minIndex)
#			diffs[minIndex] = max(max(diffs)+1, averageHeight / 4.0)
#		d = 0
#		indpNoteYs = []
#		indpYFreqs = []
#		while d < len(noteYs):
#			combine = d in combines
#			noteY = noteYs[d] if not combine else ((noteYs[d] + noteYs[d+1])/2)
#			yFreq = yFreqs[d] if not combine else (yFreqs[d] + yFreqs[d+1])
#			indpNoteYs.append(noteY)
#			indpYFreqs.append(yFreq)
#			d += 1 if not combine else 2
#
#		print("Line %d ==================================" % c)
#		print(str(noteYs) + "  =>  " + str(indpNoteYs))
#		print(str(yFreqs) + "  =>  " + str(indpYFreqs))
#		print()
#
#		axarr[0][c].set_title("Line %d" % c)
#		axarr[0][c].plot(noteYs, yFreqs, '-gD')
#		axarr[1][c].plot(indpNoteYs, indpYFreqs, '-gD')
#	plt.ion()
#	plt.show()

	print("\t\tPress space to continue...")
	while True:
		if cv2.waitKey(30) == 32:
			break

	cv2.destroyAllWindows()

	# Write the ranged notes to the output file
	pickleFile = imgFile[:imgFile.rindex(".")] + ".pickle"
	with open(pickleFile, "wb") as fileHandler:
		pickle.dump((rawImg, factor, rangedNotes), fileHandler)
	print("Results have been pickled into '%s'" % pickleFile)
	print("Thank you for using fastOmr!")




if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print()
		sys.exit(0)
