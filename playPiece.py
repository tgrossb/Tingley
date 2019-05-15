#!/usr/bin/env python3

import sys
import os
import cv2
import pickle
from rect import Rect
from params import FitParams, DrawParams
from note import Note
from pydub import AudioSegment

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

	bpm = None
	while bpm is None:
		bpm = input("How many beats per minute should this be played at (ex. 90)?  ")
		try:
			bpm = int(bpm)
			if bpm <= 0:
				bpm = None
		except ValueError:
			print("Error: '" + str(bpm) + " is not a valid integer greater than 0")
			bpm = None

	print("Building full waveform")
	msPerBar = 60000.0 / bpm * 4.0
	waveform = None
	for note in rangedNotes:
		# Open the appropriate waveform
		song = AudioSegment.from_wav("sounds/" + str(note.midi) + ".wav")
		msDuration = int(msPerBar / note.duration)
		if msDuration >= len(song):
			print("Error: BPM too low, the samples aren't long enough")
			sys.exit(0)
		sub = song[:msDuration]
		if waveform is None:
			waveform = sub
		else:
			waveform = waveform.append(sub)
	wavFileName = pickleFile[:pickleFile.rindex(".")] + ".wav"
	waveform.export(wavFileName, format = "wav")
	print("Exported concatenative waveform to '" + wavFileName + "'")

	print("Running WaveNet with the generated waveform as a seed")
	dir = pickleFile[:pickleFile.rindex("/")]
	os.system('nsynth_generate --checkpoint_path=checkpoint/model.ckpt-200000 --source_path=' + wavFileName  + ' --save_path=' + dir + ' --batch_size=4')


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print()
		sys.exit(0)
