import sys
import queue
import matplotlib.pyplot as plt
from params import FitParams

# This class handles all of the plotting required by the omr
class Plotter:
	CLOSE, DATA, X_MIN, X_MAX, Y_MIN, Y_MAX, TITLE, COLOR = range(-1, 7)

	@staticmethod
	def graphingWorker(nrows, ncols, dataQ, titles, xBounds, yBounds = [], colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']):
		fig, axarr = plt.subplots(nrows, ncols)
		fig.set_size_inches(5 * ncols, 3 * nrows)
		fig.suptitle("Scale vs. Hit Count")
		fig.subplots_adjust(hspace = 0.3)
		axarr = axarr.flatten()

		lines = []
		for c in range(len(axarr)):
			line, = axarr[c].plot([], [])
			line.set_color(colors[c%len(colors)])
			lines.append(line)
			if c < len(titles):
				axarr[c].set_title(titles[c])
			if c < len(xBounds):
				axarr[c].set_xbound(xBounds[c][0], xBounds[c][1])
				if c < len(yBounds):
					axarr[c].set_ybound(ybounds[c][0], ybounds[c][1])
				else:
					axarr[c].set_ybound(-1, 1)

		fig.canvas.draw()
		plt.ion()
		plt.show()

		while True:
			# If there is a long queue, this could be an indicator of bigger problem
			initQLength = dataQ.qsize()
			if initQLength > 20:
				print("[X] LOG/PLOTTER:    The data queue is becoming concerningly large (%d events unhandled)" % initQLength)
				sys.stdout.flush()

			init = dataQ.qsize()
			events = []
			event = dataQ.get()
			events.append(event)
			# Get any events that are waiting in the queue if multiple events made it
			while True:
				try:
					event = dataQ.get(block = False)
					events.append(event)
				except queue.Empty:
					break

			# Update data
			for subplot, requestType, data in events:
				if subplot >= (nrows * ncols):
					print("[!] LOG/PLOTTER:    Error: DataQ submitted request (code: %d) for a non existent subplot (%d)" % (requestType, subplot))
					sys.stdout.flush()
				# Add new data
				if requestType == 0:
					lines[subplot].set_xdata(data[0])
					lines[subplot].set_ydata(data[1])
				# Update x min
				elif requestType == 1:
					axarr[subplot].set_xbound(data, None)
				# Update x max
				elif requestType == 2:
					axarr[subplot].set_xbound(None, data)
				# Update y min
				elif requestType == 3:
					axarr[subplot].set_ybound(data, None)
				# Update y max
				elif requestType == 4:
					axarr[subplot].set_ybound(None, data)
				# Set title
				elif requestType == 5:
					axarr[subplot].set_title(data)
				# Set color
				elif requestType == 6:
					lines[subplot].set_color(data)
				# We outta here
				elif requestType < 0:
					return
			fig.canvas.draw()
			plt.pause(0.0001)
