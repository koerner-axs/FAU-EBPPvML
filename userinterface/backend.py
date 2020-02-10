import threading
import queue
import os
import watchdog.observers
import watchdog.events

import res.constants as constants
import res.inference as inference
import res.analytics as analytics
import res.cube_marching as cube_marching
import database


WT_TASK_PROCESS_SINGLE = 0
WT_TASK_PROCESS_BATCH = 1


class WatchdogThreadTask:
	def __init__(self, priority, data):
		self.priority = priority
		self.data = data
		self.cv = threading.Condition()
		self.result = None

	def __lt__(self, other):
		return self.priority < other.priority

	def execute(self, watchdogThread):
		raise NotImplementedError()

	def run(self, watchdogThread):
		try:
			result = self.execute(watchdogThread)
		except Exception as e:
			print('An exception was raised in the execution of a watchdog thread task:')
			print(e)
			result = -1
		with self.cv:
			self.result = result
			self.cv.notify_all()

	def join(self, timeout=None):
		with self.cv:
			self.cv.wait_for(lambda: self.result is not None, timeout)
		return self.result


class WatchdogThreadTaskBeginWatchDirectory(WatchdogThreadTask):
	def __init__(self):
		super(WatchdogThreadTaskBeginWatchDirectory, self).__init__(4, None)

	def execute(self, watchdogThread):
		print(watchdogThread.directory)
		if watchdogThread.directory is not None:
			watchdogThread.watchdog.schedule(EventHandler(watchdogThread), watchdogThread.directory, recursive=False)
		return 0


class WatchdogThreadTaskBeginProcessingDirectory(WatchdogThreadTask):
	def __init__(self):
		super(WatchdogThreadTaskBeginProcessingDirectory, self).__init__(3, None)

	def execute(self, watchdogThread):
		folder = watchdogThread.directory
		files = os.listdir(folder)
		queuedLayers = list()
		# TODO: Fetch information about which layers have been processed from the database.
		#    -> For now, assume all viable input files are intended input.
		for file in files:
			if file.endswith('.bmp') and not file.startswith('proc_'):
				try:
					layer_id = int(file.replace('.bmp', '', 1).replace('image', '', 1))
					queuedLayers.append(layer_id)
				except Exception as e:
					print('Filename not parseable!')
					print(e)
		queuedLayers.sort()
		print('Beginning batched processing of', len(queuedLayers), 'items.')
		for commitBatch in constants.batch(queuedLayers, constants.BATCHED_PROCESSING_COMMIT_SIZE):
			for batch in constants.batch(commitBatch, constants.BATCHED_PROCESSING_BATCH_SIZE):
				watchdogThread.workerThread.queueBatch(('F:/Machine Learning/FAU - EBPPvML/userinterface/test', list(batch)))
			watchdogThread.dbThread.commit()
		return 0


class WatchdogThreadTaskSetDirectory(WatchdogThreadTask):
	def __init__(self, directory):
		super(WatchdogThreadTaskSetDirectory, self).__init__(2, directory)

	def execute(self, watchdogThread):
		watchdogThread.watchdog.unschedule_all()
		watchdogThread.directory = self.data
		if watchdogThread.directory is not None:
			os.makedirs(os.path.join(watchdogThread.directory, 'results'), exist_ok=True)
		return 0


class WatchdogThreadTaskClose(WatchdogThreadTask):
	def __init__(self):
		super(WatchdogThreadTaskClose, self).__init__(2, None)

	def execute(self, watchdogThread):
		watchdogThread.watchdog.unschedule_all()
		watchdogThread.directory = None
		watchdogThread.workerThread.terminateProcessing()
		return 0


class WatchdogThreadTaskEndWatchDirectory(WatchdogThreadTask):
	def __init__(self):
		super(WatchdogThreadTaskEndWatchDirectory, self).__init__(1, None)

	def execute(self, watchdogThread):
		watchdogThread.watchdog.unschedule_all()
		return 0


class WatchdogThreadTaskEndProcessingDirectory(WatchdogThreadTask):
	def __init__(self):
		super(WatchdogThreadTaskEndProcessingDirectory, self).__init__(0, None)

	def execute(self, watchdogThread):
		watchdogThread.workerThread.terminateProcessing()
		return 0


class EventHandler(watchdog.events.FileSystemEventHandler):
	def __init__(self, watchdogThread):
		self.watchdogThread = watchdogThread

	def on_created(self, event):
		if type(event) is watchdog.events.FileCreatedEvent:
			filename = event.src_path
			directory, file = os.path.split(filename)
			if file.endswith('.bmp') and not file.startswith('proc_'):
				print('Detected creation of new image file:', file)
				try:
					layer_id = int(file.replace('.bmp', '', 1).replace('image', '', 1))
					self.watchdogThread.workerThread.queueSingle((directory, layer_id))
				except Exception as e:
					print('Filename not parseable!')
					print(e)
					return


class WatchdogThread(threading.Thread):
	def __init__(self, dbThread):
		super(WatchdogThread, self).__init__(name='WatchdogThread', daemon=True)
		self.pqueue = queue.PriorityQueue()
		self.directory = None
		self.dbThread = dbThread
		self.watchdog = None
		self.workerThread = None

	def __del__(self):
		del self.watchdog
		del self.workerThread

	def run(self):
		self.watchdog = watchdog.observers.Observer()
		self.watchdog.start()
		self.workerThread = WorkerThread(self.dbThread)
		self.workerThread.start()
		while True:
			try:
				task = self.pqueue.get()
				task.run(self)
				self.pqueue.task_done()
			except Exception as e:
				print(e)

	def queue(self, task):
		self.pqueue.put(task)

	def close(self):
		task = WatchdogThreadTaskClose()
		self.queue(task)
		return task.join()

	def queueChangeDirectoryTask(self, directory):
		task = WatchdogThreadTaskSetDirectory(directory)
		self.queue(task)
		return task.join()

	def queueBeginProcessingTask(self):
		task = WatchdogThreadTaskBeginProcessingDirectory()
		self.queue(task)
		return task.join()

	def queueBeginWatchingTask(self):
		task = WatchdogThreadTaskBeginWatchDirectory()
		self.queue(task)
		return task.join()

	def queueEndProcessingTask(self):
		task = WatchdogThreadTaskEndProcessingDirectory()
		self.queue(task)
		return task.join()

	def queueEndWatchingTask(self):
		task = WatchdogThreadTaskEndWatchDirectory()
		self.queue(task)
		return task.join()


class WorkerThread(threading.Thread):
	def __init__(self, dbThread):
		super(WorkerThread, self).__init__(name='WorkerThread', daemon=True)
		self.queue = queue.Queue()
		self.dbThread = dbThread
		self.cv = threading.Condition()
		self.isactive = False
		self.waiting = True

	def run(self):
		inference.init('.\\..\\models\\quadout\\segm_t1566481907_era3.tfkem')
		while True:
			with self.cv:
				self.waiting = True
				self.cv.notify()
			num, data = self.queue.get()
			self.waiting = False
			if num == WT_TASK_PROCESS_SINGLE:
				self._process(data)
				self.queue.task_done()
			elif num == WT_TASK_PROCESS_BATCH:
				self._process_batch(data)
				self.queue.task_done()
			else:
				raise ValueError('The worker thread received an invalid request. (tasktype=' + str(num) + ' with data: ', str(data) + ')')

	def queueSingle(self, item):
		self.queue.put((WT_TASK_PROCESS_SINGLE, item))

	def queueBatch(self, batch):
		self.queue.put((WT_TASK_PROCESS_BATCH, batch))

	def terminateProcessing(self):
		while not self.queue.empty():
			self.queue.get_nowait()
			self.queue.task_done()
		self.cv.wait_for(lambda: self.waiting)

	def _process(self, item):
		directory, layer_id = item
		print('Processing layer with id:', layer_id)

		# Correct sequencing of input files is not guaranteed at this point!
		try:
			result = inference.predict(directory, layer_id)
		except Exception as e:
			print('Processing of layer with id:', layer_id, 'failed.')
			raise e

		analysis = analytics.feed(result, layer_id, self.dbThread)
		#cube_marching.feed(result, analysis, layer_id, self.dbThread)
		print('Layer ' + str(layer_id) + ' was successfully processed.')

	def _process_batch(self, batch):
		directory, ids = batch
		try:
			result = inference.predict_batch(directory, ids)
		except Exception as e:
			raise e

		for index, layer_id in enumerate(ids):
			analysis = analytics.feed(result[index], layer_id, self.dbThread)
			#cube_marching.feed(result[index], analysis, layer_id, self.dbThread)
			#print('Layer ' + str(layer_id) + ' was successfully processed.')