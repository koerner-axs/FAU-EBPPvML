import sqlite3 as s3
import threading
import queue
import numpy as np
import io


def adapt_array(arr):
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return s3.Binary(out.read())

def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out)

s3.register_adapter(np.ndarray, adapt_array)
s3.register_converter('array', convert_array)


class DatabaseThreadTask:
	def __init__(self, priority, data, priority_secondary=0):
		self.priority = priority
		self.priority_secondary = priority_secondary
		self.data = data
		self.cv = threading.Condition()
		self.result = None

	def __lt__(self, other):
		return True if self.priority < other.priority else self.priority_secondary < other.priority_secondary

	def execute(self, dbThread):
		raise NotImplementedError()

	def run(self, dbThread):
		result = self.execute(dbThread)
		with self.cv:
			self.result = result
			self.cv.notify_all()

	def join(self, timeout=None):
		with self.cv:
			self.cv.wait_for(lambda: self.result is not None, timeout)
		return self.result


class DatabaseThreadTaskEstablishConnection(DatabaseThreadTask):
	def __init__(self, filename):
		super(DatabaseThreadTaskEstablishConnection, self).__init__(0, filename)

	def execute(self, dbThread):
		try:
			dbThread.dbConnection.commit()
			dbThread.dbConnection.close()
		except NameError:
			pass
		except AttributeError:
			pass
		except Exception as e:
			print(e)
			pass
		dbThread.dbConnection = s3.connect(self.data, detect_types=s3.PARSE_DECLTYPES)
		printDbEntry = None
		try:
			cursor = dbThread.dbConnection.cursor()
			cursor.execute('''SELECT * FROM Print''')
			printDbEntry = cursor.fetchone()
		except s3.Error as e:
			print(e)
		if printDbEntry is None:
			self.initDatabase(dbThread, 'SQLWELLASD3232')
		return 0

	def initDatabase(self, dbThread, identifier):
		curs = dbThread.dbConnection.cursor()

		tables = ['Print', 'SegmentedImages', 'PPOHistory']
		for table in tables:
			try:
				curs.execute('DROP TABLE ' + table)
			except Exception as e:
				print('Exception occurred whilest dropping table', table)
				print(e)

		tableProperties = {'Print': 'num_layers int, is_done boolean, identifier varchar not null, part_3d_orig array, part_3d_faults array, PRIMARY KEY (identifier)',
		                   'SegmentedImages': 'id int not null, layer_data array, stats_porous float, stats_bulged float, stats_fine float, stats_background float, PRIMARY KEY (id)',
		                   'PPOHistory': 'id int not null, change varchar not null, effect float, loss float, PRIMARY KEY (id)'}
		for table in tables:
			try:
				curs.execute('CREATE TABLE ' + table + ' (' + tableProperties[table] + ');')
			except Exception as e:
				print('Exception occurred whilest creating table', table)
				print(e)

		try:
			curs.execute('''INSERT INTO Print VALUES (?, ?, ?, ?, ?);''', (0, True, identifier, None, None))
		except Exception as e:
			print('Exception occurred when inserting header into the Print table')
			print(e)

		dbThread.dbConnection.commit()


class DatabaseThreadTaskCloseConnection(DatabaseThreadTask):
	def __init__(self):
		super(DatabaseThreadTaskCloseConnection, self).__init__(0, None)

	def execute(self, dbThread):
		try:
			dbThread.dbConnection.commit()
			dbThread.dbConnection.close()
		except Exception as e:
			print(e)
		dbThread.dbConnection = None
		return 0


class DatabaseThreadTaskCommit(DatabaseThreadTask):
	def __init__(self):
		super(DatabaseThreadTaskCommit, self).__init__(1, None)

	def execute(self, dbThread):
		try:
			dbThread.dbConnection.commit()
		except Exception as e:
			return e
		return 0


class DatabaseThreadTaskFetchLayer(DatabaseThreadTask):
	def __init__(self, layer_id):
		super(DatabaseThreadTaskFetchLayer, self).__init__(2, layer_id)

	def execute(self, dbThread):
		cursor = dbThread.dbConnection.cursor()
		cursor.execute('''SELECT * FROM SegmentedImages WHERE id={0};'''.format(self.data))
		return cursor.fetchone()


class DatabaseThreadTaskInsertLayer(DatabaseThreadTask):
	def __init__(self, data):
		assert(len(data) == 6)
		super(DatabaseThreadTaskInsertLayer, self).__init__(3, data, priority_secondary=data[0])

	def execute(self, dbThread):
		dbThread.dbConnection.execute('''INSERT INTO SegmentedImages VALUES (?, ?, ?, ?, ?, ?);''', self.data)
		print('Commit of layer', self.data[0], 'done.')
		return 0


class DatabaseThread(threading.Thread):
	def __init__(self):
		super(DatabaseThread, self).__init__(name='DatabaseThread', daemon=True)
		self.pqueue = queue.PriorityQueue(maxsize=5000)
		self.dbConnection = None

	def __del__(self):
		try:
			self.dbConnection.close()
		except:
			pass

	def run(self):
		while True:
			try:
				task = self.pqueue.get()
				task.run(self)
				self.pqueue.task_done()
			except Exception as e:
				print(e)

	def queue(self, task):
		self.pqueue.put(task)

	def setDatabaseFile(self, file):
		try:
			while not self.pqueue.empty():
				self.pqueue.get_nowait()
				self.pqueue.task_done()
		except:
			pass
		task = DatabaseThreadTaskEstablishConnection(file)
		self.queue(task)
		print('Waiting for database thread to confirm database file change.')
		errcode = task.join(timeout=5)
		if errcode is 0:
			print('Received confirmation.')
			return True
		else:
			print('Establishing database connection failed with errcode:', errcode)
			return False

	def closeConnection(self):
		task = DatabaseThreadTaskCloseConnection()
		self.queue(task)
		return task.join(timeout=5)

	def commit(self):
		task = DatabaseThreadTaskCommit()
		self.queue(task)
		return task.join()

	def fetchLayer(self, layer_id):
		if self.dbConnection is not None:
			task = DatabaseThreadTaskFetchLayer(layer_id)
			self.queue(task)
			return task.join(timeout=3)

	def insertLayer(self, id, img, fine, bulged, porous, background):
		task = DatabaseThreadTaskInsertLayer((id, img, porous, bulged, fine, background))
		self.queue(task)


class Database:
	def __init__(self, filepath=None):
		self.isOpen = False
		self.connection = None
		self.open(filepath)

	def isOpen(self):
		return self.isOpen

	def close(self):
		if self.connection:
			self.connection.close()
		self.filepath = None
		self.isOpen = False

	def open(self, filepath):
		if self.isOpen:
			close()
		self.filepath = filepath
		if self.filepath != None:
			self.connection = s3.connect(self.filepath, detect_types=s3.PARSE_DECLTYPES, )
			self.isOpen = True

	def commitLayer(self, layer_id):
		print('TODO: commitLayer(self, layer_id) in database')
		self.connection.commit()

	def _initialize(self, identifier):
		curs = self.connection.cursor()
		try:
			curs.execute('''DROP TABLE Print''')
			curs.execute('''DROP TABLE SegmentedImages''')
			curs.execute('''DROP TABLE PPOHistory''')
		except:
			pass
		curs.execute('''CREATE TABLE Print (
								num_layers int,
								is_done boolean,
								identifier varchar not null,
								part_3d_orig array,
								part_3d_faults array,
								PRIMARY KEY (identifier));''')
		curs.execute('''CREATE TABLE SegmentedImages (
								id int not null,
								layer_data array,
								stats_porous float,
								stats_bulged float,
								stats_fine float,
								stats_background float,
								PRIMARY KEY (id));''')
		curs.execute('''CREATE TABLE PPOHistory (
								id int not null,
								change varchar not null,
								effect float,
								loss float,
								PRIMARY KEY (id));''')

		curs.execute('''INSERT INTO Print VALUES (?, ?, ?, ?, ?);''', (0, True, identifier, None, None))
		self.connection.commit()

if __name__=='__main__':
	db = Database('database.db')
	db._initialize('AOISD')