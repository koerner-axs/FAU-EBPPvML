import os
import numpy as np
import pyglet as pg
from pyglet.gl import *
from pyglet.window import key
from pyglet.window import mouse
from res.constants import IMAGE_SIZE
import database as db
import backend

REQ_VERSION_ = '1.0'

for file in os.listdir('./res/'):
	if file.endswith('.version'):
		v = file.replace('.version', '')
		if v != REQ_VERSION_:
			raise ImportError(
				'Resource files are not of the correct version: REQ_VERSION_ = ' + REQ_VERSION_ + ' <-> ' + v)
		break


class Rect:
	# Paddings for left, bottom, right, top side.
	def __init__(self, x, y, width, height, paddings=(0, 0, 0, 0)):
		self.x = int(x)
		self.y = int(y)
		self.width = int(width)
		self.height = int(height)
		if paddings == (0, 0, 0, 0):
			self.innerRect = self
		else:
			innerX = round(self.x + paddings[0])
			innerY = round(self.y + paddings[1])
			innerWidth = round(self.width - paddings[0] - paddings[2])
			innerHeight = round(self.height - paddings[1] - paddings[3])
			self.innerRect = Rect(innerX, innerY, innerWidth, innerHeight)

	def isInside(self, px, py):
		return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

	def asTuple(self):
		return self.x, self.y, self.width, self.height

	def asVertexData(self, border=0):
		return [self.x + border, self.y + border, self.x + border, self.y + self.height - border,
		        self.x + self.width - border, self.y + self.height - border, self.x + self.width - border,
		        self.y + border]

	def getCenterX(self):
		return round(self.x + self.width / 2)

	def getCenterY(self):
		return round(self.y + self.height / 2)

	def setViewportToRect(self):
		glViewport(self.x, self.y, self.width, self.height)

	def setScissorsToRect(self):
		glScissor(self.x, self.y, self.width, self.height)

	def fillWithTexture(self, texture):
		uv = texture.tex_coords

		glEnable(GL_TEXTURE_2D)
		glBindTexture(texture.target, texture.id)

		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		glOrtho(0, self.width, 0, self.height, -1, 1)
		glMatrixMode(GL_MODELVIEW)

		glBegin(GL_QUADS)
		glColor3f(1.0, 1.0, 1.0)
		glTexCoord2f(uv[9], uv[10])
		glVertex2f(0, 0)
		glTexCoord2f(uv[0], uv[1])
		glVertex2f(0, self.height)
		glTexCoord2f(uv[3], uv[4])
		glVertex2f(self.width, self.height)
		glTexCoord2f(uv[6], uv[7])
		glVertex2f(self.width, 0)
		glEnd()

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)

		glDisable(GL_TEXTURE_2D)

	# Uses glClear command to draw the borders. Call before drawing the actual contents.
	def drawBorderToInnerRect(self, r=1.0, g=0.0, b=0.0):
		if self.innerRect != self:
			glEnable(GL_SCISSOR_TEST)
			self.setScissorsToRect()
			glClearColor(r, g, b, 1.0)
			glClear(GL_COLOR_BUFFER_BIT)
			self.innerRect.setScissorsToRect()
			glClearColor(0.0, 0.0, 0.0, 1.0)
			glClear(GL_COLOR_BUFFER_BIT)
			glDisable(GL_SCISSOR_TEST)

	# Uses glClear command to draw the inner rect. Call before drawing the actual contents.
	def drawInnerRect(self, r=1.0, g=0.0, b=0.0):
		glEnable(GL_SCISSOR_TEST)
		self.innerRect.setScissorsToRect()
		glClearColor(r, g, b, 1.0)
		glClear(GL_COLOR_BUFFER_BIT)
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glDisable(GL_SCISSOR_TEST)


class Button:
	def __init__(self, rect, caption, texture, enabled=True, doDraw=True):
		self.rect = Rect(*rect.innerRect.asTuple(), paddings=[5, 5, 5, 5])
		self.label = pg.text.Label(caption if caption else '', font_name='Times New Roman',
		                           font_size=15,
		                           x=self.rect.innerRect.getCenterX(),
		                           y=self.rect.innerRect.getCenterY(),
		                           align='center', anchor_x='center', anchor_y='center')
		self.texture = texture
		self.pressed = False
		self.enabled = enabled
		self.doDraw = doDraw

	def draw(self):
		if self.doDraw:
			# self.rect.fillWithTexture(self.texture)

			glEnable(GL_SCISSOR_TEST)
			self.rect.setScissorsToRect()
			if not self.enabled:
				glClearColor(0.1, 0.1, 0.1, 1.0)
			elif self.pressed:
				glClearColor(0.2, 0.2, 0.2, 1.0)
			else:
				glClearColor(0.35, 0.35, 0.35, 1.0)
			glClear(GL_COLOR_BUFFER_BIT)
			glClearColor(0.0, 0.0, 0.0, 1.0)
			glDisable(GL_SCISSOR_TEST)

			self.label.draw()


class TextWidget:
	def __init__(self, text):
		self.active = False
		self.batch = pg.graphics.Batch()
		self.document = pg.text.document.UnformattedDocument(text)
		self.document.set_style(0, len(self.document.text), dict(color=(255, 255, 255, 255), font_size=18))
		font = self.document.get_font()
		self.height = font.ascent - font.descent

		self.layout = pg.text.layout.IncrementalTextLayout(
			self.document, 0, 0, multiline=False, batch=self.batch)
		self.caret = pg.text.caret.Caret(self.layout)

	def computePositioning(self, x, y, width):
		self.rect = Rect(x - 2, (y - self.height / 2) - 2, width + 4, self.height + 4, paddings=[2, 2, 2, 2])
		self.layout.x = self.rect.innerRect.x + 2
		self.layout.y = self.rect.innerRect.y
		self.layout.width = self.rect.innerRect.width - 4
		self.layout.height = self.rect.innerRect.height

	def draw(self):
		self.rect.drawBorderToInnerRect()
		if self.active:
			self.rect.drawInnerRect(0.4, 0.4, 0.4)
		self.batch.draw()

	def setState(self, active):
		self.active = active
		if self.active:
			self.caret.on_activate()
		else:
			self.caret.move_to_point(0, 0)
			self.caret.on_deactivate()

	def hit_test(self, x, y):
		return self.rect.innerRect.isInside(x, y)


class RotationYawPitch:
	def __init__(self, yaw, pitch):
		self.yaw = yaw
		self.pitch = pitch

	def apply(self):
		glRotatef(self.pitch, 1, 0, 0)
		glRotatef(self.yaw, 0, 1, 0)

	def __repr__(self):
		return 'RotationYawPitch: yaw={0}, pitch={1}'.format(self.yaw, self.pitch)


class Transform:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def apply(self):
		glTranslatef(self.x, self.y, self.z)

	def __repr__(self):
		return 'Transform: x={0}, y={1}, z={2}'.format(self.x, self.y, self.z)


class InferenceState:
	def __init__(self, name, left_button, right_button, stateString):
		self.name = name
		self.left_button = left_button
		self.right_button = right_button
		self.stateString = stateString


class InferenceStateMachine:
	def __init__(self):
		self.callbackDict = dict()
		self.stateDict = self.makeStates()
		self.state = self.stateDict['ready']

	def registerCallback(self, name, function):
		self.callbackDict[name] = function

	def transitionLeft(self):
		if self.state.left_button['enabled']:
			callback = self.callbackDict[self.state.left_button['callback']]
			if callback():
				self.state = self.stateDict[self.state.left_button['next_state']]

	def transitionRight(self):
		if self.state.right_button['enabled']:
			callback = self.callbackDict[self.state.right_button['callback']]
			if callback():
				self.state = self.stateDict[self.state.right_button['next_state']]

	def forceTransition(self, newStateName) -> bool:
		if self.state.name != newStateName:
			print('Forcing inference state to', newStateName)
			self.state = self.stateDict[newStateName]
			return True
		return False

	def makeLeftButton(self, rect):
		return Button(rect, self.state.left_button['caption'], None, enabled=self.state.left_button['enabled'])

	def makeRightButton(self, rect):
		return Button(rect, self.state.right_button['caption'], None, enabled=self.state.right_button['enabled'], doDraw=self.state.right_button['enabled'])

	def getStateString(self):
		return self.state.stateString

	def makeStates(self):
		states = dict()
		states['ready'] = InferenceState('ready',
		                                 {'caption': 'Open', 'enabled': True, 'callback': 'onOpenDirectory', 'next_state': 'opened'},
		                                 {'caption': None, 'enabled': False, 'callback': None, 'next_state': None},
		                                 'Ready')
		states['opened'] = InferenceState('opened',
		                                  {'caption': 'Process Directory', 'enabled': True, 'callback': 'onProcessDirectory', 'next_state': 'processing'},
		                                  {'caption': 'Watch Directory', 'enabled': True, 'callback': 'onWatchDirectory', 'next_state': 'watching'},
		                                  'Ready')
		states['processing'] = InferenceState('processing',
		                                      {'caption': 'Stop Processing', 'enabled': True, 'callback': 'onStopProcessingDirectory', 'next_state': 'opened'},
		                                      {'caption': 'Watch Directory', 'enabled': True, 'callback': 'onWatchDirectory', 'next_state': 'processing_watching'},
		                                      'Processing directory contents')
		states['processing_watching'] = InferenceState('processing_watching',
		                                               {'caption': 'Stop Processing', 'enabled': True, 'callback': 'onStopProcessingDirectory', 'next_state': 'watching'},
		                                               {'caption': 'Stop Watching', 'enabled': True, 'callback': 'onStopWatchingDirectory', 'next_state': 'processing'},
		                                               'Processing directory contents and watching for changes')
		states['watching'] = InferenceState('watching',
		                                    {'caption': 'Process Directory', 'enabled': True, 'callback': 'onProcessDirectory', 'next_state': 'processing_watching'},
		                                    {'caption': 'Stop Watching', 'enabled': True, 'callback': 'onStopWatchingDirectory', 'next_state': 'opened'},
		                                    'Watching for changes')
		return states


class StateCockpit:
	def __init__(self, subwindowRect, mainWindow):
		self.labelInputFolder = None
		self.buttonLeft = None
		self.buttonRight = None
		self.labelState = None
		self.subwindow = subwindowRect
		self.mainWindow = mainWindow
		self.text_cursor = self.mainWindow.get_system_mouse_cursor('text')
		self.widgetInputFolder = TextWidget('F:\\Machine Learning\\FAU - EBPPvML\\userinterface\\test')
		self.stateMachine = InferenceStateMachine()
		self.registerCallbacks()
		self.computeSubwindows()

	def draw(self):
		self.subwindow.drawBorderToInnerRect(0.0, 0.0, 1.0)
		self.labelInputFolder.draw()
		self.widgetInputFolder.draw()
		self.buttonLeft.draw()
		self.buttonRight.draw()
		self.labelState.draw()

	def onResize(self, rect):
		self.subwindow = rect
		self.computeSubwindows()

	def dispatchMousePress(self, x, y, button, modifiers):
		isOnWidget = self.widgetInputFolder.hit_test(x, y)
		self.widgetInputFolder.setState(isOnWidget)
		if isOnWidget:
			if self.stateMachine.forceTransition('ready'):
				self.mainWindow.dbThread.closeConnection()
				self.mainWindow.watchdogThread.close()
			self.computeSubwindows()
			self.widgetInputFolder.caret.on_mouse_press(x, y, button, modifiers)
		elif self.buttonLeft.rect.isInside(x, y):
			self.buttonLeft.pressed = True
			self.buttonRight.pressed = False
		elif self.buttonRight.rect.isInside(x, y):
			self.buttonLeft.pressed = False
			self.buttonRight.pressed = True

	def dispatchMouseRelease(self, x, y, button, modifiers):
		if self.buttonLeft.rect.isInside(x, y) and self.buttonLeft.pressed:
			self.buttonLeft.pressed = False
			self.stateMachine.transitionLeft()
			self.computeSubwindows()
		elif self.buttonRight.rect.isInside(x, y) and self.buttonRight.pressed and self.buttonRight.enabled:
			self.buttonRight.pressed = False
			self.stateMachine.transitionRight()
			self.computeSubwindows()

	def computeSubwindows(self):
		rect = self.subwindow.innerRect
		frac_input, frac_buttons, frac_state = 0.35, 0.35, 0.3
		row_height_input = frac_input * rect.height
		row_height_buttons = frac_buttons * rect.height
		row_height_state = frac_state * rect.height

		frac_label = 0.35
		label_width = rect.width * frac_label
		label_y_middle = rect.y + row_height_state + row_height_buttons + row_height_input / 2
		self.labelInputFolder = pg.text.Label('Input Folder:', font_size=24,
		                                      x=rect.x + label_width / 2,
		                                      y=label_y_middle,
		                                      anchor_x='center', anchor_y='center', align='center')
		self.widgetInputFolder.computePositioning(rect.x + label_width, label_y_middle,
		                                          rect.width * (1 - frac_label))

		button_padding = [10, 10, 10, 10]
		button_width = rect.width / 2
		self.buttonLeft = self.stateMachine.makeLeftButton(Rect(rect.x, rect.y + row_height_state,
		                              button_width, row_height_buttons, button_padding))
		self.buttonRight = self.stateMachine.makeRightButton(Rect(rect.x + button_width, rect.y + row_height_state,
		                               button_width, row_height_buttons, button_padding))

		self.labelState = pg.text.Label(self.stateMachine.getStateString(), font_size=18,
		                                x=rect.x + rect.width / 2,
		                                y=rect.y + row_height_state / 2,
		                                anchor_x='center', anchor_y='center', align='center')

	def onOpenDirectory(self):
		path = self.widgetInputFolder.document.text
		print('onOpenDirectory: filepath=' + path)
		if not os.path.exists(path):
			print('The given path appears to be incorrect!')
			return False
		if not self.mainWindow.dbThread.setDatabaseFile(os.path.join(path, 'database.db')):
			return False
		return self.mainWindow.watchdogThread.queueChangeDirectoryTask(path) is 0

	def onProcessDirectory(self):
		print('onProcessDirectory')
		self.mainWindow.watchdogThread.queueBeginProcessingTask()
		return True

	def onWatchDirectory(self):
		print('onWatchDirectory')
		self.mainWindow.watchdogThread.queueBeginWatchingTask()
		return True

	def onStopProcessingDirectory(self):
		print('onStopProcessingDirectory')
		self.mainWindow.watchdogThread.queueEndProcessingTask()
		return True

	def onStopWatchingDirectory(self):
		print('onStopWatchingDirectory')
		self.mainWindow.watchdogThread.queueEndWatchingTask()
		return True

	def registerCallbacks(self):
		self.stateMachine.registerCallback('onOpenDirectory', self.onOpenDirectory)
		self.stateMachine.registerCallback('onProcessDirectory', self.onProcessDirectory)
		self.stateMachine.registerCallback('onWatchDirectory', self.onWatchDirectory)
		self.stateMachine.registerCallback('onStopProcessingDirectory', self.onStopProcessingDirectory)
		self.stateMachine.registerCallback('onStopWatchingDirectory', self.onStopWatchingDirectory)


class State2DAnalysis:
	def __init__(self, subwindowRect, dbThread):
		self.subwindow = subwindowRect
		self.active = False
		self.currentLayerId = 0
		self.maxLayerId = 9505
		self.computeSubwindows()
		self.dbThread = dbThread
		self.layerImageData = pg.image.ImageData(IMAGE_SIZE, IMAGE_SIZE, 'RGB',
		                                         data=(np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
		                                                        dtype=np.uint8)).tobytes(),
		                                         pitch=None)

	def dispatchKeyboardPress(self, symbol, modifiers):
		if symbol == key.LEFT:
			self.buttonPreviousImage.pressed = True
			self.buttonSlideshowPause.pressed = False
			self.buttonNextImage.pressed = False
		if symbol == key.RIGHT:
			self.buttonPreviousImage.pressed = False
			self.buttonSlideshowPause.pressed = False
			self.buttonNextImage.pressed = True

	def dispatchKeyboardRelease(self, symbol, modifiers):
		if symbol == key.LEFT and self.buttonPreviousImage.pressed:
			self.buttonPreviousImage.pressed = False
			self.showPrevious(modifiers)
		if symbol == key.RIGHT and self.buttonNextImage.pressed:
			self.buttonNextImage.pressed = False
			self.showNext(modifiers)

	def dispatchMousePress(self, x, y, button, modifiers):
		if button == mouse.LEFT:
			if self.buttonPreviousImage.rect.isInside(x, y):
				self.buttonPreviousImage.pressed = True
				self.buttonSlideshowPause.pressed = False
				self.buttonNextImage.pressed = False
			elif self.buttonSlideshowPause.rect.isInside(x, y):
				self.buttonPreviousImage.pressed = False
				self.buttonSlideshowPause.pressed = True
				self.buttonNextImage.pressed = False
			elif self.buttonNextImage.rect.isInside(x, y):
				self.buttonPreviousImage.pressed = False
				self.buttonSlideshowPause.pressed = False
				self.buttonNextImage.pressed = True

	def dispatchMouseRelease(self, x, y, button, modifiers):
		if button == mouse.LEFT:
			if self.buttonPreviousImage.rect.isInside(x, y) and self.buttonPreviousImage.pressed:
				self.buttonPreviousImage.pressed = False
				self.showPrevious(modifiers)
			elif self.buttonSlideshowPause.rect.isInside(x, y) and self.buttonSlideshowPause.pressed:
				self.buttonSlideshowPause.pressed = False
				self.toggleSlideshow()
			elif self.buttonNextImage.rect.isInside(x, y) and self.buttonNextImage.pressed:
				self.buttonNextImage.pressed = False
				self.showNext(modifiers)

	def updateLayer(self, newLayerId):
		print('Fetching layer with id', newLayerId)
		data = self.dbThread.fetchLayer(newLayerId)
		if data is None:
			print('Request for layer timed out or layer wasn\'t present: layerid=' + str(newLayerId))
			return
		_, img, stats_porous, stats_buldged, stats_fine, stats_background = data
		import cv2
		cv2.imshow('Current layer', img)
		self.layerImageData = pg.image.ImageData(IMAGE_SIZE, IMAGE_SIZE, 'BGR', data=img.tobytes(order='C'),
		                                         pitch=3 * IMAGE_SIZE)
		print('Successfully updated layer.')
		self.currentLayerId = newLayerId

		self.controlStateLabel.text = str(self.currentLayerId) + ' / ' + str(self.maxLayerId)
		self.infoboxLabels['layer_porous_label'].text = 'Porosity: {:3.1f}%'.format(stats_porous * 100)
		self.infoboxLabels['layer_bulged_label'].text = 'Bulged: {:3.1f}%'.format(stats_buldged * 100)
		self.infoboxLabels['layer_fine_label'].text = 'Fine: {:3.1f}%'.format(stats_fine * 100)
		self.infoboxLabels['layer_background_label'].text = 'Background: {:3.1f}%'.format(stats_background * 100)

	def showPrevious(self, modifiers):
		if modifiers & key.MOD_CTRL:
			# Seek to first image.
			new = 0
		elif modifiers & key.MOD_ALT:
			# Move 100 images back.
			new = max(0, self.currentLayerId - 100)
		elif modifiers & key.MOD_SHIFT:
			# Move ten images back.
			new = max(0, self.currentLayerId - 10)
		else:
			# Move one image back.
			new = max(0, self.currentLayerId - 1)
		self.updateLayer(new)

	def showNext(self, modifiers):
		if modifiers & key.MOD_CTRL:
			# Seek to last image.
			new = self.maxLayerId
		elif modifiers & key.MOD_ALT:
			# Move 100 images forward.
			new = min(self.maxLayerId, self.currentLayerId + 100)
		elif modifiers & key.MOD_SHIFT:
			# Move ten images forward.
			new = min(self.maxLayerId, self.currentLayerId + 10)
		else:
			# Move one image forward.
			new = min(self.maxLayerId, self.currentLayerId + 1)
		self.updateLayer(new)

	def toggleSlideshow(self):
		print('Toggling Slideshow')

	def computeSubwindows(self):
		## Calculate subwindows.
		fact, imagePadding, controlPadding, infoboxPadding = 0.6, [25, 25, 25, 25], [15, 15, 15, 15], [15, 15, 15, 15]
		imageSize = min(round(self.subwindow.width * fact), round(self.subwindow.height * fact))

		# Set subwindows accordingly.
		self.subwindowImage = Rect(self.subwindow.x, self.subwindow.y + self.subwindow.height - imageSize,
		                           imageSize, imageSize, paddings=imagePadding)
		self.subwindowControls = Rect(self.subwindow.x, self.subwindow.y, imageSize,
		                              self.subwindow.height - imageSize, paddings=controlPadding)
		self.subwindowInfobox = Rect(self.subwindow.x + imageSize, self.subwindow.y + self.subwindow.height - imageSize,
		                             self.subwindow.width - imageSize, imageSize, paddings=infoboxPadding)

		# Compute positioning of UI elements.
		# Controls Subwindow
		padding = 10
		minAspectRatio = 1.5
		third = self.subwindowControls.innerRect.width / 3.0

		# Previous Image Button
		posX = self.subwindowControls.innerRect.x + padding
		width = round(third - 2 * padding)
		height = round(min(width / minAspectRatio, self.subwindowControls.innerRect.height / 4))
		posY = self.subwindowControls.innerRect.y + self.subwindowControls.innerRect.height - height - padding
		self.buttonPreviousImage = Button(Rect(posX, posY, width, height),
		                                  'Previous', None)

		# Slideshow Start/Stop Button
		posX = self.subwindowControls.innerRect.x + third + padding
		self.buttonSlideshowPause = Button(Rect(posX, posY, width, height),
		                                   'UnPause', None)

		# Next Image Button
		posX = self.subwindowControls.innerRect.x + 2 * third + padding
		self.buttonNextImage = Button(Rect(posX, posY, width, height),
		                              'Next', None)

		# Control state Label
		factor = 0.6
		x, y, w, h = self.subwindowControls.innerRect.asTuple()
		self.controlStateLabelRect = Rect(x + w * ((1 - factor) / 2), y + h * ((1 - factor) / 2),
		                                  w * factor, h * factor - height - 2 * padding)
		caption = str(self.currentLayerId) + ' / ' + str(self.maxLayerId)
		self.controlStateLabel = pg.text.Label(caption, font_name='Times New Roman',
		                                       font_size=24,
		                                       x=self.controlStateLabelRect.innerRect.getCenterX(),
		                                       y=self.controlStateLabelRect.innerRect.getCenterY(),
		                                       anchor_x='center', anchor_y='center', align='center')

		# Infobox Subwindow
		frac_logo, frac_layer, frac_total = 0.20, 0.55, 0.25
		logo_padding = 5

		row_height = round(self.subwindowInfobox.innerRect.height / 5.0)
		row_width = self.subwindowInfobox.innerRect.width
		column_logo_width = round(frac_logo * row_width)
		column_layer_width = round(frac_layer * row_width)
		column_total_width = round(frac_total * row_width)
		bottom = self.subwindowInfobox.innerRect.y
		left = self.subwindowInfobox.innerRect.x

		largest_square = min(row_height, column_logo_width) - 2 * logo_padding
		pad_top_bottom = (row_height - largest_square) // 2
		pad_left_right = (column_logo_width - largest_square) // 2
		logo_padding = [pad_left_right, pad_top_bottom, pad_left_right, pad_top_bottom]

		logo_porous = (Rect(left, bottom + 3 * row_height, column_logo_width, row_height, logo_padding),
		               (1.0, 0.0, 0.0, 1.0))
		logo_bulging = (Rect(left, bottom + 2 * row_height, column_logo_width, row_height, logo_padding),
		                (0.0, 1.0, 0.0, 1.0))
		logo_fine = (Rect(left, bottom + 1 * row_height, column_logo_width, row_height, logo_padding),
		             (0.0, 0.0, 1.0, 1.0))
		self.infoboxLogos = [logo_porous, logo_bulging, logo_fine]

		column_layer_center_x = left + column_logo_width + column_layer_width // 2
		column_center_y = bottom + row_height // 2
		column_total_center_x = left + column_logo_width + column_layer_width + column_total_width // 2

		self.infoboxLabels = dict()

		self.infoboxLabels['layer_top_label'] = pg.text.Label('Layer',
		                                                      font_name='Times New Roman', font_size=20,
		                                                      x=column_layer_center_x,
		                                                      y=column_center_y + 4 * row_height,
		                                                      anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['layer_porous_label'] = pg.text.Label('Porosity: (-.-%)',
		                                                         font_name='Times New Roman', font_size=14,
		                                                         x=column_layer_center_x,
		                                                         y=column_center_y + 3 * row_height,
		                                                         anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['layer_bulged_label'] = pg.text.Label('Bulged: (-.-%)',
		                                                         font_name='Times New Roman', font_size=14,
		                                                         x=column_layer_center_x,
		                                                         y=column_center_y + 2 * row_height,
		                                                         anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['layer_fine_label'] = pg.text.Label('Fine: (-.-%)',
		                                                       font_name='Times New Roman', font_size=14,
		                                                       x=column_layer_center_x,
		                                                       y=column_center_y + 1 * row_height,
		                                                       anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['layer_background_label'] = pg.text.Label('Background: (-.-%)',
		                                                             font_name='Times New Roman', font_size=14,
		                                                             x=column_layer_center_x,
		                                                             y=column_center_y,
		                                                             anchor_x='center', anchor_y='center',
		                                                             align='center')

		self.infoboxLabels['total_top_label'] = pg.text.Label('Total',
		                                                      font_name='Times New Roman', font_size=20,
		                                                      x=column_total_center_x,
		                                                      y=column_center_y + 4 * row_height,
		                                                      anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['total_porosity_label'] = pg.text.Label('(-.-%)',
		                                                           font_name='Times New Roman', font_size=14,
		                                                           x=column_total_center_x,
		                                                           y=column_center_y + 3 * row_height,
		                                                           anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['total_bulged_label'] = pg.text.Label('(-.-%)',
		                                                         font_name='Times New Roman', font_size=14,
		                                                         x=column_total_center_x,
		                                                         y=column_center_y + 2 * row_height,
		                                                         anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['total_fine_label'] = pg.text.Label('(-.-%)',
		                                                       font_name='Times New Roman', font_size=14,
		                                                       x=column_total_center_x,
		                                                       y=column_center_y + 1 * row_height,
		                                                       anchor_x='center', anchor_y='center', align='center')
		self.infoboxLabels['total_background_label'] = pg.text.Label('(-.-%)',
		                                                             font_name='Times New Roman', font_size=14,
		                                                             x=column_total_center_x,
		                                                             y=column_center_y,
		                                                             anchor_x='center', anchor_y='center',
		                                                             align='center')

	def drawButtons(self):
		self.buttonPreviousImage.draw()
		self.buttonSlideshowPause.draw()
		self.buttonNextImage.draw()

	def drawControlState(self):
		self.controlStateLabelRect.drawInnerRect()
		self.controlStateLabel.draw()

	def drawInfobox(self):
		for logo in self.infoboxLogos:
			rect, colour = logo
			rect.drawInnerRect(colour[0], colour[1], colour[2])
		for lbl in self.infoboxLabels.values():
			lbl.draw()

	def onResize(self, rect):
		self.subwindow = rect
		self.computeSubwindows()

	def draw(self, mainWindowRect):
		## Draw the 2d analysis tool into its subwindow rect.
		self.subwindow.drawBorderToInnerRect(1, 0, 0)
		self.subwindowInfobox.drawBorderToInnerRect(1, 1, 0)
		self.subwindowControls.drawBorderToInnerRect(0, 1, 0)
		# self.subwindowImage.drawBorderToInnerRect(0, 0, 1)

		# Draw the image subwindow.
		self.subwindowImage.innerRect.setViewportToRect()
		self.subwindowImage.innerRect.fillWithTexture(self.layerImageData.get_texture())
		# self.subwindowImage.innerRect.fillWithTexture(self.pic248Texture)

		# Draw the controls subwindow.
		# Text Labels are unfortunately incompatible with limiting the viewport..
		# This is necessary, so that the positioning circuitry does not get overly complicated.
		mainWindowRect.setViewportToRect()
		self.drawButtons()
		self.drawControlState()
		self.drawInfobox()


class State3DAnalysis:
	def __init__(self, rect):
		self.subwindow = rect
		self.modelTransform = RotationYawPitch(45.0, 30.0)
		self.viewTransform = Transform(0.0, 0.0, -5.0)
		self.active = False
		self.isDraggedFromInside = False

	def applyModelViewTransform(self):
		self.viewTransform.apply()
		self.modelTransform.apply()

	def onMoveEvent(self, amount):
		# Calculate step size
		step = amount * 0.1

		# Apply transform
		self.viewTransform.z = min(0, self.viewTransform.z + step)

	def onRotateEvent(self, dx, dy):
		# Calculate step size
		stepYaw = dx * 0.1
		stepPitch = dy * -0.1

		# Apply rotation
		self.modelTransform.yaw += stepYaw
		self.modelTransform.pitch += stepPitch

	def dispatchKeyboardRelease(self, symbol, modifiers):
		if symbol == key.Y:
			self.onMoveEvent(2)
		elif symbol == key.X:
			self.onMoveEvent(-2)
		elif symbol == key.W:
			self.modelTransform.pitch += 10.0
		elif symbol == key.S:
			self.modelTransform.pitch -= 10.0
		elif symbol == key.A:
			self.modelTransform.yaw += 10.0
		elif symbol == key.D:
			self.modelTransform.yaw -= 10.0

	def dispatchMouseScroll(self, scroll_y):
		self.onMoveEvent(scroll_y)

	def dispatchMouseDrag(self, dx, dy, buttons):
		if buttons == pg.window.mouse.RIGHT or buttons == mouse.MIDDLE:
			self.onRotateEvent(dx, dy)

	def draw(self):
		# Draw the 3d analysis tool into its subwindow rect.
		if self.active:
			# Highlight the subrect if the 3d analysis subrect is in focus.
			glClearColor(0.15, 0.15, 0.15, 1.0)
			glEnable(GL_SCISSOR_TEST)
			self.subwindow.innerRect.setScissorsToRect()
			glClear(GL_COLOR_BUFFER_BIT)
			glDisable(GL_SCISSOR_TEST)
			glClearColor(0.0, 0.0, 0.0, 1.0)

		self.subwindow.innerRect.setViewportToRect()
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluPerspective(75, self.subwindow.width / float(self.subwindow.height), .001, 1000)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		self.applyModelViewTransform()

		glBegin(GL_QUADS)  # Begin drawing the color cube with 6 quads
		# Top face (y = 1.0)
		# Define vertices in counter-clockwise (CCW) order with normal pointing out
		glColor3f(0.0, 1.0, 0.0)  # Green
		glVertex3f(1.0, 1.0, -1.0)
		glVertex3f(-1.0, 1.0, -1.0)
		glVertex3f(-1.0, 1.0, 1.0)
		glVertex3f(1.0, 1.0, 1.0)

		# Bottom face (y = -1.0)
		glColor3f(1.0, 0.5, 0.0)  # Orange
		glVertex3f(1.0, -1.0, 1.0)
		glVertex3f(-1.0, -1.0, 1.0)
		glVertex3f(-1.0, -1.0, -1.0)
		glVertex3f(1.0, -1.0, -1.0)

		# Front face  (z = 1.0)
		glColor3f(1.0, 0.0, 0.0)  # Red
		glVertex3f(1.0, 1.0, 1.0)
		glVertex3f(-1.0, 1.0, 1.0)
		glVertex3f(-1.0, -1.0, 1.0)
		glVertex3f(1.0, -1.0, 1.0)

		# Back face (z = -1.0)
		glColor3f(1.0, 1.0, 0.0)  # Yellow
		glVertex3f(1.0, -1.0, -1.0)
		glVertex3f(-1.0, -1.0, -1.0)
		glVertex3f(-1.0, 1.0, -1.0)
		glVertex3f(1.0, 1.0, -1.0)

		# Left face (x = -1.0)
		glColor3f(0.0, 0.0, 1.0)  # Blue
		glVertex3f(-1.0, 1.0, 1.0)
		glVertex3f(-1.0, 1.0, -1.0)
		glVertex3f(-1.0, -1.0, -1.0)
		glVertex3f(-1.0, -1.0, 1.0)

		# Right face (x = 1.0)
		glColor3f(1.0, 0.0, 1.0)  # Magenta
		glVertex3f(1.0, 1.0, -1.0)
		glVertex3f(1.0, 1.0, 1.0)
		glVertex3f(1.0, -1.0, 1.0)
		glVertex3f(1.0, -1.0, -1.0)
		glEnd()  # End of drawing color-cube

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()


class MainWindow(pg.window.Window):
	def __init__(self, dbThread, watchdogThread, width=1320, height=720):
		super(MainWindow, self).__init__(resizable=True, width=width, height=height,
		                                 caption='EBPPvML Analysis Tool')
		self.dbThread = dbThread
		self.watchdogThread = watchdogThread
		self.set_minimum_size(1320, 720)

		## Setup window
		self.windowRect = Rect(0, 0, width, height)
		self.calculateWindows()

		## Setup state for subwindows.
		self.stateCockpit = StateCockpit(self.subwindowCockpitRect, self)
		self.state2DAnalysis = State2DAnalysis(self.subwindow2DAnalysisRect, self.dbThread)
		self.state3DAnalysis = State3DAnalysis(self.subwindow3DAnalysisRect)

		## Build user interface.
		self.buildStaticDrawVertexArray()

		self.initGL()

	def calculateWindows(self):
		## Calculate subwindow sizes.
		fact, border = 0.75, 5

		# Rects with x, y, width and height.
		self.subwindow2DAnalysisRect = Rect(0, 0, self.windowRect.width // 2,
		                                    round(self.windowRect.height * (fact)),
		                                    [1 * border, 1 * border, 0.5 * border, 0.5 * border])
		self.subwindowCockpitRect = Rect(0, self.subwindow2DAnalysisRect.height,
		                                 self.windowRect.width // 2, round(self.windowRect.height * (1 - fact)),
		                                 [1 * border, 0.5 * border, 0.5 * border, 1 * border])
		self.subwindow3DAnalysisRect = Rect(self.windowRect.width // 2, 0,
		                                    self.windowRect.width // 2, self.windowRect.height,
		                                    [0.5 * border, 1 * border, 1 * border, 1 * border])

	def buildStaticDrawVertexArray(self):
		# Build main vertex list.
		vertexData = self.subwindowCockpitRect.innerRect.asVertexData()
		# vertexData.extend(self.subwindow2DAnalysisRect.innerRect.asVertexData())
		# vertexData.extend(self.subwindow3DAnalysisRect.innerRect.asVertexData())
		self.mainVertexList = pg.graphics.vertex_list(len(vertexData) // 2,
		                                              ('v2i', tuple(vertexData)),
		                                              ('c3B', (255, 0, 0, 0, 255, 0, 128, 128, 128, 0, 0, 255)))

	def drawCockpitSubwindow(self):
		self.stateCockpit.draw()

	def draw2DAnalysisSubwindow(self):
		self.state2DAnalysis.draw(self.windowRect)

	def draw3DAnalysisSubwindow(self):
		self.state3DAnalysis.draw()

	def initGL(self):
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClearDepth(1.0)
		glEnable(GL_DEPTH_TEST)
		glDepthFunc(GL_LEQUAL)
		glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

	def on_draw(self):
		try:
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

			# Draw user interface
			self.windowRect.setViewportToRect()
			self.drawCockpitSubwindow()
			self.draw2DAnalysisSubwindow()
			self.draw3DAnalysisSubwindow()
		# self.windowRect.setViewportToRect()
		# self.mainVertexList.draw(pg.gl.GL_QUADS)
		except Exception as e:
			print(e)

	def on_resize(self, width, height):
		try:
			self.windowRect = Rect(0, 0, width, height)
			self.calculateWindows()
			self.stateCockpit.onResize(self.subwindowCockpitRect)
			self.state2DAnalysis.onResize(self.subwindow2DAnalysisRect)

			self.buildStaticDrawVertexArray()

			self.windowRect.setViewportToRect()
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			glOrtho(0, self.windowRect.width, 0, self.windowRect.height, -1, 1)
			glMatrixMode(GL_MODELVIEW)
		except Exception as e:
			print(e)

	def on_key_press(self, symbol, modifiers):
		try:
			if self.state2DAnalysis.active:
				self.state2DAnalysis.dispatchKeyboardPress(symbol, modifiers)
		except Exception as e:
			print(e)

	def on_key_release(self, symbol, modifiers):
		try:
			if self.state3DAnalysis.active:
				self.state3DAnalysis.dispatchKeyboardRelease(symbol, modifiers)
			if self.state2DAnalysis.active:
				self.state2DAnalysis.dispatchKeyboardRelease(symbol, modifiers)
		except Exception as e:
			print(e)

	def on_mouse_motion(self, x, y, dx, dy):
		if self.stateCockpit.widgetInputFolder.hit_test(x, y):
			self.set_mouse_cursor(self.stateCockpit.text_cursor)
		else:
			self.set_mouse_cursor(None)

	def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
		try:
			if self.state3DAnalysis.active:
				self.state3DAnalysis.dispatchMouseScroll(scroll_y)
		except Exception as e:
			print(e)

	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
		try:
			if self.state3DAnalysis.active and self.state3DAnalysis.isDraggedFromInside:
				self.state3DAnalysis.dispatchMouseDrag(dx, dy, buttons)
			if self.stateCockpit.widgetInputFolder.active:
				self.stateCockpit.widgetInputFolder.caret.on_mouse_drag(x, y, dx, dy, buttons, modifiers)
		except Exception as e:
			print(e)

	def on_mouse_press(self, x, y, button, modifiers):
		try:
			if self.state3DAnalysis.active and self.subwindow3DAnalysisRect.innerRect.isInside(x, y):
				self.state3DAnalysis.isDraggedFromInside = True
			if self.subwindow2DAnalysisRect.innerRect.isInside(x, y):
				self.state2DAnalysis.dispatchMousePress(x, y, button, modifiers)
			if self.subwindowCockpitRect.innerRect.isInside(x, y):
				self.stateCockpit.dispatchMousePress(x, y, button, modifiers)
		except Exception as e:
			print(e)

	def on_mouse_release(self, x, y, button, modifiers):
		try:
			if button == mouse.LEFT:
				self.state2DAnalysis.active = False
				if not self.state3DAnalysis.isDraggedFromInside or not self.state3DAnalysis.active:
					self.state3DAnalysis.active = self.subwindow3DAnalysisRect.innerRect.isInside(x, y)
				if self.subwindow2DAnalysisRect.innerRect.isInside(x, y):
					self.state2DAnalysis.active = True
					self.state2DAnalysis.dispatchMouseRelease(x, y, button, modifiers)
				if self.subwindowCockpitRect.innerRect.isInside(x, y):
					self.stateCockpit.dispatchMouseRelease(x, y, button, modifiers)

			self.state3DAnalysis.isDraggedFromInside = False
		except Exception as e:
			print(e)

	def on_text(self, text):
		if self.stateCockpit.widgetInputFolder.active:
			self.stateCockpit.widgetInputFolder.caret.on_text(text)

	def on_text_motion(self, motion):
		if self.stateCockpit.widgetInputFolder.active:
			self.stateCockpit.widgetInputFolder.caret.on_text_motion(motion)

	def on_text_motion_select(self, motion):
		if self.stateCockpit.widgetInputFolder.active:
			self.stateCockpit.widgetInputFolder.caret.on_text_motion_select(motion)


if __name__ == '__main__':
	dbThread = db.DatabaseThread()
	print('Created database thread.')
	watchdogThread = backend.WatchdogThread(dbThread)
	print('Created watchdog thread.')
	window = MainWindow(dbThread, watchdogThread)
	print('Created main window.')

	dbThread.start()
	watchdogThread.start()

	# dbThread.setDatabaseFile('database.db')
	# print('Successfully set the database file.')
	#
	# watchdogThread.queueChangeDirectoryTask('F:/Machine Learning/FAU - EBPPvML/userinterface/test')
	# print('Successfully set the watched directory.')

	# def batch(iterable, n=1):
	# 	l = len(iterable)
	# 	for ndx in range(0, l, n):
	# 		yield iterable[ndx:min(ndx + n, l)]

	# for fss in batch(range(100), 100):
	# 	for fs in batch(fss, 10):
	# 		watchdogThread.queueBatchProcessingTask('F:/Machine Learning/FAU - EBPPvML/userinterface/test', list(fs))
	# 	dbThread.commit()

	try:
		pg.app.run()
	except Exception as e:
		print(e)
	finally:
		del watchdogThread
		dbThread.commit()
		del dbThread
		print('Main thread terminating..')
		exit(0)
