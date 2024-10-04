#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:13:50 2019

@author: leandro
"""

import os
import datetime

class Log():

	logPath = None

	def __init__(self, logPath = None, settings = None, printToTerminal = True):
		"""
			logPath : path to log file
			settings: dict object with variables and values that represent current run
		"""
		self.logPath = logPath
		self.printToTerminal = printToTerminal

		if self.logPath:
			if not os.path.exists(os.path.dirname(logPath)):
				os.makedirs(os.path.dirname(logPath))

			with open(self.logPath, "w+") as f:
				f.write( "[{}] Created\n".format(datetime.datetime.now()) )
				if settings:
					f.write("\n")
					for var, value in settings.items():
						f.write( "{}: {}\n".format(var,value) )

	def setVerbose(self, v):
		self.printToTerminal = v

	def log(self, what):
		if self.printToTerminal:
			print(what)

		if self.logPath:
			with open(self.logPath, "a+") as f:
				f.write(what)
				f.write("\n")

	def debug(self, what):
		print("[DEBUG] " + what)

	def end(self):
		with open(self.logPath, "a+") as f:
			f.write( "\n[{}] Finished\n".format(datetime.datetime.now()) )
