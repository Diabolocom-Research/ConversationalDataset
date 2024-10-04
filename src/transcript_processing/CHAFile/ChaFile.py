#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:59:04 2018

@author: Leandro Garber

"""

import os
from subprocess import getstatusoutput
import re
from .log import Log

from lexical_diversity import lex_div as ld

# LINE constants. Use these for getting data from each line
LINE_UTTERANCE = "emisión"
LINE_UTTERANCE_NUMBER = "número_de_emisión"
LINE_NUMBER = "número_de_linea"
LINE_SPEAKER = "hablante"
LINE_ADDRESSEE = "destinatario"
LINE_BULLET = "bullet"
LINE_NOUNS = "sustantivos"
LINE_ADJECTIVES = "adjetivos"
LINE_VERBS = "verbos"
LINE_LIGHT_VERBS = "light-verbs"
LINE_MOR_TO_WORDS = "mor-to-words"
#################################

# Speaker constants. line[LINE_SPEAKER] will be one of these
SPEAKER_SILENCE = "SIL"
SPEAKER_TARGET_CHILD = "CHI"
SPEAKER_OTHER_CHILD = "OCH"
SPEAKER_ADULT = "ADULT"
SPEAKER_PET = "P"
SPEAKER_OTHER = "OTHER"
SPEAKER_UNKNOWN = "UNKNOWN"
SPEAKER_BOTH = "BOTH" #ADULT + (TARGET || CHILD) This is a problem so we are discarding it
SPEAKER_CODE = "COD"
###############################################

# Use these constants for calling count method
ADDRESSEE_CHILD_DIRECTED = "cds"
ADDRESSEE_CHILD_PRODUCED = "chi"
ADDRESSEE_OVER_HEARD = "ohs"
ADDRESSEE_ADULT = "adult"
ADDRESSEE_ALL = "all"

COUNT_TYPE_TOKENS = "tokens"
COUNT_TYPE_TYPES = "types"
###############################################

# Lexical diversity constants.
LEXICAL_DIVERSITY_TTR = "ttr"
LEXICAL_DIVERSITY_MATTR = "mattr"
LEXICAL_DIVERSITY_MAAS = "maas_ttr"
LEXICAL_DIVERSITY_HDD = "hdd" #vocd like, default
LEXICAL_DIVERSITY_MTLD = "mtld"
###############################################

# Language constants
LANGUAGE_SPANISH = "spa"
LANGUAGE_ENGLISH = "eng"
######################

# MOR constants. 
# i.e line[TIER_MOR][0][MOR_UNIT_LEXEMA] will store the lexeme of the first word 
ERROR_NO_MOR_FOUND = 1

MOR_ERROR_NO_MOR_FOUND = "TIER \"%MOR\", ASSOCIATED WITH A SELECTED SPEAKER"
MOR_UNIT_CATEGORIA = "categoria"
MOR_UNIT_LEXEMA = "lexema"
MOR_UNIT_CATEGORIA_LEXEMA = "categoria|lexema" #solo para la búsqueda
MOR_UNIT_EXTRA = "extra"
MOR_UNIT_AMBIGUOUS = "ambiguo"
MOR_STOP_WORDS = [  ["imp", "da", "-2S&IMP~pro:clit|3S"], #lexema o [categoria, lexema] o [categoria, lexema, extra]
					["co", "dale"],
					"okay",
					# "like",
					"right"
]
STOP_WORDS = [
	"dale",
	"okay",
	"right"
]
MOR_REPLACEMENTS = { #reemplazo de palabras que está agarrando mal el MOR, ej: n|papi debería ser n|papá
	"papi" : "papá",
	"mami" : "mamá",
	"vamos" : "i" #co|vamos -> co|i esto es porque toma como "co" algo que es "v"
}
###############################

BULLET_TAG = "\x15"

# TIER constants
TIER_MOR = "mor"
TIER_XDS = "xds"
###############################

## Internal use only. Use SPEAKER_*
ADDRESSEE_TAG = "[+ %s]"
ADDRESSEE_XDS_CHILD = "C"
ADDRESSEE_XDS_OTHER = "O"
ADDRESSEE_XDS_UNKNOWN = "U"
ADDRESSEE_XDS_ADULT = "A"
ADDRESSEE_XDS_TARGET_CHILD = "T"
ADDRESSEE_XDS_PET = "P"
ADDRESSEE_XDS_BOTH = "B"
###############

# Internal use
ADDRESSEE_CORRESPOND = {
	ADDRESSEE_XDS_CHILD : SPEAKER_OTHER_CHILD,
	ADDRESSEE_XDS_OTHER : SPEAKER_OTHER,
	ADDRESSEE_XDS_UNKNOWN : SPEAKER_UNKNOWN,
	ADDRESSEE_XDS_ADULT : SPEAKER_ADULT,
	ADDRESSEE_XDS_TARGET_CHILD : SPEAKER_TARGET_CHILD,
	ADDRESSEE_XDS_BOTH : SPEAKER_BOTH,
	ADDRESSEE_XDS_PET : SPEAKER_PET
}

CATEGORIAS_VERBOS = ["v","ger","part","imp","inf","cop", "aux"] #"cop" and "aux" are removed by default when counting verbs
CATEGORIAS_ADJETIVOS = ["adj"]
CATEGORIAS_SUSTANTIVOS = ["n", "n:gerund"] #n:gerund was found in english and want to count it as a noun

MISSING_VALUE = "?"
WORD_XXX = "xxx" #the word wasn't understood by the transcriber
###############################################

log = Log()

class ChaFile:

	def __init__(self, chaFilePath,
				 ignoreSpeakers = [ SPEAKER_SILENCE ], onlyCDS = False, includeLines = [],
				 verbose = True, language = None):
		"""Constructor. Loads the CHA file and parse it

		Args:
			chaFilePath (string): Path to the CHA file
			ignoreSpeakers (list, optional): Utterances from these speakers won't be parsed. Defaults to [ SPEAKER_SILENCE ].
			onlyCDS (bool, optional): Only child directed utterances will be parsed. Defaults to False.
			includeLines (list, optional): Only these line numbers will be parsed. Defaults to [] which means all lines.
			verbose (bool, optional): Extra information will be printed when processing. Defaults to False.
			language (string, optional): Use one of the LANGUAGE constants or None for parsing it from the CHA file. Defaults to None.
		"""

		self.noBullets = True
		self.lines = []
		self.speakers = []
		self.language = None
		self.morAmbiguousLines = []

		self.processedVerbs = False
		self.processedNouns = False
		self.processedAdjectives = False

		self.morFound = False #true if MOR was found in at least one line

		self.chaFilePath = chaFilePath
		self.ignoreSpeakers = ignoreSpeakers
		self.onlyCDS = onlyCDS
		self.includeLines = includeLines

		log.setVerbose(verbose)

		self.filename = os.path.basename(chaFilePath)
		self.filename = self.filename[0:self.filename.rfind(".")]

		self.setLanguage(language)
		self.processLines()

	def processLines(self):
		"""Internal use. Main function that parses the CHA file

		Raises:
			FileNotFoundError: The path to the CHA file does not exist
		"""

		if not os.path.isfile(self.chaFilePath):
			raise FileNotFoundError()

		with open(self.chaFilePath,"r", encoding="utf-8") as f:
			txtCHA = f.read()

		prog = re.compile(r"[\*%@]\w*:\t.*?(?=\*\w*:\t|@End|\Z)", re.S)#, re.X)
		parsedCHA = prog.findall(txtCHA)

		self.lines = []
		self.speakers = []
		lineNumber = 1
		utteranceNumber = 0

		for r in parsedCHA:
			#is header
			if r[0] == "@":
				lineNumber += r.count("\n") + 2 #2= utf8 and begin 
				continue
			
			prog = re.compile(r"(?P<tier>[\*%][\w-]*):[\s]*(?P<content>.*?)(?=[\*%][\w-]*:[\s]*|@End|\Z)", re.S)
			
			line = {
				LINE_NUMBER : lineNumber,
				LINE_UTTERANCE_NUMBER : 0
			}
			
			skipLine = False

			#build line
			for m in prog.finditer(r):
				tier = m.group("tier")
				content = m.group("content").replace("\t"," ").replace("\n", "")

				if m.group("tier")[0] == "*": #is speaker
					speaker = tier[1:]
					if not speaker in self.ignoreSpeakers:
						if not speaker in self.speakers:
							self.speakers.append(speaker)

						line[LINE_SPEAKER] = speaker
						line[LINE_UTTERANCE] = content

						if BULLET_TAG in content:
							regexBullet = r"\x15(?P<from>\d*)_(?P<to>\d*)\x15"
							progBullet = re.compile(regexBullet)#, re.S)
							parsedBullet = list(progBullet.finditer(content))

							if len(parsedBullet) > 0 :
								bulletFrom = int(parsedBullet[0].group("from"))
								bulletTo = int(parsedBullet[-1].group("to"))
								line[ LINE_BULLET ] = [bulletFrom, bulletTo]
								line[ LINE_UTTERANCE ] = progBullet.sub("", line[ LINE_UTTERANCE ])
							else:
								#esto sucede cuando los bullets son de la forma %snd:"filename"_from_to
								line[ LINE_UTTERANCE ] = line[ LINE_UTTERANCE ].replace(BULLET_TAG, "").strip()
					else:
						skipLine = True
						break
				else:
					tierName = tier[1:]
					line[tierName] = content

					if tierName == "mor":
						self.morFound = True

					#no es realmente un tier
					#esto sucede cuando los bullets son de la forma %snd:"filename"_from_to
					if tierName == "snd":
						del line[tierName]
						lstContent = content.replace(BULLET_TAG, "").split("_")
						line[ LINE_BULLET ] = [int(lstContent[-2]), int(lstContent[-1])]

					if hasattr(self, f"_parse{tierName.capitalize()}" ):
						tierProcessFunction = getattr(self, f"_parse{tierName.capitalize()}" )
						line[tierName] = tierProcessFunction( line[tierName], line[LINE_NUMBER] )
			
			
			if not skipLine:
				self._setAddressee(line)

				if not (self.onlyCDS and line[LINE_ADDRESSEE] not in [SPEAKER_TARGET_CHILD, SPEAKER_BOTH]):
					if not speaker in self.ignoreSpeakers:
						if len(self.includeLines) == 0 or line[ LINE_NUMBER ] in self.includeLines:
							utteranceNumber += 1
							line[LINE_UTTERANCE_NUMBER] = utteranceNumber
							self.lines.append(line)
			
			lineNumber += r.count("\n")

		#if MOR is found on file all lines should have at least an empty TIER_MOR
		if self.morFound:
			for l in self.getLines():
				if TIER_MOR not in l:
					l[TIER_MOR] = []

	def getLines(self):
		"""Get an array of parsed utterances

		Returns:
			list: Utterances. Access data using the LINE constants
		"""
		return self.lines

	def getLine(self, lineNumber):
		"""Get a line by its number

		Args:
			lineNumber (int): Line number

		Returns:
			dict: Utterance
		"""
		for line in self.lines:
			if line[LINE_NUMBER] == lineNumber:
				return line

		return None

	def getLinesFromTo(self, lineFrom, lineTo):
		"""Get a range of utterances

		Args:
			lineFrom (int): Line number from
			lineTo (int): Line number to

		Returns:
			list: Utterances. Access data using the LINE constants
		"""
		linesToReturn = []

		for line in self.lines:
			if line[LINE_NUMBER] >= lineFrom and line[LINE_NUMBER] < lineTo:
				linesToReturn.append(line)

		return linesToReturn

	def getLinesBySpeakers(self):
		"""Get all parsed utterance grouped by speaker

		Returns:
			list: Utterances. Access data using the LINE constants
		"""
		linesBySpeakers = {}

		for line in self.lines:
			if not line[LINE_SPEAKER] in linesBySpeakers:
				linesBySpeakers[ line[LINE_SPEAKER] ] = []

			linesBySpeakers[ line[LINE_SPEAKER] ].append(line)

		return linesBySpeakers

	def getSpeakers(self):
		"""Get all speakers involved in this transcription

		Returns:
			list: Speakers
		"""
		return self.speakers

	def setLanguage(self, lang=None):
		"""Set language for this transcription. It will be used for processing light verbs

		Args:
			lang (string, optional): Use LANGUAGE constant or None for parsing it from the CHA file. Defaults to None.

		Raises:
			FileNotFoundError: -
		"""
		if lang == None:
			LANGUAGE_RE = "@Languages:.(.+)"

			if not os.path.isfile(self.chaFilePath):
				raise FileNotFoundError()

			with open(self.chaFilePath, "r", encoding="utf-8") as f:
				loop = True
				while loop:
					line = f.readline()
					if line:
						x = re.match(LANGUAGE_RE, line)
						if x is not None:
							language = x.group(1)
							if language in [LANGUAGE_SPANISH, LANGUAGE_ENGLISH]:
								self.language = language

							loop = False
					else:
						loop = False
		else:
			self.language = lang

		if self.language is None:
			# log.log("Warning: no language found")
			pass
	def getLanguage(self):
		"""Return current language

		Returns:
			string: Language
		"""
		return self.language

	def processMorToWords(self):
		"""Adds a new field to each line containing a clean version
		of the utterance that maps one to one with MOR tier
		"""
		for l in self.getLines():
			self.processMorToWordsInLine(l)

	def processMorToWordsInLine(self,line):
		"""Adds a new field to line containing a clean version
		of the utterance that maps one to one with MOR tier

		Args:
			line (dict): The line to process
		"""
		utt = line[LINE_UTTERANCE]

		# remove everything if utterance is in a foreign language
		if re.match(r"^\[\- .*?\]", utt):
			utt = ""

		# remove [+ TARGET]		
		utt = re.sub(r"\[\+.*\]", "", utt)

		# remove words with @ and : (usually for foreign words inside a utt)
		utt = re.sub(r"[\w'`´]*?@\w*?:\w*", "", utt)

		# deletes all <slang_word> [: some_replacing_word] and keeps some_replacing_word
		replacingRegEx = r"(?:\<&?=?[\w\s'@,\-&]*\>|\w*)\s*?\[\: ([\w\s'@,-]*)\]"
		prog = re.compile(replacingRegEx)
		while prog.search(utt):
			m = prog.search(utt)

			replaceBy = ""
			if m.group(1):
				replaceBy = m.group(1)

			utt = re.sub(replacingRegEx, replaceBy, utt, count=1)

		# deletes all [=! some_comment] and keeps whats inside <>
		# i.e <you get your dog> [=! imitates] => you get your dog
		# replacingRegEx = r"(?:\<([^\<\>]*)\>|(\w+))\s*?\[=!\s[^\<\>]*\]"
		replacingRegEx = r"(?:\<(&?=?[\w\s'@,\-&\(\)]*)\>|(\S*))\s*?\[=!\s[\s\w]*\]"
		prog = re.compile(replacingRegEx)
		while prog.search(utt):
			m = prog.search(utt)
			replaceBy = ""

			if m.group(1):
				replaceBy = m.group(1)
			elif m.group(2):
				replaceBy = m.group(2)

			utt = re.sub(replacingRegEx, replaceBy, utt, count=1)
		
		# remove []		
		utt = re.sub(r"\[[^\/]*?\]", "", utt)

		utt = utt.split(" ")

		# Remove words that starts with -
		utt = list(filter( lambda w: len(w) and w[0] != "-", utt ))

		# Commas are parsed by MOR
		uttWithCommas = []
		for w in utt:
			if "," in w and len(w) > 0:
				uttWithCommas.append( w[:w.find(",")] )
				uttWithCommas.append(",")
				
				extra = w[(w.find(",")+1):]
				if len(extra) > 0:
					uttWithCommas.append(extra)
			else:
				uttWithCommas.append(w)
		utt = uttWithCommas

		# "[/]" means repetition. This and next word won't be parsed by MOR
		while "[/]" in utt:
			i = utt.index("[/]")
			if i-1 >= 0:
				del utt[i-1:i+1]

		for stopWord in STOP_WORDS:
			while stopWord in utt:
				i = utt.index(stopWord)
				del utt[i]
		
		# WORD_XXX means transcriptor couldn't understand. This word won't be parsed by MOR
		while WORD_XXX in utt:
			i = utt.index(WORD_XXX)
			del utt[i]
		
		while "(.)" in utt:
			i = utt.index("(.)")
			del utt[i]

		# Only keep words that start with a letter or a number different to 0
		utt = list(filter( lambda w: len(w) and (w[0].isalpha() or w[0] in [",","'","("] or (w[0].isdigit() and w[0] != "0" )) , utt ))

		line[LINE_MOR_TO_WORDS] = utt

		if len(line[LINE_MOR_TO_WORDS]) != len(line[TIER_MOR]):
			print(f"MorToWord failed for line [{line[LINE_NUMBER]}]")
	def morUnitToWord(self, line, morUnitIndex):
		"""Returns the word from the utterance related to the MOR unit

		Args:
			line (dict): The line to process
			morUnitIndex (int): The index of the mor unit

		Returns:
			str: The word from the utterance related
		"""
		if not LINE_MOR_TO_WORDS in line:
			self.processMorToWordsInLine(line)
		
		return line[LINE_MOR_TO_WORDS][morUnitIndex]

	def countUtterances(self, ignoreEmptyUtterances = True):
		"""Returns number of utterances ignoring empty ones based on a word criteria

		Returns:
			int: Number of utterances in the current transcript
		"""

		uttCount = 0
		for l in self.getLines():
			if ignoreEmptyUtterances and self.isUtteranceEmpty(l):
				continue
			
			uttCount += 1

		return uttCount

	def countUtterancesByAddressee(self, ignoreEmptyUtterances = True):
		"""Returns number of utterances grouped by addressee ignoring empty ones based on a word criteria

		Returns:
			dict: Number of utterances by addressee
		"""
		addressees = {}

		for l in self.getLines():
			if ignoreEmptyUtterances and self.isUtteranceEmpty(l):
				continue

			addressee = l[LINE_ADDRESSEE]
			if addressee in addressees:
				addressees[addressee] += 1
			else:
				addressees[addressee] = 1

		return addressees

	def countWordsByAddressee(self):
		"""Count words grouped by addressee

		Returns:
			dict: Number of words grouped by addresee
		"""
		assert self.morFound, "MOR tier not found"
		addressees = {}

		for l in self.getLines():
			c = self.countWordsInLine(l)

			addressee = l[LINE_ADDRESSEE]
			if addressee in addressees:
				addressees[addressee] += c
			else:
				addressees[addressee] = c

		return addressees

	def countWordsInLine(self, line):
		"""Number of words in the utterance

		Args:
			line (dict): A parsed utterance from getLines()

		Returns:
			int: Number of words in the utterance
		"""
		assert self.morFound, "MOR tier not found"
		dontCount = ["cm", "?"] 

		c = 0
		for morUnit in line[TIER_MOR]:
			if morUnit != MISSING_VALUE and morUnit[MOR_UNIT_CATEGORIA] not in dontCount:
				c += 1

		return c

	def countNounsByAddressee(self):
		"""Number of nouns grouped by addressee

		Returns:
			dict: Addressees and number of nouns
		"""
		self.populateNouns()

		addressees = {}

		for l in self.getLines():
			c = len( l[LINE_NOUNS] )

			addressee = l[LINE_ADDRESSEE]
			if addressee in addressees:
				addressees[addressee] += c
			else:
				addressees[addressee] = c

		return addressees

	def getNounsInLine(self, linea):
		"""Returns a list of indexes of nouns in the MOR tier

		Args:
			linea (dict): Utterance

		Returns:
			list: List of indexes
		"""
		assert self.morFound, "MOR tier not found"
		
		if not self.processedVerbs:
			#in english, sometimes MOR mark as noun a word that is a verb
			self.populateVerbs()

		nouns = []

		if linea[TIER_MOR] != MISSING_VALUE: 	
			for i, morUnit in enumerate(linea[TIER_MOR]):
				if morUnit[MOR_UNIT_CATEGORIA] in CATEGORIAS_SUSTANTIVOS:
					if not i in linea[LINE_VERBS]: #in english, sometimes MOR mark as noun a word that is a verb
						nouns.append(i)
		
		return nouns

	def populateNouns(self):
		"""Populate LINE_NOUNS for every line with the indexes of the MOR line where nouns are found
		"""
		assert self.morFound, "MOR tier not found"

		if self.processedNouns: return
		
		if not self.processedVerbs:
			#in english, sometimes MOR mark as noun a word that is a verb
			self.populateVerbs()

		for linea in self.getLines():
			linea[LINE_NOUNS] = self.getNounsInLine(linea)
		
		self.processedNouns = True

	def countAdjectivesByAddressee(self):
		"""Number of adjectives grouped by addressee

		Returns:
			dict: Addressees and number of adjectives
		"""
		if not self.processedAdjectives:
			self.populateAdjectives()

		addressees = {}

		for l in self.getLines():
			c = len(l[LINE_ADJECTIVES])

			addressee = l[LINE_ADDRESSEE]
			if addressee in addressees:
				addressees[addressee] += c
			else:
				addressees[addressee] = c

		return addressees
	def getAdjectivesInLine(self, linea):
		"""Returns a list of indexes of adjectives in the MOR tier

		Args:
			linea (dict): Utterance

		Returns:
			list: List of indexes
		"""
		assert self.morFound, "MOR tier not found"
		adjetivos = []

		if linea[TIER_MOR] != MISSING_VALUE: 	
			for i, morUnit in enumerate(linea[TIER_MOR]):
				if morUnit[MOR_UNIT_CATEGORIA] in CATEGORIAS_ADJETIVOS:
					adjetivos.append(i)
				elif (	self.language == LANGUAGE_SPANISH and i>0 and 
						morUnit[MOR_UNIT_CATEGORIA] == "part" and
						linea[TIER_MOR][i-1][MOR_UNIT_CATEGORIA] == "cop"):
					adjetivos.append(i)

				# en español: si es part y el anterior es cop y no termina en [ando, endo] es adjetivo
		 
		return adjetivos
	
	def populateAdjectives(self):
		"""Populate LINE_ADJECTIVES from each line with the indexes of the MOR tier where adjectives are found
		"""
		assert self.morFound, "MOR tier not found"
		
		if self.processedAdjectives: return

		for linea in self.getLines():
			linea[LINE_ADJECTIVES] = self.getAdjectivesInLine(linea)
		
		self.processedAdjectives = True

	def populateVerbs(self, countCopAux = False, processLightVerbs = True):
		"""Populate LINE_VERBS for every line with the indexes of the MOR tier where verbs are found

		Args:
			countCopAux (bool, optional): Should we count cop and aux as verbs ?. Defaults to False.
			processLightVerbs (bool, optional): Should we process light verbs. Defaults to True.
		"""
		if self.processedVerbs: return

		for l in self.getLines():
			self.getVerbsInLine(l, countCopAux, processLightVerbs)
		
		self.processedVerbs = True

	def _processLightVerbs(self, lineaMor, verbos):
		"""Internal use. Language dependent. Process light verbs to be skipped when counting verbs

		Args:
			lineaMor (dict): MOR tier for an utterance. Note that it is a dict, not a list
			verbos (list): List of verbs

		Returns:
			list: MOR tier without light verbs
		"""
		if self.language == LANGUAGE_SPANISH:
			# V1 SÍ AUX: ir a, tener que, poder, haber, estar, dejar de, deber, acabar de,
			# terminar de, haber que, estar por, empezar a, empezar por, comenzar a, poner a, volver a

			#1) Cuando es auxiliar + infinitivos, contar el infinitivo
			#2) Cuando es auxiliar + gerundio, contar el gerundio
			#3) Cuando es auxiliar + participio, contar el participio

			raices = [
				[["i"],["a"]],
				[["tene"],["que"]],
				[["pode"]],
				[["habe"]],
				[["esta"]],
				[["debe"]],
				[["habe"],["que"]],
				[["esta"],["por"]],
				
				#[["deja"],["de"]],    #estos que comenté son los que cambiamos cuando hicimos aclew completo
				
				#[["acaba"],["de"]],
				#[["termina"],["de"]],

				#[["empeza"],["a"]],
				#[["empeza"],["por"]],

				#[["comienzo"],["a"]],
				#[["comenza"],["a"]],

				#[["pone"],["a"]],
				#[["volve"],["a"]]
			]

			siguiente_palabra_de_auxiliar = [ "inf", "ger", "part" ]

			#4) Cuando es copulativo + gerundio contar el gerundio
			#5) Cuando es copulativo + participio contar el participio.

			criteriosCategoria = [
					[ ["cop"],["ger","part"] ]
			]

			#6) No contar el copula o auxiliar si está solo (1versión) / contar el cópula o el auxiliar si esta sólo (2 versión)
			#countCopAux = False # Hago los dos mas abajo

			#7) En los casos de 2 o 3 Verbos (verbos conjugado, infinitivo, gerundio o participio), contar el que no es ni auxiliar, ni copula.
			#estoy contando frase verbal + verbo final

			#8) si hay 2 o más verbos conjugados coordinados en una emisión se toman todos.

			#1) 2) y 3) auxiliares    aux + (inf,ger,part)

			for raiz in raices:
				criteriaType = [ MOR_UNIT_LEXEMA for c in raiz ]
				criteria = raiz + [siguiente_palabra_de_auxiliar]
				criteriaType.append(MOR_UNIT_CATEGORIA)
				morIndexes = self._checkCriteria( list(lineaMor.values()), criteria, criteriaType )
				# existe el patron
				while len(morIndexes) > 0:
					#agarro la última palabra y la agrego como verbo
					trueIndex = list(lineaMor.keys())[ morIndexes[-1] ]
					verbos.append( trueIndex )

					trueIndexesToDelete = []
					for morIndex in morIndexes:
						trueIndexesToDelete.append( list(lineaMor.keys())[ morIndex ] )
					for i in trueIndexesToDelete:
						del lineaMor[ i ]

					criteriaType = [ MOR_UNIT_LEXEMA for c in raiz ]
					criteria = raiz + [siguiente_palabra_de_auxiliar]
					criteriaType.append(MOR_UNIT_CATEGORIA)
					morIndexes = self._checkCriteria( list(lineaMor.values()), criteria, criteriaType )

			#cop + (ger | part)
			for criterio in criteriosCategoria:
				morIndexes = self._checkCriteria( list(lineaMor.values()), criterio, MOR_UNIT_CATEGORIA )
				while len(morIndexes) > 0:
					# aca habría que mirar si el segundo (trueIndex) termina en ando o endo, en tal caso
					# se hace como se está haciendo, en caso contrario habría que retirarlo, es adjetivo,
					# entonces también habría que tocar la parte de adjetivos 
					trueIndex = list(lineaMor.keys())[ morIndexes[1] ]
					verbos.append( trueIndex )

					trueIndexesToDelete = []
					for morIndex in morIndexes:
						trueIndexesToDelete.append( list(lineaMor.keys())[ morIndex ] )
					for i in trueIndexesToDelete:
						del lineaMor[ i ]

					morIndexes = self._checkCriteria( list(lineaMor.values()), criterio, MOR_UNIT_CATEGORIA )
		elif self.language == LANGUAGE_ENGLISH:		
			criterias = [
				[["part|go"], ["to"], ["n", *CATEGORIAS_VERBOS]],
				[["part|go"], ["n", *CATEGORIAS_VERBOS]], #gonna
				[["go"], [*CATEGORIAS_VERBOS]],
				[["have"], ["to"], [*CATEGORIAS_VERBOS]],
				[["do"], [*CATEGORIAS_VERBOS]],
				[["do"],["not"],[*CATEGORIAS_VERBOS]],
				[["use"], ["to"], [*CATEGORIAS_VERBOS]],



				# [["like"], ["to"], [*CATEGORIAS_VERBOS]], # se decidió no agregarla
				# [["want"], ["to"], [*CATEGORIAS_VERBOS]], # se decidió no agregarla
				# [["try"], ["to"], [*CATEGORIAS_VERBOS]], # se decidió no agregarla
			]

			for criteria in criterias:
				criteriaType = [ MOR_UNIT_LEXEMA if "|" not in c[0] else MOR_UNIT_CATEGORIA_LEXEMA for c in criteria ]
				del criteriaType[-1]
				criteriaType.append(MOR_UNIT_CATEGORIA)

				morIndexes = self._checkCriteria( list(lineaMor.values()), criteria, criteriaType )
				# existe el patron
				while len(morIndexes) > 0:
					#agarro la última palabra y la agrego como verbo
					trueIndex = list(lineaMor.keys())[ morIndexes[-1] ]
					verbos.append( trueIndex )

					trueIndexesToDelete = []
					for morIndex in morIndexes:
						trueIndexesToDelete.append( list(lineaMor.keys())[ morIndex ] )
					for i in trueIndexesToDelete:
						del lineaMor[ i ]

					criteriaType = [ MOR_UNIT_LEXEMA if "|" not in c else MOR_UNIT_CATEGORIA_LEXEMA for c in criteria ]
					del criteriaType[-1]
					criteriaType.append(MOR_UNIT_CATEGORIA)
					morIndexes = self._checkCriteria( list(lineaMor.values()), criteria, criteriaType )
			
			# Don't count LET'S as a verb
			stopWords = [
				{MOR_UNIT_CATEGORIA: 'v', MOR_UNIT_LEXEMA: 'let', MOR_UNIT_EXTRA: '~pro:obj|us'}
			]

			trueIndexesToDelete = []
			for i in lineaMor:
				morUnit = lineaMor[i]
				if morUnit in stopWords:
					trueIndexesToDelete.append(i)
			for i in trueIndexesToDelete:
				del lineaMor[ i ]
			
		return lineaMor

	def getVerbsInLine(self, linea, countCopAux = False, processLightVerbs = True ):
		"""Gets verbs in line and store them in LINE_VERBS

		Args:
			linea (dict): Utterance from getLines()
			countCopAux (bool, optional): Should we count cop and aux. Defaults to False.
			processLightVerbs (bool, optional): Should we skip light verbs. Defaults to True.

		Returns:
			list: Array of indexes to verbs
		"""
		verbos = []
		# no se borra nada del MOR original
		lineaMor={}
		for i, m in enumerate(linea[TIER_MOR]):
			lineaMor[i] = m

		if processLightVerbs:
			assert self.language != None, "language not set"
			lineaMor = self._processLightVerbs(lineaMor, verbos)

		if len(lineaMor) != len(linea[TIER_MOR]):
			linea[LINE_LIGHT_VERBS] = True

		#verbos normales
		verbosIndividualesAContar = CATEGORIAS_VERBOS.copy()

		if not countCopAux:
			verbosIndividualesAContar.remove("cop")
			verbosIndividualesAContar.remove("aux")
		
		if self.language == LANGUAGE_ENGLISH:
			verbosIndividualesAContar.remove("inf") #esto saca el to

		morIndexes = self._checkCriteria( list(lineaMor.values()), [ verbosIndividualesAContar ], MOR_UNIT_CATEGORIA )
		while len(morIndexes) > 0:
			trueIndex = list(lineaMor.keys())[ morIndexes[0] ]
			verbos.append( trueIndex )

			trueIndexesToDelete = []
			for morIndex in morIndexes:
				trueIndexesToDelete.append( list(lineaMor.keys())[ morIndex ] )
			for i in trueIndexesToDelete:
				del lineaMor[ i ]

			morIndexes = self._checkCriteria( list(lineaMor.values()), [ verbosIndividualesAContar ], MOR_UNIT_CATEGORIA )

		linea[LINE_VERBS] = verbos
		linea[LINE_VERBS].sort()
		
		return verbos

	def countVerbsByAddressee(self, countCopAux = False, processLightVerbs = True):
		"""Number of verbs grouped by addressee
		Args:
			countCopAux (bool, optional): Should we count cop and aux. Defaults to False.
			processLightVerbs (bool, optional): Should we skip light verbs. Defaults to True.
		Returns:
			dict: Addressees and number of verbs
		"""
		addressees = {}

		if not self.processedVerbs:
			self.populateVerbs(countCopAux=countCopAux, processLightVerbs=processLightVerbs)

		for l in self.getLines():
			c = len( l[LINE_VERBS] )

			addressee = l[LINE_ADDRESSEE]
			if addressee in addressees:
				addressees[addressee] += c
			else:
				addressees[addressee] = c

		return addressees
	
	def count(self, what=LINE_UTTERANCE, addressee=ADDRESSEE_ALL, countType=COUNT_TYPE_TOKENS, countCopAux = False, processLightVerbs = True):
		"""Count tokens or types for any word or for nouns, verbs or adjectives

		Args:
			what ([type]): LINE_VERBS, LINE_NOUNS, LINE_ADJECTIVES or LINE_UTTERANCE for all kinds of lexical categories
			addressee ([type], optional): ADDRESSEE_ALL, ADDRESSEE_CHILD_DIRECTED, ADDRESSEE_OVER_HEARD, ADDRESSEE_CHILD_PRODUCED or ADDRESSEE_ADULT. Defaults to ADDRESSEE_ALL.
			countType ([type], optional): COUNT_TYPE_TOKENS or COUNT_TYPE_TYPES. Defaults to COUNT_TYPE_TOKENS.
			countCopAux (bool, optional): Should we count cop and aux as verbs ?. Defaults to False.
			processLightVerbs (bool, optional): Should we process light verbs. Defaults to True.
			
		Returns:
			int: count
		"""

		if what not in [ LINE_VERBS, LINE_NOUNS, LINE_ADJECTIVES, LINE_UTTERANCE ]:
			raise Exception("'what' argument should be LINE_VERBS, LINE_NOUNS or LINE_ADJECTIVES or LINE_UTTERANCE")

		if what == LINE_VERBS:
			self.populateVerbs(countCopAux=countCopAux, processLightVerbs=processLightVerbs)
		elif what == LINE_NOUNS:
			self.populateNouns()
		elif what == LINE_ADJECTIVES:
			self.populateAdjectives()
		
		c = {}
		
		def add(l):
			nonlocal c

			if what != LINE_UTTERANCE:
				for index in l[what]:
					v = l[TIER_MOR][index][MOR_UNIT_LEXEMA]
					if v in c:
						c[v] += 1
					else:
						c[v] = 1
			else:
				for morUnit in l[TIER_MOR]:
					v = morUnit[MOR_UNIT_CATEGORIA] + "|" + morUnit[MOR_UNIT_LEXEMA] + morUnit[MOR_UNIT_EXTRA]
					if v in c:
						c[v] += 1
					else:
						c[v] = 1
		
		if addressee == ADDRESSEE_ALL:
			for l in self.getLines():
				add( l )
		elif addressee == ADDRESSEE_CHILD_DIRECTED:
			for l in self.getLines():
				if l[LINE_ADDRESSEE] == SPEAKER_TARGET_CHILD:
					add( l )
		elif addressee == ADDRESSEE_CHILD_PRODUCED:
			for l in self.getLines():
				if l[LINE_SPEAKER] == SPEAKER_TARGET_CHILD:
					add( l )
		elif addressee == ADDRESSEE_OVER_HEARD:
			for l in self.getLines():
				if l[LINE_ADDRESSEE] != SPEAKER_TARGET_CHILD and l[LINE_SPEAKER] != SPEAKER_TARGET_CHILD :
					add( l )
		elif addressee == ADDRESSEE_ADULT:
			for l in self.getLines():
				if l[LINE_ADDRESSEE] == SPEAKER_ADULT and l[LINE_SPEAKER] != SPEAKER_TARGET_CHILD :
					add( l )
		
		# print(c)

		if countType == COUNT_TYPE_TOKENS:
			return sum(c.values())
		else:
			return len(c.keys())

	def findLinesByMorCriteria(self, criteria, criteriaType=MOR_UNIT_CATEGORIA):
		"""Finds utterances that follow a criteria

		Args:
			criteria (list): Criteria. Example: [ ["part_of_speech-1"], [ [ "part_of_speech-1","part_of_speech-2 ] ] ]. 
			An utterance will match if it contains two adjacent words: one with "part_of_speech-1" and the second one "part_of_speech-1" or "part_of_speech-2"
			criteriaType (str or list, optional): Search for part-of-speech (categoria léxica) or lexeme. If using a list it must have the same number of elements as criteria. Defaults to MOR_UNIT_CATEGORIA. MOR_UNIT_CATEGORIA_LEXEMA matches "<part_of_speech>|<lexeme>"

		Returns:
			list: List of dict with line and matched criteria
		"""
		results = []

		assert self.morFound, "MOR tier not found"
		assert isinstance( criteria, list ) and len(criteria) > 0 and isinstance( criteria[0], list ), "invalid criteria, expected: [ [], ...]"

		for line in self.lines:
			if "mor" in line:
				matchedCriteria = self._checkCriteria(line["mor"], criteria, criteriaType)
				if len(matchedCriteria) > 0:
					result = {
						"line" : line,
						"matchedCriteria" : []
					}

					for morUnitIndex in matchedCriteria:
						result["matchedCriteria"].append( line["mor"][morUnitIndex] )

					results.append(result)
		return results

	def applyMorCriteriaInLine(self, line, criteria, criteriaType = MOR_UNIT_CATEGORIA):
		"""Same as findLinesByMorCriteria but for one line

		Args:
			line (dict): Utterance
			criteria (list): [description]
			criteriaType (str or list, optional): [description]. Defaults to MOR_UNIT_CATEGORIA.

		Returns:
			list: List of indexes of the words matching the criteria
		"""
		return self._checkCriteria( line["mor"], criteria, criteriaType )

	def getLexicalDiversity(self, addressee=ADDRESSEE_ALL, metric=LEXICAL_DIVERSITY_HDD, extraParam = None):
		lines = []

		if addressee == ADDRESSEE_ALL:
			lines = self.getLines()
		elif addressee == ADDRESSEE_CHILD_DIRECTED:
			for l in self.getLines():
				if l[LINE_ADDRESSEE] == SPEAKER_TARGET_CHILD:
					lines.append(l)
		elif addressee == ADDRESSEE_CHILD_PRODUCED:
			for l in self.getLines():
				if l[LINE_SPEAKER] == SPEAKER_TARGET_CHILD:
					lines.append(l)
		elif addressee == ADDRESSEE_OVER_HEARD:
			for l in self.getLines():
				if l[LINE_ADDRESSEE] != SPEAKER_TARGET_CHILD and l[LINE_SPEAKER] != SPEAKER_TARGET_CHILD :
					lines.append(l)
		
		tokens = []
		for l in lines:
			token = ""
			for morUnit in l[TIER_MOR]:
				token += morUnit[MOR_UNIT_CATEGORIA] + "|" + morUnit[MOR_UNIT_LEXEMA] + morUnit[MOR_UNIT_EXTRA]
			tokens.append(token)
		
		result = -1
		if metric == LEXICAL_DIVERSITY_HDD:
			result = ld.hdd(tokens)
		elif metric == LEXICAL_DIVERSITY_MAAS:
			result = ld.maas_ttr(tokens)
		elif metric == LEXICAL_DIVERSITY_MTLD:
			result = ld.mtld(tokens)
		elif metric == LEXICAL_DIVERSITY_MATTR:
			if extraParam == None:
				extraParam = 50 #default window_size 50
			result = ld.mattr(tokens, extraParam) 
		elif metric == LEXICAL_DIVERSITY_TTR:
			result = ld.ttr(tokens)
		
		return result
	
	def isUtteranceEmpty(self, line):
		"""Returns True if the utterance is empty based on a word criteria

		Args:
			line (dict): A line

		Returns:
			bool: True if the utterance is considered empty. False otherwise
		"""

		if len(line[TIER_MOR]) == 0:
			considerEmptyWords = []

			if self.language == LANGUAGE_SPANISH:
				considerEmptyWords = [
					"ríe",
					"rie",
					"llora",
					"besa",
					"tose",
					"silba",
					"silva",
					"aplaude",
					"camina",
					"llorisquea",
					"chasquea",
					"sopla",
					"lloriqueo",
					"zapatea",
					"bosteza",
					"suspira"
				]
			elif self.language == LANGUAGE_ENGLISH:
				considerEmptyWords = [
					"laughs",
					"whistles",
					"cries",
					"sobs",
					"giggles",
					"chuckles",
					"whines",
					"yawns",
					"squeals",
					"kisses",
					"toots",
					"claps",
					"blowskisses",
					"wagglestongue",
					"clickstongue",
					"clicks",
					"sighs"
				]

			for w in considerEmptyWords:
				if w in line[LINE_UTTERANCE]:
					return True

			emptyUtt = line[LINE_UTTERANCE].replace("0", "")
			emptyUtt = emptyUtt.replace(".", "")
			emptyUtt = emptyUtt.strip()
			if emptyUtt == "":
				return True

			return False
		
		return False
	
	def getTurnsBySpeaker( self, addressee, allowIntervining = True ):
		"""Get utterances grouped by speakers turns

		Args:
			addressee (str): ADDRESSEE_CHILD_DIRECTED or ADDRESSEE_ADULT
			allowIntervining (bool, optional): Only for ADDRESSEE_ADULT, allow intervening utterances. Defaults to True.

		Returns:
			dict: Utterances grouped by speakers turns
		"""

		TURN_CDS_MAX_INTERVENING_CHILD = 1 # cuantas intervenciones del CHI
		TURN_CDS_MAX_INTERVENING_OTHER = 3 # cuantas intervenciones de otros hablantes
		TURN_ADS_MAX_INTERVENING_OTHER = 3 # cuantas intervenciones de otros hablantes
		TURN_MAX_TIME = 5000 #ms 

		speakers = self.getSpeakers()

		def endTurno():
			nonlocal turnos, turno, speaker
			nonlocal qtyIntervencionChild, qtyIntervencionOther
			
			turnos[speaker].append(turno)
			turno = []

			qtyIntervencionChild = 0
			qtyIntervencionOther = 0

		turnos = {}

		if addressee == ADDRESSEE_CHILD_DIRECTED: ### Turnos dirigidos a target child
			target = SPEAKER_TARGET_CHILD

			for speaker in speakers:
				if speaker in [SPEAKER_CODE] :
					continue

				turnos[speaker] = []
				turno = []

				qtyIntervencionChild = 0
				qtyIntervencionOther = 0
				
				for l in self.getLines():
					if self.isUtteranceEmpty(l):
						continue

					if not turno: # nuevo turno
						if l[LINE_SPEAKER] == speaker and l[LINE_ADDRESSEE] == target:
							turno.append(l)
					else: # turno existente
						tiempoActual = tiempoAnterior = None

						if LINE_BULLET in l and LINE_BULLET in turno[-1]:
							tiempoActual = l[LINE_BULLET][0]
							tiempoAnterior = turno[-1][LINE_BULLET][1]
						
						if tiempoActual and (tiempoActual - tiempoAnterior >= TURN_MAX_TIME):
							endTurno()
						else:
							if l[LINE_SPEAKER] == speaker: #el hablante es correcto
								if l[LINE_ADDRESSEE] == target:
									turno.append(l)
								else:
									endTurno()
							else: #el hablante cambió
								if l[LINE_SPEAKER] == SPEAKER_TARGET_CHILD:
									qtyIntervencionChild += 1
									if qtyIntervencionChild > TURN_CDS_MAX_INTERVENING_CHILD:
										endTurno()
								else:
									qtyIntervencionOther += 1
									if qtyIntervencionOther > TURN_CDS_MAX_INTERVENING_OTHER:
										endTurno()
		
		elif addressee == ADDRESSEE_ADULT: ### Turnos entre adultxs
			target = SPEAKER_ADULT

			for speaker in speakers:
				if speaker in [SPEAKER_TARGET_CHILD, SPEAKER_OTHER_CHILD, SPEAKER_CODE] :
					continue

				turnos[speaker] = []
				turno = []

				qtyIntervencionOther = 0
				
				for l in self.getLines():
					if self.isUtteranceEmpty(l):
						continue

					if not turno: # nuevo turno
						if l[LINE_SPEAKER] == speaker and l[LINE_ADDRESSEE] == target:
							turno.append(l)
					else: # turno existente
						tiempoActual = tiempoAnterior = None

						if LINE_BULLET in l and LINE_BULLET in turno[-1]:
							tiempoActual = l[LINE_BULLET][0]
							tiempoAnterior = turno[-1][LINE_BULLET][1]
						
						if tiempoActual and (tiempoActual - tiempoAnterior >= TURN_MAX_TIME):
							endTurno()
						else:
							if l[LINE_SPEAKER] == speaker: #el hablante es correcto
								if l[LINE_ADDRESSEE] == target: #si habla con el chico u otro se corta el turno
									turno.append(l)
								else:
									endTurno()
							else: #el hablante cambió (sea el chico o no)
								if allowIntervining:
									qtyIntervencionOther += 1
									if qtyIntervencionOther > TURN_ADS_MAX_INTERVENING_OTHER:
										endTurno()
								else:
									endTurno()
		
		else:
			raise Exception("'addressee' argument should be ADDRESSEE_CHILD_DIRECTED or ADDRESSEE_ADULT")
						
		return turnos
	
	def countTurns( self, addressee, allowIntervining = True ):
		"""Count turns by addressee

		Args:
			addressee (str): ADDRESSEE_CHILD_DIRECTED or ADDRESSEE_ADULT
			allowIntervining (bool, optional): Only for ADDRESSEE_ADULT, allow intervening utterances. Defaults to True.

		Returns:
			int: Turn count
		"""
		turns = self.getTurnsBySpeaker(addressee, allowIntervining)
		count = 0

		for speaker in turns:
			count += len(turns[speaker])

		return count

	def _checkCriteria(self, mor, criteria, criteriaType):
		"""Internal use. Checks the MOR tier for the criteria (just once)

		Args:
			mor (list): MOR tier
			criteria (list): Same as findLinesByMorCriteria
			criteriaType (str or list): Same as findLinesByMorCriteria

		Returns:
			list: List of indexes of the words matching the criteria
		"""
		if mor == MISSING_VALUE:
			return []

		matchedCriteria = []
		currentCriteria = 0
		found = False

		for morUnitIndex in range(len(mor)):
			morUnit = mor[morUnitIndex]

			if morUnit == MISSING_VALUE:
				continue

			found = False
			for c in criteria[currentCriteria]:
				isAMatch = False
				currentCriteriaType = criteriaType

				if isinstance(criteriaType, list):
					currentCriteriaType = criteriaType[currentCriteria]
				
				if currentCriteriaType == MOR_UNIT_CATEGORIA_LEXEMA:
					if not "|" in c:
						raise Exception("Criteria type is MOR_UNIT_CATEGORIA_LEXEMA but criteria doesn't include | symbol")
					
					arrCriteria = c.split("|")

					isAMatch = morUnit[ MOR_UNIT_LEXEMA ] == arrCriteria[1] and morUnit[ MOR_UNIT_CATEGORIA ] == arrCriteria[0]
				else:
					isAMatch = morUnit[ currentCriteriaType ] == c
					
				if isAMatch:
					matchedCriteria.append(morUnitIndex)
					currentCriteria += 1
					found = True
					break

			if not found and currentCriteria > 0:
				matchedCriteria = []
				currentCriteria = 0


			if currentCriteria == len(criteria):
				return matchedCriteria

		return []
	
	def _parseMor(self, morContent, lineNumber):
		"""Internal use. Parse MOR tier

		Args:
			morContent (string): Content of the MOR tier
			lineNumber (int): Line number

		Returns:
			list: Parsed MOR tier
		"""
		morContent = morContent.split(" ")

		arrMorData = []

		for morUnit in morContent:
			lstMorUnit = []

			if "^" in morUnit:
				if lineNumber not in self.morAmbiguousLines:
					self.morAmbiguousLines.append(lineNumber)
				
				lstMorUnit = morUnit.split("^")
				morUnit = lstMorUnit[0]
				lstMorUnit = lstMorUnit[1:]

			parsedMorUnit = self._parseMorUnit(morUnit)
			if parsedMorUnit != {}:
				if len(lstMorUnit) > 0 :
					parsedMorUnit[MOR_UNIT_AMBIGUOUS] = lstMorUnit

				arrMorData.append( parsedMorUnit )

		return arrMorData

	def _parseMorUnit(self, morUnit):
		"""Internal use. Parse a MOR unit (one word)

		Args:
			morUnit (string): One word as described by MOR

		Returns:
			dict: A dict representation of the MOR unit
		"""
		if morUnit in MOR_STOP_WORDS:
			return {}
		
		MOR_REGEX = r"([A-zÀ-ú:#\?']*)\|([A-zÀ-ú]*)(.*)"

		matches = re.match(MOR_REGEX, morUnit)
		if matches != None: #no agarra ni . ! ? 
			if len(matches.groups()) == 3:
				morCategoria = matches.group(1)
				morLexema = matches.group(2)
				morExtra = matches.group(3)

				for stopWord in MOR_STOP_WORDS:
					if type(stopWord) is list:
						if morLexema == stopWord[1] and morCategoria == stopWord[0]:
							if len(stopWord) == 2:
								return {}
							else:
								if stopWord[2] in morExtra:
									return {}
					else:
						if morLexema == stopWord:
							return {}

				#reemplazo de palabras que está agarrando mal el MOR
				if morLexema in MOR_REPLACEMENTS:
					morLexema = MOR_REPLACEMENTS[morLexema]

				parsedMorUnit = {
					MOR_UNIT_CATEGORIA : morCategoria,
					MOR_UNIT_LEXEMA : morLexema,
					MOR_UNIT_EXTRA : morExtra
				}

				return parsedMorUnit
			else:
				log.log(f"Warning: Malformed mor unit \"{morUnit}\"")
				return {}
		else:
			return {}

	def _setAddressee(self, line):
		"""Internal use. Set normalized addressee 

		Args:
			line (dict): Utterance
		"""
		addressee = SPEAKER_ADULT

		if not TIER_XDS in line:
			# This is the way we usually do it. e.g. +CHI
			for target in [ SPEAKER_TARGET_CHILD, SPEAKER_OTHER_CHILD ]:
				tag = ADDRESSEE_TAG % target
				if tag in line[LINE_UTTERANCE]:
					addressee = target
					line[LINE_UTTERANCE] = line[LINE_UTTERANCE].replace(tag, "").strip()
		elif line[TIER_XDS] != MISSING_VALUE:
			if line[TIER_XDS] in ADDRESSEE_CORRESPOND:
				addressee = ADDRESSEE_CORRESPOND[ line[TIER_XDS] ]
			else:
				log.log(f"Warning unknown addressee '{line[TIER_XDS]}' in line '{line[LINE_NUMBER]}'")
				addressee = ADDRESSEE_XDS_UNKNOWN

		line[LINE_ADDRESSEE] = addressee

		# This was the way elan2cha worked. Leaving this for compatibility
		# if ADDRESSEE_ALT_TAG in strLine:
		# 	for target in [ ADDRESSEE_ALT_TAG_ADULT,
		# 					ADDRESSEE_ALT_TAG_CHILD,
		# 					ADDRESSEE_ALT_TAG_OTHER,
		# 					ADDRESSEE_ALT_TAG_TARGET_CHILD,
		# 					ADDRESSEE_ALT_TAG_UNKNOWN]:
		# 		tag = ADDRESSEE_ALT_TAG_TEMPLATE % target
		# 		if tag in strLine:
		# 			if not USE_ALT_TARGET_CHILD:
		# 				addressee = ADDRESSEE_CORRESPOND[target]
		# 			else:
		# 				addressee = ADDRESSEE_CORRESPOND_ALT[target]
		#
		# 			strLine = strLine.replace(tag, "").strip()
		#
		# return addressee, strLine
