"""
MANUAL https://talkbank.org/manuals/CHAT.html
"""
from unidecode import unidecode
from .utils_preprocess import *


def remove_special_utterance_terminators(text, replace=" "):
	"""
	MANUAL 9.11  Special Utterance Terminators
	In addition to the three basic utterance terminators, CHAT provides a
	series of more complex utterance terminators to mark various special
	functions.
	These special terminators all begin with the + symbol and end with
	one of the three basic utterance terminators.
	"""
	special_utterance_terminators = [
		"+…", # Trailing Off
		"+..?", # Trailing Off of a Question
		"+!?", # Question With Exclamation
		"+/.", # Interruption
		"+/?", # Interruption of a Question
		"+//.", # Self-Interruption
		"+//?", # Self-Interrupted Question
		"+.", # Transcription Break
		'+"/.', # Quotation Follows
		'+".', # Quotation Precedes
		'+"', # Quoted Utterance
		"+^", # Quick Uptake
		"+,", # Self Completion
		"++", # Other Completion
	]
	text = regex_replace_substr(text, special_utterance_terminators, replace=replace)
	return text


NEW_TAGS = {
	"non-verbal": "✦non verbal✦",# "mhh", "mhm", "mh", "hm", "um", "umph", "euhm", "uhuh", "uhu", "huh", "uh",
	"laugh": "✦laugh✦",# hhh**
	"unintelligible": "✦unintelligible✦", # Unintelligible words with an unclear phonetic shape should be transcribed as xxx.
	"special-character": "✦special character✦", # represents speech intonation such as shift to high pitch, falling to mid etc.
	"pause": "✦pause✦", # well pause like pause pause
	"bracket": "✦bracket✦", # many of the above and below things (events, interposed words) are first annotated with brackets!
	"special-utterance-terminator": "✦special utterance terminator✦",# see above for what it means (Trailing off, Questions with Exclamation, Interruption of a Question etc).
	"event": "✦event✦", # A list of these Simple Events appears in the CHAT manual and includes cough, groan, sneeze, mumble etc.
	"interposed-word": "✦interposed word✦", # , such as “yeah” or “mhm”, within a longer discourse from the speaker who has the floor without breaking up the utterance of the main speaker.
}
for k in NEW_TAGS:
	NEW_TAGS[k] = " " + NEW_TAGS[k] + " "


def preprocess_row(text,
	remove_tags=False,
	remove_beg_unfinished_word=False,
	lower=False,
	remove_accents=False):
	my_specific_delimiter = "✌️" # can be any character that is not in the original text

	# remove special utterance
	text = remove_special_utterance_terminators(text, replace=NEW_TAGS["special-utterance-terminator"])

	# clean space for preprocess
	text = space_text_for_preprocess(text)

	# remove special characters and delimiters
	remove_list = ["⌊", "⌋", "⌈", "⌉", ":"]
	text = regex_replace_substr(text, remove_list, replace="")

	# remove special characters and delimiters
	remove_list = [
		# special characters
		"↗", "→", "°", "≈", "↓", "⁎", "↓", "↑",
		"\x02", "\x01r", "≠", "↫", "↘", "↗", "☺",
		# CHAT specific delimiters
		"<", ">", "^",
	]
	text = regex_replace_substr(text, remove_list, replace=NEW_TAGS["special-character"])

	# remove pauses (0.5) and (...)
	text = regex_replace_pattern(text, "(", ")", replace=NEW_TAGS["pause"], exclude_if_contains_alpha=True)

	# remove all text between []
	text = regex_replace_pattern(text, "[", "]", replace=NEW_TAGS["bracket"])

	# remove Events: "&="
	text = replace_text_after(text, "&=", replace=NEW_TAGS["event"])

	# remove Interposed Word: "&*"
	text = replace_text_after(text, "&*", replace=NEW_TAGS["interposed-word"])

	# remove Fillers: "&+", "&-", "&~"
	for filler in ["&+", "&-", "&~"]:
		text = replace_text_after(text, filler)

	# remove Long Events: "&{l=* ... &}l=*", "&{n=* ... &}n=*"
	text = replace_text_after(text, "&{l=*", specific_end="&}l=*")
	text = replace_text_after(text, "&{n=*", specific_end="&}n=*")

	# remove specific pattern from annotator
	remove_list = [" ∙h "]
	text = regex_replace_substr(text, remove_list, replace=" ")

	if remove_beg_unfinished_word:
		# remove solo or duo char with "-" ex: " j- ", " i- ", " me- "
		text = regex_replace_pattern(text, " ", "- ", replace=" ", max_alpha_char=2)

	# remove CHAT specific delimiters that we needed before
	remove_list = ["+", "*", "~", "=", "(", ")", "∙"]
	text = regex_replace_substr(text, remove_list, replace="")

	# remove @1 when speaker spelling each letter of a word
	remove_list = ["@1 ", "@l ", "@1", "@l"]
	text = regex_replace_substr(text, remove_list, replace="")

	# remove multiple hhh for laugh
	remove_list = [" " + "h" * i + " " for i in range(2, 7)]
	text = regex_replace_substr(text, remove_list, replace=NEW_TAGS["laugh"])

	# remove non verbal expressions
	remove_list = [
		"mhh", "mhm", "mh", "hm", "um",
		"umph", "euhm",
		"uhuh", "uhu", "huh", "uh",
	]
	remove_list = [" " + r + " " for r in remove_list]
	text = regex_replace_substr(text, remove_list, replace=NEW_TAGS["non-verbal"])

	# remove unintelligible
	remove_list = [" xxx "]
	text = regex_replace_substr(text, remove_list, replace=NEW_TAGS["unintelligible"])

	if remove_tags:
		text = regex_replace_substr(text, NEW_TAGS.values(), replace=" ")

	if lower:
		text = text.lower()

	if remove_accents:
		text = unidecode(text)

	text = " ".join(text.split())
	return text


def preprocess_talkbank_text(text, **kwargs):
	text = str(text)
	text = text.split("\n")
	text = [preprocess_row(t, **kwargs) for t in text]
	text = [t for t in text if len(t) > 0]
	text = "\n".join(text)
	if not text:
		text = "<EMPTY>"
	return text


def apply_preprocess_talkbank_text(path="raw_pred.csv", col="true", ncol="preprocess_true"):
	import pandas as pd

	df = pd.read_csv(path, encoding="utf-8")
	df[ncol] = df[col].apply(lambda x: preprocess_talkbank_text(x, remove_tags=True, remove_beg_unfinished_word=True, lower=True, remove_accents=True))
	df.to_csv(path, index=False, encoding="utf-8")
	text = df[col][203]
	ntext = df[ncol][203]

	print("\n".join(text[:30]))
	print("\n\n")
	print("\n".join(ntext[:30]))


def big_test(path="raw_pred.csv"):
	import pandas as pd

	df = pd.read_csv(path, encoding="utf-8")

	text = df["true"][203]
	ntext = preprocess_talkbank_text(text, remove_tags=True, remove_beg_unfinished_word=False, lower=True, remove_accents=True)

	text = text.split("\n")
	ntext = ntext.split("\n")
	print("\n".join(text[:30]))
	print("\n\n")
	print("\n".join(ntext[:30]))


def unitary_test():
	text = "ouais on est parti↘ (0.3) mais euh (0.4) j'ai arrête dans une euh s- sous une arbre↘ (0.8) euh et: (1.3) et il y A des skinheadS↗ TROIS skinheadS↗ (0.6) et il m'a vu↘  (1.8)"
	ntext = preprocess_talkbank_text(text)
	print(text)
	print("\n\n")
	print(ntext)


if __name__ == '__main__':
	# unitary_test()

	# big_test()

	apply_preprocess_talkbank_text()
	