import re
import string


def get_alpha_char(text):
	return ''.join(filter(str.isalpha, text))


def contains_alpha(text):
	return True if re.search('[a-zA-Z]', text) is not None else False


def contains_alnum(text):
	"""
	If string contains alphanumeric characters: True else False
	"""
	return bool(re.search('[a-z0-9]', text, re.IGNORECASE))


def replace_text_after(text, after, break_chars="  !.,;" + chr(32), specific_end=None, replace=" "):
	"""
	Replace the text between
		- specific substring (after)
		- a specific character (break_chars)
	Args:
		specific_end (str): if specified, stop the remove only if the specific end is found
	Warning: The text replaced can be a small smegment or the entire text if the break_chars are not well chosen
	"""
	if after in text:
		while after in text:
			position = text.rindex(after)
			i = position
			while i < len(text):
				if specific_end is not None and text[i:].startswith(specific_end):
					break
				elif specific_end is None and text[i] in break_chars:
					break
				i += 1
			text = text[:position] + replace + text[i:]
	return text


def regex_replace_pattern(text, start, end, replace="",
	only_num=False, exclude_if_contains_alpha=False, max_alpha_char=None):
	"""
	Replace a substring starting with 'start' and ending with 'end'
	"""
	pattern = rf"(\{start}.+?\{end})"
	while True:
		pauses = re.findall(pattern, text)
		if len(pauses) == 0:
			break
		skipped = 0
		for p in pauses:
			skipped += 1
			# remove the smallest one
			last_end = p.index(end)
			tmp_p = p[:last_end + 1]
			last_start = tmp_p.rindex(start)
			tmp_p = p[last_start:]
			last_end = tmp_p.index(end)
			p = tmp_p[:last_end] + end
			if only_num:
				tmp = p.replace(start, "").replace(end, "").replace(",", ".").replace(" ", "")
				tmp = int(float(tmp))
				try:
					tmp = int(float(tmp))
				except:
					continue
			if max_alpha_char is not None and\
			len(get_alpha_char(p)) > max_alpha_char:
				continue
			if exclude_if_contains_alpha and contains_alpha(p):
				continue
			text = text.replace(p, replace)
			skipped -= 1
		if skipped == len(pauses):
			break
	return text


def find_substrings(text, substrings):
	"""
	Search for substrings in text
	Return:
		index (int): the index of the first substring found or -1 if none
	"""
	for i, s in enumerate(substrings):
		if s in text:
			return i
	return -1


def regex_replace_substr(text, remove_list, replace=""):
	"""
	Replace a list of substring in given string
	"""
	while find_substrings(text, remove_list) >= 0:
		text = re.sub(r'|'.join(map(re.escape, remove_list)), replace, text)
	return text


def replace_dict_substr(text, replace_dict):
	"""
	Replace substrings according to replace_dict
	"""
	for k, v in replace_dict.items():
		text = text.replace(k, v)
	return text


def space_text_for_preprocess(text):
	"""
	Return a space formated string for preprocessing
		Only one space between characters ex: " there is only one space "
		Startswith " " , endswith " " ex : ' hello '
		Punctuation with space before and after ex : ' . ', ' ? ', ' ! '
	Return:
		text (str)
		ex: ' This is a p0sSiBle return , because this FoRmaT sp@ces '
	"""
	text = " ".join(text.split())
	text = " " + text + " "
	replace_dict = {}
	punct_list = "!?.,;"
	for p in punct_list:
		replace_dict[p] = f" {p} "
	text = replace_dict_substr(text, replace_dict)
	return text