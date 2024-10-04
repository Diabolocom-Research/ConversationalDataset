
[![DOI](https://zenodo.org/badge/201303243.svg)](https://zenodo.org/badge/latestdoi/201303243)


# CHA file parser for Python

Class for parsing [CLAN's](http://dali.talkbank.org/clan/) CHA file.

Made by Leandro Garber from [CIIPME-CONICET](http://www.ciipme-conicet.gov.ar/)

## Features
* Utterances as a list of strings
* MOR tier as objects
* Easily add more custom tiers 
* Count tokens and types of words, utterances, nouns, verbs and adjectives. Filter by child directed, child produced and overheard speech.
* Count main verbs, either referring to physical or mental actions. Auxiliary verbs present in periphrastic verbs are excluded. (spanish only)

## Usage

### Import
```python
import sys
sys.path.insert(0, '<path_to_cloned_repo>')

from ChaFile import *
```

### Instance

```python
cha = ChaFile(<path_to_cha_file>)
```
Options
   
```python
cha = ChaFile(<path_to_cha_file> )
```
### Get utterances
```python
lines = cha.getLines()
```
Each line is an object with:
* LINE_UTTERANCE : The text of the utterance
* LINE_NUMBER 
* LINE_SPEAKER
* LINE_ADDRESSEE
* LINE_BULLET : Timestamp
* TIER_MOR : A list of objects with MOR data: MOR_UNIT_LEXEMA and MOR_UNIT_CATEGORIA
* ... any other tier

### Cite

Garber, L. (2019). CHA file python parser. Zenodo. https://doi.org/10.5281/zenodo.3364020
