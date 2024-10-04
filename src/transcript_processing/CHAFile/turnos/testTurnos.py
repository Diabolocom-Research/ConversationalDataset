import sys
sys.path.insert(0, '../')

from ChaFile import *
from glob import glob
import json

"""
. Tanto las intervening como las non-intervening son muchas menos que las utts

. TODO: Checkear los VS ADS porque estan dirigidos a cualquiera
"""

# for f in glob("/home/leandro/Code/ciipme/aclew/highvolResults/listos_cha/*.cha"):
#     cha = ChaFile( f, language = LANGUAGE_SPANISH )
#     filename = os.path.basename(f)

#     turnos = getTurnos()
#     cantTurnos = 0
#     for t in turnos:
#         cantTurnos += len(turnos[t])
	
#     cantLineas = len(cha.getLines())

#     print(f"{filename}: {cantTurnos} / {cantLineas}")

### Uno solo

# chaPath = "/home/macramole/Code/ciipme/aclew/highvolResults/listos_cha/donatow-a1-nsm.elan.cha"
# chaPath = "/home/macramole/Code/ciipme/aclew/highvolResults/listos_cha/camilam-a1-nsm.elan.cha"
chaPath = "/home/macramole/Code/ciipme/aclew/highvolResults/listos_cha/felixv-a3-nsm.elan.cha"

# chaPath = "/home/macramole/Code/ciipme/corpusESPON/longi_audio1/codificados/codificados bullets elan/facundoa-a1-nsm.cha"
# chaPath = "/home/macramole/Code/ciipme/corpusESPON/longi_audio1/codificados/codificados bullets elan/francescal-a1-nsm.cha"

cha = ChaFile( chaPath, language = LANGUAGE_SPANISH )

print("")
print(chaPath)
print("")

uttsBySpeaker = cha.getLinesBySpeakers()
uttsBySpeakerByAddressee = {}
for s in uttsBySpeaker:
	uttsToChild = 0
	uttsToAdult = 0
	for line in uttsBySpeaker[s]:
		if line[LINE_ADDRESSEE] == SPEAKER_ADULT:
			uttsToAdult += 1
		elif line[LINE_ADDRESSEE] == SPEAKER_TARGET_CHILD:
			uttsToChild += 1

	uttsBySpeakerByAddressee[s] = {
		"child" : uttsToChild,
		"adult" : uttsToAdult,
	}

turnos = cha.getTurns( ADDRESSEE_CHILD_DIRECTED )
with open("testTurnosCDS.json", "w") as f:
	json.dump(turnos, f)

print("")
print("***")
print("CDS (turnos/utts)")
print("***")
for speaker in turnos:
	cantTurnos = len(turnos[speaker])
	print(f"{speaker}: {cantTurnos} / { uttsBySpeakerByAddressee[speaker]['child'] }")

turnos = cha.getTurns( ADDRESSEE_ADULT, True )
for speaker in turnos:
	turnos[speaker][:] = [ utts for utts in turnos[speaker] if len(utts) > 1 ]
with open("testTurnosADS_intervening.json", "w") as f:
	json.dump(turnos, f)

print("")
print("***")
print("ADS intervening (turnos/utts)")
print("***")
for speaker in turnos:
	cantTurnos = len(turnos[speaker])
	print(f"{speaker}: {cantTurnos} / { uttsBySpeakerByAddressee[speaker]['adult'] }")

turnos = cha.getTurns( ADDRESSEE_ADULT, False )
for speaker in turnos:
	turnos[speaker][:] = [ utts for utts in turnos[speaker] if len(utts) > 1 ]
with open("testTurnosADS_NON-intervening.json", "w") as f:
	json.dump(turnos, f)

print("")
print("***")
print("ADS NON-intervening (turnos/utts)")
print("***")
for speaker in turnos:
	cantTurnos = len(turnos[speaker])
	print(f"{speaker}: {cantTurnos} / { uttsBySpeakerByAddressee[speaker]['adult'] }")