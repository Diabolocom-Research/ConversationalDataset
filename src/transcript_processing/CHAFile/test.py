from ChaFile import *
from glob import glob

chaPaths = glob("/home/macramole/Code/ciipme/aclew/highvolResults/listos_cha/*.cha")
# chaPaths = glob("/home/macramole/Code/ciipme/aclew/highVolFromGit/allCHA/eng/*.cha")

# chaPath = "/home/macramole/Code/ciipme/aclew/highvolResults/listos_cha/donatow-a1-nsm.elan.cha"
# chaPath = "/home/macramole/Code/ciipme/corpusESPON/longi_audio1/codificados/codificados bullets elan/alma-a1-nsb.cha"
# cha = ChaFile( chaPath, language=LANGUAGE_SPANISH )

for c in chaPaths:
    cha = ChaFile( c, language=LANGUAGE_SPANISH )
    # cha = ChaFile( c, language=LANGUAGE_ENGLISH )
    print(cha.countUtterances(False), " / ", cha.countUtterances(True))


