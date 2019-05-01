import os
import glob
import shutil
import time
import datetime

folder = "/home/wc-gpu/MasterThesis/models/research/object_detection/training"
dest = "/home/wc-gpu/MasterThesis/models/research/object_detection/training/keep"

stage = 2000
while True:
    copied = False
    print ("current stage", stage)
    for fn in os.listdir(folder):
        try:
            n = int(fn.split(".")[1].split("-")[1])
            if n > stage:
                for model_file in glob.glob(folder + "/*"+str(n)+"*"):
                    shutil.copy(model_file, dest)
                    print("copied ", model_file)
                copied = True
                stage += 2000
        except:
            continue
    if copied:
        print ("Done copying")
    else:
        print ("Not found stage", stage)

        print ("waiting 10 miuntes")
        currentDT = datetime.datetime.now()
        print(str(currentDT))
        time.sleep(600)
