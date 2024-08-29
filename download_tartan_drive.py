from tqdm import tqdm
import subprocess

with open("/media/yufeng/azfiles.txt") as f:
    filenames = [line.rstrip() for line in f]


for i, filename in enumerate(tqdm(filenames, desc="Processing obs")):
    command = ["wget", "http://airlab-share.andrew.cmu.edu/dataset-icra22/" + filename]
    print("Executing command: ", command)
    print(subprocess.check_output(command))
