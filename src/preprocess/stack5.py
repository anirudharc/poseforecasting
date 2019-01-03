import os
import csv 

DIR = os.getcwd()
print("Current:", DIR)
path = DIR + "/mturk"

videos = []
for vid in os.listdir(path):
    video = os.path.join(path, vid)
    # print(video)
    if os.path.isfile(video):
        print(vid)
        videos.append(vid)

print("COMBINED:")
combined = list(map(lambda i:videos[i:i+5], range(0,len(videos)-1, 5)))
print(combined)

with open('videos_5col.csv', mode='w') as fp:
    video_writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for idx, sample in enumerate(combined):
        print(idx, sample[0],sample[1],sample[2],sample[3],sample[4])
        video_writer.writerow([sample[0],sample[1],sample[2],sample[3],sample[4]])
