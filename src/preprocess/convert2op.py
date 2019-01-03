# import cv2
import os
import numpy as np
import PIL as Image
import json as json
from tqdm import tqdm 

JSON_PATH = "/media/bighdd7/arayasam/dataset/UCF101/openpose_output/"
CNV_POSE = "/media/bighdd1/arayasam/poseforecasting/preprocess/heatmaps/"

def convert():
    
    count = 0
    for video_name in tqdm(sorted(os.listdir(JSON_PATH))):

        # # Testing/dev block
        # in_count = 0
        # count += 1
        # if count > 5:
        #     break

        video_path = os.path.join(JSON_PATH,video_name)

        frame_mask = []
        occlusion = []
        openpose_npy = np.empty((0,25,3))

        for idx, frame in enumerate(sorted(os.listdir(video_path))):
            # # Testing/dev block
            # in_count += 1
            # if in_count > 10:
            #     break

            frame_path = os.path.join(video_path, frame)
            with open(frame_path, "r") as fp:
                f_json = json.load(fp)
            
            # Filter frame: No personn , add more filers...
            if len(f_json['people']) > 0:
                # Note frameID
                frame_mask.append(idx)

                # Note if joints are occluded
                if 0 in f_json['people'][0]['pose_keypoints_2d']:
                    occlusion.append(1)
                else:
                    occlusion.append(0)

                # Append heatmap for current frame
                pose_list = f_json['people'][0]['pose_keypoints_2d']
                pose_np = np.asarray(pose_list).reshape(-1, 3)
                openpose_npy = np.append(openpose_npy, np.expand_dims(pose_np, axis=0), axis=0)

        #### Save as .npz file
        save_path = CNV_POSE + video_name + ".npz"
        np.savez(save_path, heatmap=openpose_npy, occlusion=occlusion, frameId=frame_mask)

    return None

    
if __name__ == "__main__":
    convert()
