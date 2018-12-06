# import cv2
import os
import numpy as np
import PIL as Image
import json as json

JSON_PATH = "/media/bighdd7/arayasam/dataset/UCF101/openpose_output/"


def json_to_numpy():
    
    count = 0
    for video_name in sorted(os.listdir(JSON_PATH)):

        # Testing/dev block
        in_count = 0
        count += 1
        if count > 5:
            break

        video_path = os.path.join(JSON_PATH,video_name)

        frame_mask = []
        occlusion = []
        openpose_npy = np.empty((0,25,3))
        print(openpose_npy)

        for idx, frame in enumerate(sorted(os.listdir(video_path))):
            # # Testing/dev block
            # in_count += 1
            # if in_count > 10:
            #     break

            frame_path = os.path.join(video_path, frame)
            
            with open(frame_path, "r") as fp:
                f_json = json.load(fp)
            
            if len(f_json['people']) > 0:
                frame_mask.append(idx)

                # Note if joints are occluded
                if 0 in f_json['people'][0]['pose_keypoints_2d']:
                    occlusion.append(1)
                else:
                    occlusion.append(0)

                pose_list = f_json['people'][0]['pose_keypoints_2d']
                pose_np = np.asarray(pose_list).reshape(-1, 3)

                openpose_npy = np.append(openpose_npy, np.expand_dims(pose_np, axis=0), axis=0)
                # # Test print state - REMOVE
                # print("______________________________________________________________________________")
                # print(idx, frame_path)
                # print(pose_list)
                # print("______________________________________________________________________________")
                # print(pose_np)
                # print(pose_np.shape)
                # print("______________________________________________________________________________")

        # print(frame_mask)
        # print(occlusion)
        # print(openpose_npy)

        #### ^ Save as .npz file
        
        print(openpose_npy.shape)
        print("######################################################################################################################")


    return None

    
if __name__ == "__main__":
    json_to_numpy()
