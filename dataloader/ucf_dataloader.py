import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure
import cv2
import os
import numpy as np
import os.path

BASE_DIR = os.getcwd()
DATA_DIR = '/media/bighdd7/arayasam/dataset/UCF101'
# DATA_DIR = '/home/ubuntu/dataset/UCF101'
RGB_DIR  = DATA_DIR + '/jpegs_256/'
POSE_DIR = DATA_DIR + '/openpose_output/'

# POTION_DIR = DATA_DIR + '/potion/'

UCF_LIST = BASE_DIR + '/UCF_list/'
VIDEO_DICT = BASE_DIR

# # Set randome seed
# torch.random.manual_seed(args.seed)                                                                │
# np.random.seed(args.seed)                                                                          │
# torch.cuda.manual_seed_all(args.seed) 


class ucf_dataset(Dataset):  
    def __init__(self, dic, rgb_dir, pose_dir, mode, transform=None):

        self.keys = list(dic.keys())
        self.values= list(dic.values())
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name):
        rgb_path = self.rgb_dir + 'v_' + video_name + '/'
        pose_path = self.pose_dir + 'v_' + video_name + '/'
  
        img = self.potion_transform(video_name)

        img = Image.fromarray(img)
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name = self.keys[idx]
            
        elif self.mode == 'val':
            video_name = self.keys[idx]

        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        
        data ={}
        if self.mode=='train':
            # Maintaining a dict for future extension of features
            data['potion'] = self.load_ucf_image(video_name)
                    
            sample = (data, label)

        elif self.mode=='val':
            data['potion'] = self.load_ucf_image(video_name)
            
            sample = (data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class ucf_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, rgb_path, pose_path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.rgb_path=rgb_path
        self.pose_path=pose_path
        self.frame_count={}
        self.train_video, self.val_video, self.test_video = self.load_video_dict(ucf_list, ucf_split)
        
    def load_video_dict(self, ucf_list, ucf_split, train_split=0.6, val_split=0.2):

        if os.path.isfile(VIDEO_DICT + "/video_dict.pickle"):
            print("Loading existing video dictionary...")
            with open('video_dict.pickle', 'rb') as handle:
                final_dic = pickle.load(handle)

        else:
            # split the training and testing videos
            print("Creating video dictionary...")
            splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
            train_video, val_video = splitter.split_video()
            final_dic = dict(list(train_video.items()) + list(val_video.items()))

            print(os.path.join(VIDEO_DICT + '/video_dict.pickle'))
            with open(os.path.join(VIDEO_DICT + '/video_dict.pickle'), 'wb') as handle:
                pickle.dump(final_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

        filter_dic = []
        #Filter for exisiting videos:
        for key, value in final_dic.items():
            rgb = self.pose_path + "v_" + key
            pose = self.rgb_path + "v_" + key

            if os.path.isdir(rgb) and os.path.isdir(rgb):
                filter_dic.append((key, value))

        n = len(filter_dic)
        np.random.shuffle(filter_dic)
        train_end = int(n*train_split)
        val_end = int(n*(train_split+val_split))
        train_video, val_video, test_video = dict(filter_dic[0:train_end]),\
                 dict(filter_dic[train_end+1:val_end]), dict(filter_dic[val_end + 1:])
        
        print("Filtered videos retained: ", n)
        print("Train {}, Val {}, Test {}".format(len(train_video), len(val_video), len(test_video)))
        # action_labels = splitter.get_action_index()
        # print(action_labels)

        return train_video, val_video, test_video

    def run(self):
        train_loader = self.train()
        val_loader = self.validate()
        test_loader = self.test()
        return train_loader, val_loader, test_loader

    def train(self):
        training_set = ucf_dataset(dic=self.train_video, rgb_dir=self.rgb_path, pose_dir=self.pose_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Training data :',len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = ucf_dataset(dic=self.val_video, rgb_dir=self.rgb_path, pose_dir=self.pose_path, mode='val', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('==> Validation data :',len(validation_set),'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

    def test(self):
        test_set = ucf_dataset(dic=self.test_video, rgb_dir=self.rgb_path, pose_dir=self.pose_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Test data :',len(test_set),'frames')

        test_loader = DataLoader(
            dataset=test_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return test_loader

if __name__ == '__main__':
    
    dataloader = ucf_dataloader(BATCH_SIZE=1, num_workers=1, 
                                rgb_path=RGB_DIR,
                                pose_path=POSE_DIR,
                                ucf_list=UCF_LIST,
                                ucf_split='01')
    train_loader, val_loader, test_loader = dataloader.run()
    print(len(train_loader), len(val_loader), len(test_loader))

    for data in test_loader:
        print(data)
