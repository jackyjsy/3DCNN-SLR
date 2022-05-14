import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

"""
Implementation of Sign Language Dataset
"""
class Sign_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=226, train=True, transform=None, test_clips=10):
        super(Sign_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test_clips = test_clips
        # self.signers = 50
        # self.repetition = 5
        # if self.train:
        #     self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        # else:
        #     self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        # self.data_folder = []
        # try:
        #     obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
        #     self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        # except Exception as e:
        #     print("Something wrong with your data path!!!")
        #     raise
        self.sample_names = []
        self.labels = []
        self.data_folder = []
        label_file = open(label_path, 'r', encoding='utf-8')
        for line in label_file.readlines():
            line = line.strip()
            line = line.split(',')

            self.sample_names.append(line[0])
            self.data_folder.append(os.path.join(data_path, line[0]))
            # print(line[1])
            self.labels.append(int(line[1]))



        # print(self.data_folder)
        # print(aaa)
        # self.labels = {}
        # try:
        #     label_file = open(self.label_path, 'r')
        #     for line in label_file.readlines():
        #         line = line.strip()
        #         line = line.split('\t')
        #         self.labels[line[0]] = line[1]
        # except Exception as e:
        #     raise
    # def frame_indices_tranform(self, video_length, sample_duration):

    #     if video_length <= sample_duration:
    #         # frame_indices =  np.arange(video_length)
    #         frame_indices = np.linspace(0, video_length - 1, num=sample_duration).astype(int)
    #     elif video_length * 0.9 > sample_duration:
    #         random_start = random.randint(0, int(video_length * 0.1))
    #         random_end = random.randint(int(video_length * 0.9), video_length) - 1
    #         frame_indices = np.linspace(random_start, random_end, num=sample_duration).astype(int)
    #     else:
    #         frame_indices = np.linspace(0, video_length - 1, num=sample_duration).astype(int)
    #     # print(frame_indices.shape)
    #     return frame_indices

    def frame_indices_tranform(self, video_length, sample_duration):
        # print(frame_indices.shape)
        if video_length > sample_duration:
            random_start = random.randint(0, video_length - sample_duration)
            frame_indices = np.arange(random_start, random_start + sample_duration)
            # if frame_indices[-1] > video_length:
            #     print('get ', video_length, sample_duration, random_start, random_start + sample_duration)
            # print(video_length, sample_duration, random_start, random_start + sample_duration)
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration]
        assert frame_indices.shape[0] == sample_duration
        return frame_indices

    def frame_indices_tranform_test(self, video_length, sample_duration, clip_no=0):
        # print(frame_indices.shape)
        if video_length > sample_duration:
            start = (video_length - sample_duration) // (self.test_clips - 1) * clip_no
            frame_indices = np.arange(start, start + sample_duration)
        elif video_length == sample_duration:
            frame_indices = np.arange(sample_duration)
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration]

        return frame_indices

    def random_crop_paras(self, input_size, output_size):
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i+output_size, j+output_size

    def read_images(self, folder_path, clip_no=0):
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        # start = 1
        # step = int(len(os.listdir(folder_path))/self.frames)
        if self.train:
            index_list = self.frame_indices_tranform(len(os.listdir(folder_path)), self.frames)
            flip_rand = random.random()
            # angle = (random.random() - 0.5) * 20
            crop_box = self.random_crop_paras(256, 224)
        else:
            # index_list = np.linspace(0, len(os.listdir(folder_path)) - 1, num=self.frames).astype(int)  + 1
            # index_list = np.arange(len(os.listdir(folder_path))) + 1
            index_list = self.frame_indices_tranform_test(len(os.listdir(folder_path)), self.frames, clip_no)
        
        # for i in range(self.frames):
        for i in index_list:
            # image = Image.open(os.path.join(folder_path, '{:04d}.jpg').format(start+i*step))  #.convert('L')
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg').format(i))
            if self.train:
                if flip_rand > 0.5:
                    image = ImageOps.mirror(image)
                # image = transforms.functional.rotate(image, angle) 
                image = image.crop(crop_box)
                assert image.size[0] == 224
            else:
                crop_box = (16, 16, 240, 240)
                image = image.crop(crop_box)
                # assert image.size[0] == 224
            if self.transform is not None:
                image = self.transform(image)

            # flow x - y
            image = image[:2,:,:]
            if self.train and flip_rand > 0.5:
                image[0, :, :] = - image[0, :, :]
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # T, C, H, W

        # print(images.shape)
        return images

    def __len__(self):
        return len(self.data_folder)

    def __getitem__(self, idx):
        selected_folder = self.data_folder[idx]
        if self.train:
            images = self.read_images(selected_folder)
        else:
            images = []
            for i in range(self.test_clips):
                if i > self.test_clips * 0.1 and i < self.test_clips * 0.8:
                    images.append(self.read_images(selected_folder, i))
            images = torch.stack(images, dim=0)
            # M, T, C, H, W
        label = torch.LongTensor([self.labels[idx]])
        # print(images.size(), ', ', label.size())
        return {'data': images, 'label': label}

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]





# Test
if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
    # dataset = CSL_Isolated(data_path="/home/haodong/Data/CSL_Isolated/color_video_125000",
    #     label_path='/home/haodong/Data/CSL_Isolated/dictionary.txt', transform=transform)    # print(len(dataset))
    # print(dataset[1000]['images'].shape)
    # dataset = CSL_Skeleton(data_path="/home/haodong/Data/CSL_Isolated/xf500_body_depth_txt",
    #     label_path="/home/haodong/Data/CSL_Isolated/dictionary.txt", selected_joints=['SPINEBASE', 'SPINEMID', 'HANDTIPRIGHT'], split_to_channels=True)
    # print(dataset[1000])
    # label = dataset[1000]['label']
    # print(dataset.label_to_word(label))
    # dataset[1000]
    dataset = CSL_Continuous(
        data_path="/home/haodong/Data/CSL_Continuous/color",
        dict_path="/home/haodong/Data/CSL_Continuous/dictionary.txt",
        corpus_path="/home/haodong/Data/CSL_Continuous/corpus.txt",
        train=True, transform=transform
        )
    # dataset = CSL_Continuous_Char(
    #     data_path="/home/haodong/Data/CSL_Continuous/color",
    #     corpus_path="/home/haodong/Data/CSL_Continuous/corpus.txt",
    #     train=True, transform=transform
    #     )
    print(len(dataset))
    images, tokens = dataset[1000]
    print(images.shape, tokens)
    print(dataset.output_dim)
