import os
import glob
import torch
import cv2
import rasterio
import numpy as np
from tqdm import tqdm
import torch.utils.data as data


def resiz_4pl(img, size):
    if len(img.shape) == 3:
        imgs = np.zeros((img.shape[0], size[0], size[1]))
        for i in range(img.shape[0]):
            per_img = np.squeeze(img[i, :, :])
            per_img = cv2.resize(per_img, size, interpolation=cv2.INTER_AREA)
            imgs[i, :, :] = per_img
    else:
        imgs = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return imgs

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])


def normalize_S2(imgs):
    for i in range(len(S2_MEAN)):
        imgs[i, :, :] = (imgs[i, :, :] - S2_MEAN[i]) / S2_STD[i]
    return imgs

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]

# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    with rasterio.open(path) as data:
        s2 = data.read(S2_BANDS_HR )

    s2 = np.clip(s2, 0, 10000)
    s2 = resiz_4pl(s2, (222, 222))
    s2 = s2.astype(np.float32)
    s2 = normalize_S2(s2)
    return s2


class S2(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self, path, use_s2hr=True, use_s2mr=False, use_s2lr=False, transform=False,):
        """Initialize the dataset"""

        # inizialize
        super(S2, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.transform = transform
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        train_list = []
        for seasonfolder in ['ROIs1158_spring', 'ROIs1868_summer', 'ROIs1970_fall', 'ROIs2017_winter']:
            train_list += [os.path.join(seasonfolder, x) for x in
                           os.listdir(os.path.join(path, seasonfolder)) if "s2_" in x]
            #train_list = [os.path.join(x, y) for x in train_list for y in os.listdir(os.path.join(path, x))]
            sample_dirs = train_list

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                self.samples.append({"s2": s2_loc, "id": s2_loc})

        print("loaded", len(self.samples), "samples from the dataset")


    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        img = load_s2(sample["s2"], self.use_s2hr, self.use_s2mr, self.use_s2lr)

        return {'image': img, 'id': sample["id"]}


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nOSCD_S2 validation")
    data_dir = "/workplace/S2BYOL/OSCD"
    ds = S2(data_dir, use_s2hr=True, use_s2mr=True)
    s = ds.__getitem__(0)
    print("id:", s["id"], "\n", "input shape:", s["image"].shape, "\n")

