import os
import cv2
import glob
import rasterio
import numpy as np
from itertools import combinations
import torch.utils.data as data


def resiz_4pl(img, size):
    imgs = np.zeros((img.shape[0], size[0], size[1]))
    for i in range(img.shape[0]):
        per_img = np.squeeze(img[i, :, :])
        per_img = cv2.resize(per_img, size, interpolation=cv2.INTER_AREA)
        imgs[i, :, :] = per_img
    return imgs

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def normalize_S2(imgs):
    for i in range(4):
        imgs[i, :, :] = (imgs[i, :, :] - S2_MEAN[i]) / S2_STD[i]
    return imgs

S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def normalize_S1(imgs):
    for i in range(2):
        imgs[i, :, :] = (imgs[i, :, :] - S1_MEAN[i]) / S1_STD[i]
    return imgs

L8_MEAN = np.array([0.13946152, 0.12857966, 0.12797806, 0.23040992])
L8_STD  = np.array([0.01898952, 0.02437881, 0.03323532, 0.04915179])

def normalize_L8(imgs):
    for i in range(4):
        imgs[i, :, :] = (imgs[i, :, :] - L8_MEAN[i]) / L8_STD[i]
    return imgs

pl_MEAN = np.array([620.56866, 902.1002, 1011.31476, 2574.5764])
pl_STD  = np.array([219.36754, 254.2806, 350.12357, 535.43195])

def normalize_pl(imgs):
    for i in range(4):
        imgs[i, :, :] = (imgs[i, :, :] - pl_MEAN[i]) / pl_STD[i]
    return imgs

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]

L8_BANDS_HR = [2, 3, 4, 5]
L8_BANDS_MR = [5, 6, 7, 9, 12, 13]
L8_BANDS_LR = [1, 10, 11]

pl_BANDS_HR = [1, 2, 3, 4]

S1_BANDS_HR = [1, 2]

# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(S2_BANDS_HR)
    s2 = s2 * 10000  # because the rediation correction !!!!!!!!!!!!!!
    s2 = resiz_4pl(s2, (128, 128))
    s2 = np.clip(s2, 0, 10000)
    s2 = normalize_S2(s2)
    s2 = s2.astype(np.float32)
    return s2

def load_l8(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + L8_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + L8_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + L8_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        l8 = data.read(bands_selected)
    l8 = l8.astype(np.float32)
    l8 = np.clip(l8, 0, 1)
    l8 = normalize_L8(l8)
    return l8

# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read([1, 2])
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 = normalize_S1(s1)
    return s1

# util function for reading s2 data
def load_pl(path):

    with rasterio.open(path) as data:
        pl = data.read(pl_BANDS_HR)
    pl = pl.astype(np.float32)
    pl = np.clip(pl, 0, 10000)
    pl = normalize_pl(pl)
    return pl



# this function for classification and most important is for weak supervised
def load_s2sample(sample, use_s2hr, use_s2mr, use_s2lr, unlabeled=False):

    img_pair_asc = []
    # length of image list
    length_asc = len(sample['img_asc'])
    sample_asc = sample['img_asc']
    for i in range(length_asc):
        img_asc = load_s2(sample_asc[i][0], use_s2hr, use_s2mr, use_s2lr)
        img_asc = np.concatenate((img_asc, load_s2(sample_asc[i][1], use_s2hr, use_s2mr, use_s2lr)), axis=0)
        img_pair_asc.append({'img_asc_'+str(i): img_asc})

    if unlabeled:
        return {'image_asc': img_pair_asc, 'id': sample['id'], 'length': length_asc}
    else:
        return {'image_asc': img_pair_asc, 'id': sample['id'], 'length': length_asc}


def load_s1sample(sample, unlabeled=False):
    img_pair_asc = []
    # length of image list
    length_asc = len(sample['img_asc'])
    sample_asc = sample['img_asc']
    for i in range(length_asc):
        img_asc = load_s1(sample_asc[i][0])
        img_asc = np.concatenate((img_asc, load_s1(sample_asc[i][1])), axis=0)
        img_pair_asc.append({'img_asc_' + str(i): img_asc})

    if unlabeled:
        return {'image_asc': img_pair_asc, 'id': sample['id'], 'length': length_asc}
    else:
        return {'image_asc': img_pair_asc, 'id': sample['id'], 'length': length_asc}

def load_plsample(sample, unlabeled=False):
    img_pair_asc = []
    # length of image list
    length_asc = len(sample['img_asc'])
    sample_asc = sample['img_asc']
    for i in range(length_asc):
        img_asc = load_pl(sample_asc[i][0])
        img_asc = np.concatenate((img_asc, load_pl(sample_asc[i][1])), axis=0)
        img_pair_asc.append({'img_asc_' + str(i): img_asc})

    if unlabeled:
        return {'image_asc': img_pair_asc, 'id': sample['id'], 'length': length_asc}
    else:
        return {'image_asc': img_pair_asc, 'id': sample['id'], 'length': length_asc}


class ImgPair(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 datatype='S2',
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 unlabeled=True,):
        """Initialize the dataset"""

        # inizialize
        super(ImgPair, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.unlabeled = unlabeled
        self.datatype = datatype
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        train_list = []
        for seasonfolder in [
            '1286_2921_13', '2528_4620_13', '4426_3835_13', '5754_3601_13', '6752_3104_13',
            '1311_3077_13', '2569_4513_13', '4523_4210_13', '5830_3834_13', '6752_3115_13',
            '1330_3107_13', '2624_4314_13', '4553_3325_13', '5863_3800_13', '6761_3129_13',
            '1417_3281_13', '2697_3715_13', '4622_3159_13', '5912_3937_13', '6810_3478_13',
            '1487_3335_13', '2789_4694_13', '4666_2369_13', '5926_3715_13', '6813_3313_13',
            '1700_3100_13', '2832_4366_13', '4780_3377_13', '5989_3554_13', '6824_4117_13',
            '1973_3709_13', '2850_4139_13', '4791_3920_13', '6204_3495_13', '7026_3201_13',
            '2029_3764_13', '3002_4273_13', '4806_3588_13', '6353_3661_13', '7312_3008_13',
            '2065_3647_13', '3830_3914_13', '4838_3506_13', '6381_3681_13', '7367_5050_13',
            '2196_3885_13', '3998_3016_13', '4856_4087_13', '6466_3380_13', '7513_4968_13',
            '2235_3403_13', '4127_2991_13', '4881_3344_13', '6475_3361_13', '7517_4908_13',
            '2415_3082_13', '4223_3246_13', '5111_4560_13', '6678_3579_13', '8077_5007_13',
            '2459_4406_13', '4254_2915_13', '5125_4049_13', '6688_3456_13',
            '2470_5030_13', '4421_3800_13', '5195_3388_13', '6730_3430_13']:
            train_list += [os.path.join(seasonfolder, x) for x in os.listdir(os.path.join(path, seasonfolder)) if 'Images' in x]

        sample_dirs = train_list

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/201*.tif"), recursive=True)
            # ascending
            s2_locations_asc = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:8]))
            comb_list_asc = [[s2_locations_asc[i], s2_locations_asc[i-1]] for i in range(1, len(s2_locations_asc))]
            #comb_list_asc = [comb_list_asc[i] for i in range(11)]
            comb_list_asc = list(reversed(comb_list_asc))
            self.samples.append({'img_asc': [(s1_loc, s2_loc) for (s1_loc, s2_loc) in comb_list_asc], 'id': folder})

        print("loaded", len(self.samples), "samples from the dfc2020 subset")

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]

        if self.datatype == 'S2':
            data_sample = load_s2sample(sample, self.use_s2hr, self.use_s2mr, self.use_s2lr, unlabeled=self.unlabeled)
        elif self.datatype == 'S1':
            data_sample = load_s1sample(sample, unlabeled=self.unlabeled)
        elif self.datatype == 'PL':
            data_sample = load_plsample(sample, unlabeled=self.unlabeled)
        else:
            print('no this data!!!')
            data_sample = None

        return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nOSCD_S2 validation")
    data_dir = "../TimeCD"
    ds = ImgPair(data_dir, use_s1=True, use_s2hr=True, use_s2mr=True)
    s = ds.__getitem__(0)
    print("input shape:", len(s["image_asc"]), "\n")


