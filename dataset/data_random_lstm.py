import os
import cv2
import glob
import rasterio
import numpy as np
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


# data augmenttaion
class RandomCrop(object):
    """给定图片，随机裁剪其任意一个和给定大小一样大的区域.

    Args:
        output_size (tuple or int): 期望裁剪的图片大小。如果是 int，将得到一个正方形大小的图片.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample, unlabeld=True):

        if unlabeld:
            image, id = sample['image'], sample['id']
            lc = None
        else:
            image, id, lc = sample['image'], sample['id'], sample['label']

        _, _, h, w = image.shape
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, :, top: top + new_h, left: left + new_w]


        # load label
        if unlabeld:
            return {'image': image, 'id': id}
        else:
            lc = lc[:, top: top + new_h, left: left + new_w]
            return {'image': image, 'label': lc, 'id': id}


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
        s2 = data.read(bands_selected)
    # this order cannot be wrong!!!!!
    if len(bands_selected) == 4:
        s2 = s2 * 10000  # because the rediation correction !!!!!!!!!!!!!!
        s2 = resiz_4pl(s2, (512, 512))
        s2 = np.clip(s2, 0, 10000)
        s2 = normalize_S2(s2)
    else:
        s2 = resiz_4pl(s2, (512, 512))
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

def load_lc(path):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)
    lc = resiz_4pl(lc, (512, 512))
    return lc


def load_s2sample(sample, use_s2hr, use_s2mr, use_s2lr, unlabeled=False):

    # load s2 data
    img = load_s2(sample["img"][0], use_s2hr, use_s2mr, use_s2lr)[np.newaxis, :]
    for i in range(1, len(sample['img'])):
        img = np.concatenate((img, load_s2(sample["img"][i], use_s2hr, use_s2mr, use_s2lr)[np.newaxis, :]), axis=0)

    # load label data
    label = load_lc(sample["ref"][0])[np.newaxis, :]
    for i in range(1, len(sample["ref"])):
        label = np.concatenate((label, load_lc(sample["ref"][i])[np.newaxis, :]), axis=0)

    # load label
    if unlabeled:
        return {'image': img, 'label': label, 'id': sample['id']}
    else:
        return {'image': img, 'label': label, 'id': sample['id']}

def load_s1sample(sample, unlabeled=False):

    # load s1 data
    img = load_s1(sample["img"][0])[np.newaxis, :]
    for i in range(1, len(sample['img'])):
        img = np.concatenate((img, load_s1(sample["img"][i])[np.newaxis, :]), axis=0)
    # load label data
    label = load_lc(sample["ref"][0])[np.newaxis, :]
    for i in range(1, len(sample["ref"])):
        label = np.concatenate((label, load_lc(sample["ref"][i])[np.newaxis, :]), axis=0)
    # load label
    if unlabeled:
        return {'image': img, 'label': label, 'id': sample['id']}
    else:
        return {'image': img, 'label': label, 'id': sample['id']}

def load_plsample(sample, unlabeled=False):

    # load pl data
    img = load_pl(sample["img"][0])[np.newaxis, :]
    for i in range(1, len(sample['img'])):
        img = np.concatenate((img, load_pl(sample["img"][i])[np.newaxis, :]), axis=0)
    # load label data
    label = load_lc(sample["ref"][0])[np.newaxis, :]
    for i in range(1, len(sample["ref"])):
        label = np.concatenate((label, load_lc(sample["ref"][i])[np.newaxis, :]), axis=0)
    # load label
    if unlabeled:
        return {'image': img, 'label': label, 'id': sample['id']}
    else:
        return {'image': img, 'label': label, 'id': sample['id']}


class MTCD(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 datatype="val",
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 unlabeled=True,
                 transform=False,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(MTCD, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.unlabeled = unlabeled
        self.datatype = datatype
        # define transform
        if transform:
            self.transform = RandomCrop(crop_size)
        else:
            self.transform = None
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        train_list = []

        for seasonfolder in [
            'T1286_2921_13', 'T2309_3217_13', 'T4421_3800_13', 'T5863_3800_13', 'T6752_3104_13',
            'T1327_3160_13', 'T2345_3680_13', 'T4517_4915_13', 'T5912_3937_13', 'T6752_3115_13',
            'T1429_3296_13', 'T2383_3079_13', 'T4780_3377_13', 'T5916_3785_13', 'T6761_3129_13',
            'T1433_3310_13', 'T2470_5030_13', 'T4791_3920_13', 'T5927_3715_13', 'T6762_3348_13',
            'T1446_2989_13', 'T2569_4513_13', 'T4815_3378_13', 'T6186_3574_13', 'T6763_3346_13',
            'T1474_3210_13', 'T2697_3715_13', 'T4815_3379_13', 'T6460_3366_13', 'T6764_3347_13',
            'T1479_3214_13', 'T2832_4366_13', 'T4816_3380_13', 'T6460_3370_13', 'T6813_3313_13',
            'T1487_3335_13', 'T2979_4481_13', 'T4819_3372_13', 'T6461_3368_13', 'T6824_4117_13',
            'T1567_3314_13', 'T3041_4643_13', 'T4838_3506_13', 'T6475_3361_13',
            'T1736_3318_13', 'T3699_3757_13', 'T4852_3239_13', 'T6665_3433_13',
            'T1973_3709_13', 'T3911_3441_13', 'T4881_3344_13', 'T6678_3548_13',
            'T2037_3758_13', 'T3998_3016_13', 'T4999_3521_13', 'T6678_3579_13', 'T7367_5050_13',
            'T2176_3279_13', 'T4102_2726_13', 'T5156_3514_13', 'T6679_3549_13',
            'T2037_3758_13', 'T4127_2991_13', 'T5184_3399_13', 'T6681_3552_13',
            'T2265_3451_13', 'T4196_2710_13', 'T5195_3388_13', 'T6691_3363_13',
            'T2284_2983_13', 'T4254_2915_13', 'T5753_3282_13', 'T6730_3430_13', 'T7394_5018_13']:

            train_list += [os.path.join(seasonfolder, x) for x in os.listdir(os.path.join(path, seasonfolder)) if 'Images' in x]

        sample_dirs = train_list
        self.samples = []

        for folder in sample_dirs:
            asc_names = folder.split('/')[0] + '_asc_*.tif'
            s2_locations = glob.glob(os.path.join(path, f"{folder}/20*.tif"), recursive=True)
            asc_labels = glob.glob(os.path.join(path, f"{folder}/{asc_names}"), recursive=True)
            asc_labels = sorted(asc_labels, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
            # ascending
            s2_locations_asc = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:6]), reverse=True)
            self.samples.append({'img': s2_locations_asc, 'ref': asc_labels, 'id': folder})

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
        if self.transform:
            return self.transform(data_sample, unlabeld=self.unlabeled)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


class MTCD_absent(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 datatype="val",
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 unlabeled=True,
                 transform=False,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(MTCD_absent, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.unlabeled = unlabeled
        self.datatype = datatype
        # define transform
        if transform:
            self.transform = RandomCrop(crop_size)
        else:
            self.transform = None
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        train_list = []

        for seasonfolder in [
            'T2309_3217_13', 'T4421_3800_13', 'T5863_3800_13', 'T6752_3104_13',
            'T1327_3160_13', 'T2345_3680_13', 'T4517_4915_13', 'T5912_3937_13', 'T6752_3115_13',
            'T1429_3296_13', 'T2383_3079_13', 'T4780_3377_13', 'T5916_3785_13', 'T6761_3129_13',
            'T1433_3310_13', 'T2470_5030_13', 'T4791_3920_13', 'T5927_3715_13', 'T6762_3348_13',
            'T1446_2989_13', 'T2569_4513_13', 'T4815_3378_13', 'T6186_3574_13', 'T6763_3346_13',
            'T1474_3210_13', 'T2697_3715_13', 'T4815_3379_13', 'T6460_3366_13', 'T6764_3347_13',
            'T1479_3214_13', 'T2832_4366_13', 'T4816_3380_13', 'T6460_3370_13', 'T6813_3313_13',
            'T1487_3335_13', 'T2979_4481_13', 'T4819_3372_13', 'T6461_3368_13', 'T6824_4117_13',
            'T1567_3314_13', 'T3041_4643_13', 'T4838_3506_13', 'T6475_3361_13',
            'T3699_3757_13', 'T4852_3239_13', 'T6665_3433_13',
            'T1973_3709_13', 'T3911_3441_13', 'T4881_3344_13', 'T6678_3548_13',
            'T2037_3758_13', 'T3998_3016_13', 'T4999_3521_13', 'T6678_3579_13', 'T7367_5050_13',
            'T2176_3279_13', 'T4102_2726_13', 'T5156_3514_13', 'T6679_3549_13',
            'T2037_3758_13', 'T4127_2991_13', 'T5184_3399_13', 'T6681_3552_13',
            'T2265_3451_13', 'T4196_2710_13', 'T5195_3388_13', 'T6691_3363_13',
            'T2284_2983_13', 'T4254_2915_13', 'T5753_3282_13', 'T7394_5018_13']:

            train_list += [os.path.join(seasonfolder, x) for x in os.listdir(os.path.join(path, seasonfolder)) if 'Images' in x]

        sample_dirs = train_list
        self.samples = []

        for folder in sample_dirs:
            asc_names = folder.split('/')[0] + '_asc_*.tif'
            s2_locations = glob.glob(os.path.join(path, f"{folder}/20*.tif"), recursive=True)
            asc_labels = glob.glob(os.path.join(path, f"{folder}/{asc_names}"), recursive=True)
            asc_labels = sorted(asc_labels, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
            # ascending
            s2_locations_asc = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:6]), reverse=True)
            self.samples.append({'img': s2_locations_asc, 'ref': asc_labels, 'id': folder})

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
        if self.transform:
            return self.transform(data_sample, unlabeld=self.unlabeled)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

class MTCD13(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 datatype="val",
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 unlabeled=True,
                 transform=False,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(MTCD13, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.unlabeled = unlabeled
        self.datatype = datatype
        # define transform
        if transform:
            self.transform = RandomCrop(crop_size)
        else:
            self.transform = None
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        train_list = []

        for seasonfolder in [
            'T1286_2921_13', 'T2309_3217_13', 'T4421_3800_13', 'T5863_3800_13', 'T6752_3104_13',
            'T1327_3160_13', 'T2345_3680_13', 'T4517_4915_13', 'T5912_3937_13', 'T6752_3115_13',
            'T1429_3296_13', 'T2383_3079_13', 'T4780_3377_13', 'T5916_3785_13', 'T6761_3129_13',
            'T1433_3310_13', 'T2470_5030_13', 'T4791_3920_13', 'T5927_3715_13', 'T6762_3348_13',
            'T1446_2989_13', 'T2569_4513_13', 'T4815_3378_13', 'T6186_3574_13', 'T6763_3346_13',
            'T1474_3210_13', 'T2697_3715_13', 'T4815_3379_13', 'T6460_3366_13', 'T6764_3347_13',
            'T1479_3214_13', 'T2832_4366_13', 'T4816_3380_13', 'T6460_3370_13', 'T6813_3313_13',
            'T1487_3335_13', 'T2979_4481_13', 'T4819_3372_13', 'T6461_3368_13', 'T6824_4117_13',
            'T1567_3314_13', 'T3041_4643_13', 'T4838_3506_13', 'T6475_3361_13',
            'T1736_3318_13', 'T3699_3757_13', 'T4852_3239_13', 'T6665_3433_13',
            'T1973_3709_13', 'T3911_3441_13', 'T4881_3344_13', 'T6678_3548_13',
            'T2037_3758_13', 'T3998_3016_13', 'T4999_3521_13', 'T6678_3579_13', 'T7367_5050_13',
            'T2176_3279_13', 'T4102_2726_13', 'T5156_3514_13', 'T6679_3549_13',
            'T2037_3758_13', 'T4127_2991_13', 'T5184_3399_13', 'T6681_3552_13',
            'T2265_3451_13', 'T4196_2710_13', 'T5195_3388_13', 'T6691_3363_13',
            'T2284_2983_13', 'T4254_2915_13', 'T5753_3282_13', 'T6730_3430_13', 'T7394_5018_13']:

            train_list += [os.path.join(seasonfolder, x) for x in os.listdir(os.path.join(path, seasonfolder)) if 'Images' in x]

        sample_dirs = train_list
        self.samples = []

        for folder in sample_dirs:
            asc_names = folder.split('/')[0] + '_asc_*.tif'
            s2_locations = glob.glob(os.path.join(path, f"{folder}/20*.tif"), recursive=True)
            asc_labels = glob.glob(os.path.join(path, f"{folder}/{asc_names}"), recursive=True)
            asc_labels = sorted(asc_labels, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
            # ascending
            s2_locations_asc = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:6]), reverse=True)
            self.samples.append({'img': s2_locations_asc, 'ref': asc_labels, 'id': folder})

        print("loaded", len(self.samples), "samples from the dfc2020 subset")

    def __getitem__(self, index):
        """Get a single example from the dataset"""
        # get and load sample from index file
        sample = self.samples[index]
        if self.datatype == 'S2':
            data_sample = load_s2sample(sample, True, True, True, unlabeled=self.unlabeled)
        elif self.datatype == 'S1':
            data_sample = load_s1sample(sample, unlabeled=self.unlabeled)
        elif self.datatype == 'PL':
            data_sample = load_plsample(sample, unlabeled=self.unlabeled)
        else:
            print('no this data!!!')
            data_sample = None
        if self.transform:
            return self.transform(data_sample, unlabeld=self.unlabeled)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nOSCD_S2 validation")
    data_dir = "../MTCD"
    ds = MTCD(data_dir, use_s1=True, use_s2hr=True, use_s2mr=True)
    s = ds.__getitem__(0)
    print("input shape:", len(s["image_asc"]), "\n")


