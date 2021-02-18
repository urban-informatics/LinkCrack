from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from myPath import Path
from torchvision import transforms
from dataloader.utils import ReadIndex
from dataloader import transforms as tr


class CrackSegmentation(Dataset):
    """
    TunnelCrack dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('TunnelCrack'),
                 split='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/test
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self.split = split

        if split == 'train':
            self.img_id = ReadIndex(os.path.join(base_dir, 'train.txt'), shuffle=True)
        elif split == 'test':
            self.img_id = ReadIndex(os.path.join(base_dir, 'test.txt'), shuffle=False)
        else:
            print('Split index {} not available.'.format(split))
            raise ValueError

        self.args = args

    def __len__(self):
        return len(self.img_id)


    def __getitem__(self, index):

        _img = Image.open(os.path.join(self._base_dir, self.img_id[index][0])).convert('RGB')
        _lab = Image.open(os.path.join(self._base_dir, self.img_id[index][1])).convert('L')

        sample = {'image': _img, 'label': _lab}

        if self.split == 'train':
            sample = self.transform_tr(sample)
        else:
            sample = self.transform_val(sample)


        return sample

    def transform_tr(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.base_size),
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.base_size),
            tr.ToTensor()])

        return composed_transforms(sample)



    def __str__(self):
        return 'TunnelCrack(split=' + str(self.split) + ')'


