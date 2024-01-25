import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from preprocesser import preprocess

from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from pathlib import Path

NOTES_CAT = 2

class CocoLoader(Dataset):

    @staticmethod
    def generate_datasets(root_path: Path, transform=None, target_transform=None):
        train_path = root_path/'train'
        test_path = root_path/'test'
        val_path = root_path/'valid'

        def gen(path: Path):
            return CocoLoader(
                        root=path,
                        annotation_file=path/'_annotations.coco.json',
                        transform=transform,
                        target_transform=target_transform
                    )

        return gen(train_path), gen(val_path), gen(test_path)


    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root: Path, annotation_file: Path, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annotation_file=annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = cv2.imread(str(self.root/path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ## ON VEUT SEULEMENT LES ANNEAUX
        target = [t for t in target if t['category_id']==NOTES_CAT]

        return img, target


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == "__main__":
    dataset, _, _ = CocoLoader.generate_datasets(
            Path('/Users/andlat/Documents/FIRST2024/COCO dataset/FRC2024/'),
            preprocess,
        )

    for img, target in dataset:
        cv2.imshow("image", img)
        print(f'Nombre de notes: {len(target)}')

        cv2.waitKey(0)