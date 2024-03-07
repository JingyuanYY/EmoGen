import torch

_ = torch.manual_seed(123)

from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from torchvision import transforms
# Dataloader
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder

from tqdm import tqdm

import os
import numpy as np


class ImageFolderDataset(Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

    def __init__(self, dir, transform=None):
        self.dir = dir
        # find the all valid img paths under the dir
        self.img_paths = self._find_imgs(self.dir)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]) if transform is None else transform

    def _is_valid_img_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def _find_imgs(self, dir):
        img_paths = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if self._is_valid_img_file(fname):
                    path = os.path.join(root, fname)
                    img_paths.append(path)
        return img_paths

    def __getitem__(self, index):
        try:
            img_path = self.img_paths[index]
            img = Image.open(img_path).convert('RGB')
        except:
            # random sample
            print('random sample')
            img_path = self.img_paths[np.random.randint(0, len(self))]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)


class FIDRunner:
    def __init__(self,
                 real_img_root,
                 fake_img_root,
                 fid_num_features=2048,
                 labels=[],
                 ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.real_img_root = real_img_root
        self.fake_img_root = fake_img_root
        self.fid_num_features = fid_num_features
        if len(labels) == 0:
            real_labels = sorted(
                [_ for _ in os.listdir(real_img_root) if os.path.isdir(os.path.join(real_img_root, _))])
        else:
            real_labels = labels
        self.real_label2fid = {label: self._init_fid() for label in real_labels}
        self.fake_labels = sorted(
            [_ for _ in os.listdir(fake_img_root) if os.path.isdir(os.path.join(fake_img_root, _))])

    def _init_fid(self):
        return FrechetInceptionDistance(normalize=True).to(self.device)

    def _load(self, label, fid, real=True):
        real_or_fake = 'real' if real else 'fake'
        img_root = self.real_img_root if real else self.fake_img_root
        npz_path = os.path.join(img_root, f'{label}_fid_{self.fid_num_features}.npz')
        try:
            npz_data = np.load(npz_path)
        except FileNotFoundError:
            return False
        try:
            setattr(fid, f'{real_or_fake}_features_sum', torch.from_numpy(npz_data[f'features_sum']).to(self.device))
            setattr(fid, f'{real_or_fake}_features_cov_sum',
                    torch.from_numpy(npz_data[f'features_cov_sum']).to(self.device))
            setattr(fid, f'{real_or_fake}_features_num_samples',
                    torch.from_numpy(npz_data[f'features_num_samples']).to(self.device))
        except KeyError:
            return False
        return True

    def _save(self, label, fid, real=True):
        real_or_fake = 'real' if real else 'fake'
        img_root = self.real_img_root if real else self.fake_img_root
        npz_path = os.path.join(img_root, f'{label}_fid_{self.fid_num_features}.npz')
        npz_data = {
            f'features_sum': getattr(fid, f'{real_or_fake}_features_sum').cpu().numpy(),
            f'features_cov_sum': getattr(fid, f'{real_or_fake}_features_cov_sum').cpu().numpy(),
            f'features_num_samples': getattr(fid, f'{real_or_fake}_features_num_samples').cpu().numpy(),
        }
        np.savez(npz_path, **npz_data)

    def aggregate_compute(self, aggregated_fid):
        if aggregated_fid.real_features_num_samples < 2 or aggregated_fid.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (aggregated_fid.real_features_sum / aggregated_fid.real_features_num_samples).unsqueeze(0)
        mean_fake = (aggregated_fid.fake_features_sum / aggregated_fid.fake_features_num_samples).unsqueeze(0)

        cov_real_num = aggregated_fid.real_features_cov_sum - aggregated_fid.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (aggregated_fid.real_features_num_samples - 1)
        cov_fake_num = aggregated_fid.fake_features_cov_sum - aggregated_fid.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (aggregated_fid.fake_features_num_samples - 1)

        def _compute_fid(mu1, sigma1, mu2, sigma2):
            a = (mu1 - mu2).square().sum(dim=-1)
            b = sigma1.trace() + sigma2.trace()
            c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

            return a + b - 2 * c

        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)

    def run(self):
        # update the real imgs
        for label, fid in tqdm(self.real_label2fid.items(), desc='update real imgs'):
            if self._load(label, fid, real=True):
                continue
            dataloader = DataLoader(
                ImageFolderDataset(os.path.join(self.real_img_root, label)),
                batch_size=256,
                shuffle=False,
            )
            self._update(fid, dataloader, real=True)
            self._save(label, fid, real=True)

        # update the fake imgs
        for label in tqdm(self.fake_labels, desc='update fake imgs'):
            try:
                fid = self.real_label2fid[label]
            except KeyError:
                raise KeyError(f'The fake label {label} is not in the real labels.')
            dataloader = DataLoader(
                ImageFolderDataset(os.path.join(self.fake_img_root, label)),
                batch_size=256,
                shuffle=False,
            )
            self._update(fid, dataloader, real=False)

        aggregated_fid = self._init_fid()
        print('{0}fid{0}'.format('-' * 10))
        for label, fid in self.real_label2fid.items():
            self._aggregated(aggregated_fid, fid)
            print(f'{label}: {fid.compute():.2f}')
        print(f'aggregated: {self.aggregate_compute(aggregated_fid).detach().item():.2f}')

    def _aggregated(self, aggregated_fid, fid):
        # aggregated_fid.orig_dtype = fid.orig_dtype
        aggregated_fid.real_features_sum += fid.real_features_sum
        aggregated_fid.real_features_cov_sum += fid.real_features_cov_sum
        aggregated_fid.real_features_num_samples += fid.real_features_num_samples
        aggregated_fid.fake_features_sum += fid.fake_features_sum
        aggregated_fid.fake_features_cov_sum += fid.fake_features_cov_sum
        aggregated_fid.fake_features_num_samples += fid.fake_features_num_samples

    def _update(self, fid, dataloader, real):
        for img in dataloader:
            img = img.to(self.device)
            fid.update(img, real=real)


if __name__ == '__main__':
    # real_img_root = '/mnt/d/data/EmoSet/emotion_scene/amusement'
    # fake_img_root = '/mnt/d/result/ti/origin_emb_scene'
    # fake_img_root = '/mnt/d/result/ti/origin_emotion'
    # fake_img_root = '/mnt/d/result/ti/origin_scene'
    # fake_img_root = '/mnt/d/result/ti/origin_scene'
    # fake_img_root = '/mnt/d/result/ti/emb_scene/images'
    # fake_img_root = '/mnt/d/result/ti/emb_emotion'
    # runner = Runner('/mnt/d/data/EmoSet/emotion_scene/amusement', fake_img_root, device=device)

    runner = FIDRunner('/mnt/d/dataset/EmoSet/image/',
                       'runs/test',
                       labels=[])
    runner.run()
