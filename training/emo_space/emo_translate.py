import PIL.Image as Image
import os
import torch
import torch.nn as nn
from model import BackBone
from torchvision import transforms
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import *


class image_encoder(nn.Module):
    def __init__(self):
        super(image_encoder, self).__init__()
        self.resnet = BackBone()
        state = torch.load("weights/2023-08-22-best.pth")
        self.resnet.load_state_dict(state)
        # print(self.resnet)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet.children())[1:-1])
        # print(self.resnet50)

    def forward(self, x):
        out = self.resnet50(x)
        return out


class EmoSet(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.emotion = ["amusement", "sadness", "awe", "anger", "contentment", "disgust", "fear", "excitement"]

        self.image_paths = []
        for root, _, file_path in os.walk(data_root):
            for file in file_path:
                if file.endswith("jpg"):
                    self.image_paths.append(os.path.join(root, file))


    def __getitem__(self, i):
        path = self.image_paths[i]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        emo, number = path.split('/')[-1].split('_')
        example = {}
        example["image"] = image
        example["emo"] = emo
        example["number"] = number
        return example

    def __len__(self):
        return len(self.image_paths)


data_root = "EmoSet/image/"
batch_size = 5
tfm = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


cnn = image_encoder()
cnn.eval()
Emodata = EmoSet(data_root=data_root, transform=tfm)
train_dataloader = torch.utils.data.DataLoader(
        Emodata, batch_size=batch_size, shuffle=False, num_workers=5
    )

accelerator_project_config = ProjectConfiguration()

accelerator = Accelerator(
    project_config=accelerator_project_config,
)
cnn, train_dataloader = accelerator.prepare(cnn, train_dataloader)

progress_bar = tqdm(range(0, len(train_dataloader)), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")

for step, batch in enumerate(train_dataloader):
    img = batch["image"]
    vec = cnn(img)
    size = len(batch['emo'])
    for i in range(size):
        emo, number = batch["emo"][i], batch["number"][i]
        path = f"./emo_space/{emo}"
        os.makedirs(path, exist_ok=True)
        torch.save(vec[i], f"./emo_space/{emo}/{emo}_{number}_CLIPImg.pt")
    progress_bar.update(1)
print("finish")