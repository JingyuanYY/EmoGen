import sys
import torchvision.models as models
import torch
import os
from PIL import Image
import random
import lpips
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm



class EmoDataset(Dataset):
    def __init__(self, data_root):
        self.tfm = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_paths = []
        self.data_root = data_root
        for root, _, file_path in os.walk(self.data_root):
            for file in file_path:
                if file.endswith("jpg"):
                    self.image_paths.append(os.path.join(root, file))
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path = self.image_paths[i]
        example = {}
        image = Image.open(path).convert('RGB')
        example['image'] = self.tfm(image)
        return example


def get_biggest_prob(x):
    max_values, _ = torch.max(x, dim=1)
    max_values = max_values.unsqueeze(1)
    return max_values


def Lpips(img0, img1, loss_fn_alex):  # pair_image should be two image, RBG, range(0~1)
    d = loss_fn_alex(img0, img1)
    return d.item()


def DistanceOfCos(img0, img1, model, processor):  # pair_image should be two image, RBG, range(0~1)
    data_pro = processor(images=[img0, img1], return_tensors="pt", padding=True).to(model.device)
    data_pro = model.get_image_features(**data_pro)
    d = 1 - F.cosine_similarity(data_pro[0,:].unsqueeze(0), data_pro[1,:].unsqueeze(0))
    mse = F.mse_loss(data_pro[0,:].unsqueeze(0), data_pro[1,:].unsqueeze(0))
    return d.item(), mse.item()


def Semantic_diversity(wkdir, subdir, num_sample, device):
    emotion_list = ["amusement","awe","contentment","excitement",
                    "anger","disgust","fear","sadness"]
    images_path = []
    curdir = os.path.join(wkdir, subdir)
    model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    loss_fn_alex = lpips.LPIPS(net='alex')
    record = {"lpips_score":[], "difference_score":[], "mse_score":[]}
    # search one emotion's semantic_diversity
    for emotion in emotion_list:
        # get all image in the dir
        cur_path = os.path.join(curdir, emotion)
        for root, _, file_path in os.walk(cur_path):
            for file in file_path:
                if file.endswith("jpg"):
                    path = os.path.join(root, file)
                    images_path.append(path)

        # randomly select two picture and do distance calculation
        total_lpips = []
        difference_list = []
        mse_list = []
        for _ in range(num_sample):
            random_image_pair = random.sample(images_path, 2)
            tfm = transforms.ToTensor()
            img0 = tfm(Image.open(random_image_pair[0]))
            img1 = tfm(Image.open(random_image_pair[1]))
            tmp_lpips_score = Lpips(img0, img1, loss_fn_alex)
            tmp_difference_score, mse_distance = DistanceOfCos(img0, img1, model, processor)
            total_lpips.append(tmp_lpips_score)
            difference_list.append(tmp_difference_score)
            mse_list.append(mse_distance)
        lpips_score = sum(total_lpips)/num_sample
        mse_score = sum(mse_list)/num_sample
        difference_score = sum(difference_list)/num_sample
        with open(f"{curdir}/evaluation.txt", "a") as f:
            f.write(f"---------{emotion}------------- \n")
            f.write(f"LPIPS score: {lpips_score:.3f} \n")
            f.write(f"Semantic diversity score (cos): {difference_score:.4f} \n")
            f.write(f"Semantic diversity score (MSE): {mse_score:.4f} \n")
        record["mse_score"].append(mse_score)
        record["lpips_score"].append(lpips_score)
        record["difference_score"].append(difference_score)
    lpips_score = sum(record["lpips_score"])/len(record["lpips_score"])
    difference_score = sum(record["difference_score"])/len(record["difference_score"])
    mse_score = sum(record["mse_score"])/len(record["mse_score"])
    print(f"LPIPS score: {lpips_score:.3f} \n")
    print(f"Semantic diversity score (MSE): {mse_score:.4f} \n")
    with open(f"{curdir}/evaluation.txt", "a") as f:
        # 在文件末尾追加写入文本内容
        f.write(f"---------Average------------- \n")
        f.write(f"LPIPS score: {lpips_score:.3f} \n")
        f.write(f"Semantic diversity score (cos): {difference_score:.4f} \n")
        f.write(f"Semantic diversity score (MSE): {mse_score:.4f} \n")


@torch.no_grad()
def Semantic_clarity(wkdir, subdir, device):
    cur_dir = os.path.join(wkdir, subdir)
    val_dataset = EmoDataset(cur_dir)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    # picture_num = len(val_dataset)
    val_loader = tqdm(val_loader, file=sys.stdout)

    # 1. scene classifier
    arch = 'resnet50'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)
    scene_classifier = models.__dict__[arch](num_classes=365).to(device)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    scene_classifier.load_state_dict(state_dict)
    scene_classifier.eval()

    # 2.object classifier
    object_classifier = models.resnet50(pretrained=True).to(device)
    object_classifier.eval()

    pred_list = []
    for step, data in enumerate(val_loader):
        pred_scene = scene_classifier(data['image'].to(device))
        prob_pred_scene = F.softmax(pred_scene)
        pred_object = object_classifier(data['image'].to(device))
        prob_pred_object = F.softmax(pred_object)
        big_scene = get_biggest_prob(prob_pred_scene)
        big_object = get_biggest_prob(prob_pred_object)
        for i in range(data['image'].shape[0]):
            pred_list.append(big_scene[i].cpu().item() if big_scene[i] > big_object[i] else big_object[i].cpu().item())
    clarity_score = sum(pred_list)/len(pred_list)
    print(f"Semantic_Clarity_score: {clarity_score:.3f} \n")
    with open(f"{cur_dir}/evaluation.txt", "a") as f:
        # 在文件末尾追加写入文本内容
        f.write(f"Semantic_Clarity_score: {clarity_score:.3f} \n")

if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    num_sample = 10
    file = "/mnt/d/Emo-generation/DB_5"
    sub_dir = 'img'
    Semantic_clarity(file, sub_dir, device)
    Semantic_diversity(file, sub_dir, num_sample, device)

