import sys
import torch
import os
from model import *
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
from tqdm.auto import tqdm
import argparse
import random
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import torch.functional as F


@torch.no_grad()
def count_relate(img, model, processor):
    with open(f'dataset_balance/all_attribute_object_scene.pkl', 'rb') as f:
        attribute_total = pickle.load(f)
    data_pro = processor(images=img, text=attribute_total, return_tensors="pt", padding=True).to(model.device)
    data_pro = model(**data_pro)
    score = data_pro.logits_per_image.squeeze(0)
    indice = torch.argmax(score, dim=0)
    relate_semantic = attribute_total[indice.item()]
    relate_score = score[indice.item()].item()

    return relate_semantic, relate_score


@torch.no_grad()
def inference(arg, emotion):
    working_path = arg.working_path
    device = torch.device(arg.device if torch.cuda.is_available() else "cpu")  # TODO
    placeholder_token = f"<{emotion}>"
    save_dir = f"{working_path}/img/{emotion}"
    prompt = [placeholder_token]
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 50  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    # generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
    batch_size = len(prompt)
    num_picture = arg.num_picture
    repo_id = arg.repo_id
    model = CLIPModel.from_pretrained("clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("clip-vit-large-patch14")
    vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")
    vae.to(device)

    tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")

    class image_encoder(nn.Module):
        def __init__(self):
            super(image_encoder, self).__init__()
            self.resnet = BackBone()
            self.resnet = torch.nn.Sequential(*list(self.resnet.children())[1:-1])

        def forward(self, x):
            out = self.resnet(x)
            return out

    model_dict = {
        "FC": lambda args: FC(),
        "MLP": lambda args: MLP(args.num_fc_layers, args.need_ReLU, args.need_LN, args.need_Dropout),
        "SimpleMLP": lambda args: SimpleMLP(args.need_ReLU, args.need_Dropout),
    }
    mapper = model_dict[arg.mapper_name](arg)
    state = torch.load(os.path.join(working_path, "mapper.pth"))
    mapper.load_state_dict(state)
    mapper.eval()
    e_mean = torch.load(f"emo_space/{emotion}_mean_v2.pt")
    e_std = torch.load(f"emo_space/{emotion}_std_v2.pt")
    normal = torch.distributions.Normal(e_mean, e_std)

    def save_pic(emotion, img, path, semantic, score):
        if path is not None:
            os.makedirs(path, exist_ok=True)
        try:
            files = sorted([x for x in os.listdir(path) if x.endswith("_v2.jpg")], key=lambda x: int(x.split("_")[1]))
            num = int(files[-1].split("_")[1])
            img.save(f"{path}/{emotion}_{num + 1}_{semantic}_{score:.2f}_v2.jpg")
        except:
            img.save(f"{path}/{emotion}_0_{semantic}_{score:.2f}_v2.jpg")

    text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder")
    text_encoder.to(device)

    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode("cat", add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    placeholder_token_ids = tokenizer.convert_tokens_to_ids([placeholder_token])
    token_id = placeholder_token_ids[0]
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data

    text = prompt
    templates = [
        "{} bag",
    ]
    if arg.use_prompt:
        text = random.choice(templates).format(prompt[0])

    text_input = tokenizer(
        text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    # with torch.no_grad():
    #     text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    # print(tokenizer.get_vocab())
    # print(text_embeddings.shape)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet")
    unet.to(device)

    scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)

    for _ in range(num_picture):
        e_vec = normal.sample((1,))

        embed = mapper(e_vec)
        token_embeds[token_id] = embed
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            # generator=generator,
        )
        latents = latents.to(device)
        latents = latents * scheduler.init_noise_sigma

        with torch.no_grad():
            hiddenstate = text_encoder(text_input.input_ids.to(device))
            text_embeddings = hiddenstate[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        semantic, score = count_relate(pil_images[0], model, processor)
        save_pic(emotion, pil_images[0], save_dir, semantic, score)


@torch.no_grad()
def emo_cls(cur_dir, device, weight):
    classifier = emo_classifier().to(device)
    state = torch.load(weight, map_location=device)
    classifier.load_state_dict(state)
    classifier.eval()

    CLIPmodel = CLIPModel.from_pretrained("clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("clip-vit-large-patch14")

    class EmoDataset(Dataset):
        def __init__(self, data_root, processor):
            self.emotion_list_8 = {"amusement": 0,
                                   "awe": 1,
                                   "contentment": 2,
                                   "excitement": 3,
                                   "anger": 4,
                                   "disgust": 5,
                                   "fear": 6,
                                   "sadness": 7}
            self.emotion_list_2 = {"amusement": 0,
                                   "awe": 0,
                                   "contentment": 0,
                                   "excitement": 0,
                                   "anger": 1,
                                   "disgust": 1,
                                   "fear": 1,
                                   "sadness": 1
                                   }
            self.tfm = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            self.image_paths = []
            self.processor = processor
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
            data = self.processor(images=image, return_tensors="pt", padding=True)
            data['pixel_values'] = data['pixel_values'].squeeze(0)
            example['image'] = data
            # data = self.model.get_image_features(**data)
            example['emotion_8'] = self.emotion_list_8[path.split('/')[-2]]
            example['emotion_2'] = self.emotion_list_2[path.split('/')[-2]]
            return example

    val_dataset = EmoDataset(cur_dir, processor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    picture_num = len(val_dataset)
    val_loader = tqdm(val_loader, file=sys.stdout)
    score_8 = 0
    score_2 = 0
    acc_num_2 = 0
    acc_num_8 = 0

    def eightemotion(Emo, Emo_num, Emo_score, pre, label, correct):

        for i in range(label.shape[0]):
            emo_label = label[i][0].item()
            Emo[emo_label] += correct[i].item()
            Emo_num[emo_label] += 1
            Emo_score[emo_label] += pre[i][emo_label]
        return Emo, Emo_num, Emo_score

    Emo = [0] * 8
    Emo_num = [0] * 8
    Emo_score = [0] * 8
    Emotion = ["amusement", "awe", "contentment",
               "excitement",
               "anger",
               "disgust",
               "fear",
               "sadness"
               ]
    for step, data in enumerate(val_loader):
        images = data['image'].to(device)
        clip = CLIPmodel.get_image_features(**images)
        pred = classifier(clip.to(device))
        labels_8 = data['emotion_8'].to(device).unsqueeze(1)
        labels_2 = data['emotion_2'].to(device).unsqueeze(1)
        pred_emotion_8 = torch.argmax(pred, dim=1, keepdim=True)
        p_8 = F.softmax(pred)
        p_2 = p_8.reshape((p_8.shape[0], 2, 4))
        p_2 = torch.sum(p_2, dim=2)
        p_2 = p_2.reshape((p_8.shape[0], -1))

        pred_emotion_2 = torch.argmax(p_2, dim=1, keepdim=True)

        pred_score_8 = torch.gather(p_8, dim=1, index=labels_8)
        pred_score_2 = torch.gather(p_2, dim=1, index=labels_2)

        acc_num_2 += (labels_2 == pred_emotion_2).sum().item()
        score_2 += torch.sum(pred_score_2).item()
        acc_num_8 += (labels_8 == pred_emotion_8).sum().item()
        score_8 += torch.sum(pred_score_8).item()
        eightemotion(Emo, Emo_num, Emo_score, p_8, labels_8, (labels_8 == pred_emotion_8))
    acc_8 = (acc_num_8 / picture_num) * 100
    total_score_8 = score_8 / picture_num
    acc_2 = (acc_num_2 / picture_num) * 100
    total_score_2 = score_2 / picture_num
    with open(os.path.join(cur_dir, 'evaluation.txt'), "a") as f:
        f.write(f'emo_score (8 class): {total_score_8:.2f}' + '\n')
        f.write(f'accuracy (8 class): {acc_8:.2f}%' + '\n')
        f.write(f'emo_score (2 class): {total_score_2:.2f}' + '\n')
        f.write(f'accuracy (2 class): {acc_2:.2f}%' + '\n')
        for i in range(8):
            tmp = Emo[i] / Emo_num[i] * 100
            f.write(f'{Emotion[i]} accuracy:{tmp:.2f}% score:{(Emo_score[i]/Emo_num[i]):.2f} \n')


def generate(cur_dir, device,model, num_fc_layers=1, need_LN=False, need_ReLU=False, need_Dropout=False, use_prompt=False):
    emotion_list = ["amusement", "excitement", "awe", "contentment", "fear", "disgust", "anger", "sadness"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_picture', type=int, default=10)
    parser.add_argument('--repo_id', type=str, default="stable-diffusion-v1-5/")
    parser.add_argument('--device', type=str, default=device)
    #####################################################################################
    parser.add_argument('--working_path', type=str, default=cur_dir)
    parser.add_argument('--mapper_name', type=str, default=model)
    parser.add_argument('--num_fc_layers', type=int, default=num_fc_layers)
    parser.add_argument("--need_LN", type=bool, default=need_LN)
    parser.add_argument("--need_ReLU", type=bool, default=need_ReLU)
    parser.add_argument("--need_Dropout", type=bool, default=need_Dropout)
    parser.add_argument("--use_prompt", type=bool, default=use_prompt)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    opt = parser.parse_args()
    for emo in emotion_list:
        inference(opt, emo)


if __name__ == "__main__":
    import json

    file = [
        "runs/test",
    ]
    # choose which epoch do you want to generate
    epochs = [0]
    device = "cuda:0"

    # emotion_classifier's weight
    weight = "weights/Clip_emotion_classifier/time_2023-11-12_03-29-best.pth"

    # 从 JSON 文件中读取参数
    for f in file:
        with open(f'{f}/params.json', 'r') as f:
            params_json = f.read()

        params = json.loads(params_json)
        globals().update(params)
        origin = output_dir
        try:
            for i in epochs:
                output_dir = os.path.join(origin, str(i))
                # use_prompt = True
                # generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout, use_prompt)
                generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout)
                emo_cls(output_dir, device, weight)
        except:
            output_dir = origin
            # use_prompt = True
            # generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout, use_prompt)
            generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout)
            emo_cls(output_dir, device, weight)
