import itertools
import sys

import torch
import os
import torch.functional as F
from model import *
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL, \
    DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
from tqdm.auto import tqdm
import argparse
from accelerate.utils import set_seed
from torchvision import transforms
import pickle
iiiii


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
def inference(arg):
    working_path = arg.working_path
    device = torch.device(arg.device if torch.cuda.is_available() else "cpu")  # TODO
    if arg.seed is not None:
        set_seed(arg.seed)
    placeholder_token = f"<nomatter>"

    prompt = [placeholder_token]
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 50  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    # generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
    batch_size = len(prompt)
    num_picture = arg.num_picture
    repo_id = arg.repo_id
    model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    linear_project = model.text_projection
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

    classifier = BackBone().to(device)
    state = torch.load("weights/image_encoder/2023-08-22-best.pth", map_location=device)
    classifier.load_state_dict(state)
    classifier.eval()
    @torch.no_grad()
    def save_pic(emotion, img, path, semantic, score, device):
        idx2label = [
            "amusement",
            "awe",
            "contentment",
            "excitement",
            "anger",
            "disgust",
            "fear",
            "sadness"
        ]
        classifier = BackBone().to(device)
        state = torch.load("/home/ubuntu/code/Emo-generation/weights/2023-08-22-best.pth", map_location=device)
        classifier.load_state_dict(state)
        classifier.eval()
        tfm = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        something = tfm(img).unsqueeze(0)
        pred = F.softmax(classifier(something.to(device)))
        index = torch.argmax(pred, dim=1)
        related_score = pred[0, index.item()]
        pred_emotion = idx2label[index.item()]
        if path is not None:
            os.makedirs(path, exist_ok=True)
        try:
            files = sorted([x for x in os.listdir(path) if x.endswith(".jpg")], key=lambda x: int(x.split("_")[0]))
            num = int(files[-1].split("_")[0])
            img.save(f"{path}/{num + 1}_{emotion}_{semantic}_{score:.2f}-{pred_emotion}-{related_score:.2f}.jpg")
        except:
            img.save(f"{path}/0_{emotion}_{semantic}_{score:.2f}-{pred_emotion}-{related_score:.2f}.jpg")

    @torch.no_grad()
    def generate_origin_image(text, args, path, tokenizer, text_encoder, unet, classifier, processor, model):
        if os.path.isfile(f"{path}/{text}/origin_evaluation.txt"):
            return 0
        images = []
        pipeline = DiffusionPipeline.from_pretrained(
            args.repo_id,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            safety_checker=None,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(text, num_inference_steps=50).images[0]
            images.append(image)
        os.makedirs(f"{path}/{text}/origin", exist_ok=True)
        for image in images:
            try:
                files = sorted([x for x in os.listdir(f"{path}/{text}/origin") if x.endswith(".jpg")], key=lambda x: int(x.split(".")[0]))
                num = int(files[-1].split(".")[0])
                image.save(f"{path}/{text}/origin/{num + 1}.jpg")
            except:
                image.save(f"{path}/{text}/origin/0.jpg")

        Emotion = ["amusement", "awe", "contentment", "excitement",
                   "anger", "disgust", "fear", "sadness" ]
        tfm = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        score = torch.tensor(([0]*8), dtype=torch.float32)
        for image in images:
            image = image.convert('RGB')
            img = tfm(image).unsqueeze(0)
            tmp = F.softmax(classifier(img.to(device))).squeeze(0)
            score += tmp.cpu()
        score = score / len(images)

        with open(os.path.join(f"{path}/{text}", 'origin_evaluation.txt'), "w") as f:
            for i in range(8):
                f.write(f'{Emotion[i]} score:{score[i].item():.2f} \n')

        # calculate text-embedding of attribute
        input = processor(images=images, text=attribute, return_tensors="pt", padding=True).to(device)
        output = model(**input)
        text_embed = output.text_embeds  # (1,768)
        img_embed = output.image_embeds  # (n,768)
        img_embed = torch.sum(img_embed, dim=0, keepdim=True)/len(images)
        torch.save(text_embed, f"{path}/{text}/origin_text_embed.pt")
        torch.save(img_embed, f"{path}/{text}/origin_img_embed.pt")


    text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder")
    text_encoder.to(device)

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
    max_length = tokenizer.model_max_length
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet")
    unet.to(device)

    scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    emotion_list = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]
    combinations = list(itertools.combinations(emotion_list, 2))

    for (emotion_1, emotion_2) in combinations:
        tmp_img_list = []
        save_dir = f"{working_path}/{emotion_1}/{emotion_2}"
        e_mean = torch.load(f"./emo_space/property/{emotion_1}_mean_v2.pt")
        e_std = torch.load(f"./emo_space/property/{emotion_1}_std_v2.pt")
        normal_1 = torch.distributions.Normal(e_mean, e_std)

        e_mean = torch.load(f"./emo_space/property/{emotion_2}_mean_v2.pt")
        e_std = torch.load(f"./emo_space/property/{emotion_2}_std_v2.pt")
        normal_2 = torch.distributions.Normal(e_mean, e_std)

        text = f"{prompt[0]} vishal"
        text_input = tokenizer(
            text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        for _ in range(num_picture):

            e_vec_1 = normal_1.sample((1,))
            e_vec_2 = normal_2.sample((1,))

            embed_1 = mapper(e_vec_1)
            token_embeds[token_id] = embed_1
            embed_2 = mapper(e_vec_2)
            token_embeds[33648] = embed_2
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
            tmp_img_list.append(pil_images[0])
            semantic, score = count_relate(pil_images[0], model, processor)
            save_pic(emotion_1, pil_images[0], save_dir, semantic, score, device)

def generate(cur_dir, device, model, num_fc_layers=1, need_LN=False, need_ReLU=False, need_Dropout=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_picture', type=int, default=130)  ##############  TODO
    parser.add_argument('--repo_id', type=str, default="/mnt/d/model/stable-diffusion-v1-5/")
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--num_validation_images', type=int, default=10)
    #####################################################################################
    parser.add_argument('--working_path', type=str, default=cur_dir)
    parser.add_argument('--mapper_name', type=str, default=model)
    parser.add_argument('--num_fc_layers', type=int, default=num_fc_layers)
    parser.add_argument("--need_LN", type=bool, default=need_LN)
    parser.add_argument("--need_ReLU", type=bool, default=need_ReLU)
    parser.add_argument("--need_Dropout", type=bool, default=need_Dropout)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    opt = parser.parse_args()
    inference(opt)


if __name__ == "__main__":
    import json

    file = [
        "test"
    ]
    epochs = [0]
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
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
                generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout)
        except:
            output_dir = origin
            generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout)
