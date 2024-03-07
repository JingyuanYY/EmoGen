import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
import datetime
import numpy as np
import pickle
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.tensorboard import SummaryWriter

from model import *
from inference import inference, generate
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
import json
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


def parse_args(pretrained_model_name_or_path, emotion, train_data_dir, learnable_property, max_train_steps,
               num_train_epochs, attr_rate, threshold, seed, emo_rate,
               learning_rate, output_dir, model, num_fc_layers, need_LN=False, need_ReLU=False, need_Dropout=False):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model,
        help="choose the model use to map",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=threshold,
        help="choose the model use to map",
    )
    parser.add_argument(
        "--num_fc_layers",
        type=int,
        default=num_fc_layers,
        help="If the model is MLP, how many fully connected layers do you need?",
    )
    parser.add_argument(
        "--attr_rate",
        type=float,
        default=attr_rate,
    )
    parser.add_argument(
        "--emo_rate",
        type=float,
        default=emo_rate,
    )
    parser.add_argument(
        "--need_LN",
        type=bool,
        default=need_LN,
    )
    parser.add_argument(
        "--need_ReLU",
        type=bool,
        default=need_ReLU,
    )
    parser.add_argument(
        "--need_Dropout",
        type=bool,
        default=need_Dropout,
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=pretrained_model_name_or_path,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=emotion,
        help="Emotion to learn.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=train_data_dir, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<dummy>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="cat", help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=list, default=learnable_property,
                        help="Choose between 'object' and 'scene'")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=seed, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=max_train_steps,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="<dummy>",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20000000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", default=True,
        help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def generate_attr(processor, model, attribute_total):
    data_pro = processor(text=attribute_total, return_tensors="pt", padding=True).to(model.device)
    data_pro = model.module.get_text_features(**data_pro)
    return data_pro


def get_coefficient():
    with open('dataset_balance/attr_coefficient_emotion_scene.pkl', 'rb') as f:
        coffeicient = pickle.load(f)
    with open('dataset_balance/attr_coefficient_emotion_object.pkl', 'rb') as f:
        tmp = pickle.load(f)
        coffeicient.update(tmp)
    # coffeicient = {key: 1 if value > 0 else 0 for key, value in coffeicient.items()}
    return coffeicient

def gudiance_attr(attribute, tokenizer, text_encoder, weight_dtype):
    # TODO
    ids = tokenizer(
        attribute,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    vec = text_encoder(ids)[1].to(dtype=weight_dtype)
    return vec


def read_attr():
    properties = ["object", "scene"]
    attribute_pro = {"object": [], "scene": []}
    attribute_total = []
    attribute_emo = {}
    for property in properties:
        with open(f'dataset_balance/{property}_attr.pkl', 'rb') as f:
            useful_attr = pickle.load(f)
            tmp = []
            for key in useful_attr:
                tmp.extend(useful_attr[key])
                try:
                    attribute_emo[key].extend(useful_attr[key])
                except:
                    attribute_emo[key] = []
                    attribute_emo[key].extend(useful_attr[key])
            attribute_pro[property].extend(tmp)
            attribute_total.extend(tmp)
    return attribute_total, attribute_emo


class image_encoder(nn.Module):
    def __init__(self):
        super(image_encoder, self).__init__()
        self.resnet = BackBone()
        state = torch.load("/home/ubuntu/code/Emo-generation/weights/2023-08-22-best.pth")
        self.resnet.load_state_dict(state)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[1:-1])

    def forward(self, x):
        out = self.resnet(x)
        return out


class TextualInversionDataset(Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            learnable_property=None,  # [object, scene]
            emotion=None,
            size=512,
            repeats=1,
            flip_p=0.5,
            set="train",
            placeholder_token="*",
            center_crop=False,
    ):
        if learnable_property is None:
            learnable_property = ["scene"]
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.image_paths = []
        self.attribute_list = []
        if learnable_property != ["all"]:
            for folder in learnable_property:
                self.data_root = os.path.join(data_root, folder)
                with open(f'./dataset_balance/{folder}_norepeat.pkl', 'rb') as f:
                    tmp = pickle.load(f)
                    self.image_paths.extend(tmp)
            for path in self.image_paths:
                attribute = path.split('/')[-2].split(')')[-1].lower().replace(' ', '_')
                if (attribute in self.attribute_list) is False:
                    self.attribute_list.append(attribute)
        else:
            self.data_root = "EmoSet/image/"  # change it into your EmoSet file location
            for root, _, file_path in os.walk(self.data_root):
                for file in file_path:
                    if file.endswith("jpg"):
                        tmp = os.path.join(root, file)
                        attribute = tmp.split('/')[-2].split(')')[-1]
                        if (attribute in self.attribute_list) is False:
                            self.attribute_list.append(attribute.lower())
                        if emotion != "all":
                            if file.startswith(emotion):
                                self.image_paths.append(os.path.join(root, file))
                        else:
                            self.image_paths.append(os.path.join(root, file))
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.tfm = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def __len__(self):
        return self._length

    def get_all_attribute(self):
        return self.attribute_list

    def __getitem__(self, i):
        example = {}
        path = self.image_paths[i % self.num_images]
        image = Image.open(path)
        img_feat = self.tfm(image.copy())
        example["image"] = img_feat
        if self.learnable_property != "all":
            example["attribute"] = path.split('/')[-2].split(')')[-1].lower().replace(' ','_')
        else:
            example["attribute"] = ' '
        example["emotion"] = path.split('/')[-1].split('_')[0]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = placeholder_string

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size))

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def save_pic(img, path):
    if path is not None:
        os.makedirs(path, exist_ok=True)
    try:
        files = sorted([x for x in os.listdir(path) if x.endswith(".jpg")], key=lambda x: int(x.split(".")[0]))
        num = int(files[-1].split(".")[0])
        img.save(f"{path}/{num + 1}.jpg")
    except:
        img.save(f"{path}/0.jpg")


def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
    # writer = SummaryWriter(log_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    model = CLIPModel.from_pretrained("model/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("model/clip-vit-large-patch14")

    # Load image encoder and FC
    encoder = image_encoder()

    classifier = emo_classifier()
    state = torch.load("weights/Clip_emotion_classifier/time_2023-11-12_03-29-best.pth", map_location='cpu')
    classifier.load_state_dict(state)
    classifier.eval()

    model_dict = {
        "FC": lambda args: FC(),
        "MLP": lambda args: MLP(args.num_fc_layers, args.need_ReLU, args.need_LN, args.need_Dropout),
        "SimpleMLP": lambda args: SimpleMLP(args.need_ReLU, args.need_Dropout),
    }
    mapper = model_dict[args.model](args)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    classifier.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    parameters = list(mapper.parameters())

    optimizer = optimizer_cls(
        parameters,  # TODO need to change
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        emotion=args.emotion,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    if args.validation_epochs is not None:
        args.validation_steps = args.validation_epochs * len(train_dataset) // accelerator.num_processes

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler, encoder, mapper, model, classifier = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler, encoder, mapper, model, classifier
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    classifier.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Emotion_generation")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            args.resume_from_checkpoint = None
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # initialize the hook method
    grad_pseudo = torch.tensor([1], requires_grad=False)

    def change_grad(grad):
        return grad + grad_pseudo

    label2idx = {
      "amusement": 0,
      "awe": 1,
      "contentment": 2,
      "excitement": 3,
      "anger": 4,
      "disgust": 5,
      "fear": 6,
      "sadness": 7
    }
    total_attr, _ = read_attr()
    total_attr_embed = generate_attr(processor, model, total_attr).detach()
    attr_coefficient = get_coefficient()
    num = 0
    linear_project = model.module.text_projection.to(weight_dtype)
    for epoch in range(first_epoch, args.num_train_epochs):
        encoder.eval()
        mapper.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    num += total_batch_size
                continue

            with accelerator.accumulate(encoder):
                with accelerator.accumulate(mapper):
                    # Project image into emotion space
                    emo_v = encoder(batch["image"]).to(dtype=weight_dtype)

                    # Project emo_v into embedding space
                    pred_emd = mapper(emo_v)

                    # Set hook
                    pred_emd.register_hook(change_grad)

                    # Change the embedding of new token
                    token_embeds = text_encoder.module.get_input_embeddings().weight.data
                    token_embeds[placeholder_token_ids] = pred_emd

                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    output = text_encoder(batch["input_ids"])
                    encoder_hidden_states = output[0].to(dtype=weight_dtype)
                    pooled_output = output[1].to(dtype=weight_dtype)

                    project_semantic = linear_project(pooled_output)
                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss_reconstruction = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # loss of distance to attribute target

                    if batch["attribute"][0].lower() in total_attr:
                        score = F.cosine_similarity(total_attr_embed, project_semantic).unsqueeze(0)
                        fun_loss_attr = nn.CrossEntropyLoss()
                        index_attr = total_attr.index(batch["attribute"][0].lower())
                        index_attr = torch.tensor([index_attr]).detach().to(score.device)
                        loss_attr = fun_loss_attr(score, index_attr)
                    else:
                        loss_attr = torch.tensor([0.0], requires_grad=True).to(mapper.device)
                    if batch["attribute"][0] in attr_coefficient:
                        attr_rate = attr_coefficient[batch["attribute"][0]][label2idx[batch['emotion'][0]]].item()
                        if attr_rate < args.threshold:
                            attr_rate = 0
                    else:
                        attr_rate = 0

                    # emotion-guidance loss
                    fun_loss_emo = nn.CrossEntropyLoss()
                    pre_emo = classifier(project_semantic)
                    index_emo = label2idx[batch["emotion"][0]]
                    index_emo = torch.tensor([index_emo]).detach().to(pre_emo.device)
                    loss_emo = fun_loss_emo(pre_emo, index_emo)

                    # assume that distance under threshold is the same object
                    loss_forward = (1 - attr_rate)*loss_reconstruction + args.attr_rate * attr_rate * loss_attr + args.emo_rate * loss_emo

                    accelerator.backward(loss_forward)
                    grad_pseudo = text_encoder.module.get_input_embeddings().weight.grad[-1].detach().unsqueeze(0)

                    # fake loss in order to backward
                    loss_fake = torch.mean(pred_emd)
                    loss = 0 * loss_fake
                    accelerator.backward(loss)
                    text_encoder.module.get_input_embeddings().weight.grad[-1] *= 0

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[min(placeholder_token_ids): max(placeholder_token_ids) + 1] = False

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                    if global_step % args.validation_steps == 0:
                        tmp = os.path.join(args.output_dir, f"{global_step // len(train_dataloader)}")
                        os.makedirs(tmp, exist_ok=True)
                        torch.save(accelerator.unwrap_model(mapper).state_dict(),
                                   os.path.join(
                                       os.path.join(args.output_dir, f"{global_step // len(train_dataloader)}"),
                                       "mapper.pth"))
                    for tracker in accelerator.trackers:
                        tracker.writer.add_scalar("Loss", loss_forward, global_step)
                        tracker.writer.add_scalar("loss_reconstruction", loss_reconstruction, global_step)
                        tracker.writer.add_scalar("loss_attribute", loss_attr, global_step)
                        tracker.writer.add_scalar("loss_emo", loss_emo, global_step)
            logs = {"loss": loss.detach().item(), "loss_attr": loss_attr.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        torch.save(accelerator.unwrap_model(mapper).state_dict(), os.path.join(args.output_dir, "mapper.pth"))

    accelerator.end_training()


if __name__ == "__main__":
    import yaml

    def parameter(file_name):
        with open(file_name, 'r') as file:
            params = yaml.safe_load(file)

        args = parse_args(**params)
        params["project_name"] = os.path.basename(__file__)
        params_json = json.dumps(params)
        os.makedirs(f'{params["output_dir"]}',exist_ok=True)
        with open(f'{params["output_dir"]}/params.json', 'w') as f:
            f.write(params_json)
        return args

    # Choose your config file
    file_name = 'config/config.yaml'

    args = parameter(file_name)
    main(args)