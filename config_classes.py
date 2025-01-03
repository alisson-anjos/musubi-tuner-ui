# config_classes.py
from typing import List, Optional
import toml
import json
import os
from collections import OrderedDict


class GeneralConfig:
    def __init__(
        self,
        resolution: Optional[List[int]] = None,
        caption_extension: Optional[str] = None,
        batch_size: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
    ):
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.batch_size = batch_size
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale

    def to_toml(self) -> OrderedDict:
        config = OrderedDict()
        config["general"] = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "batch_size": self.batch_size,
            "enable_bucket": self.enable_bucket,
            "bucket_no_upscale": self.bucket_no_upscale,
        }
        
        config["datasets"] = []
    
        return config


class Dataset:
    def __init__(self, type: str):
        self.type = type

    def to_toml(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class ImageDataset(Dataset):
    def __init__(
        self,
        image_directory: Optional[str] = None,
        image_jsonl_file: Optional[str] = None,
        caption_extension: Optional[str] = None,
        resolution: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        enable_bucket: Optional[bool] = None,
        bucket_no_upscale: Optional[bool] = None,
        cache_directory: Optional[str] = None,
    ):
        super().__init__("image")
        self.image_directory = image_directory
        self.image_jsonl_file = image_jsonl_file
        self.caption_extension = caption_extension
        self.resolution = resolution
        self.batch_size = batch_size
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.cache_directory = cache_directory

    def to_toml(self) -> OrderedDict:
        dataset_config = OrderedDict()
        
        if self.image_directory:
            dataset_config['image_directory'] = self.image_directory
        if self.image_jsonl_file:
            dataset_config['image_jsonl_file'] = self.image_jsonl_file
        if self.caption_extension:
            dataset_config['caption_extension'] = self.caption_extension
        if self.resolution:
            dataset_config['resolution'] = self.resolution
        if self.batch_size is not None:
            dataset_config['batch_size'] = self.batch_size
        if self.enable_bucket is not None:
            dataset_config['enable_bucket'] = self.enable_bucket
        if self.bucket_no_upscale is not None:
            dataset_config['bucket_no_upscale'] = self.bucket_no_upscale
        if self.cache_directory:
            dataset_config['cache_directory'] = self.cache_directory
            
        return dataset_config


class VideoDataset(Dataset):
    def __init__(
        self,
        video_directory: Optional[str] = None,
        video_jsonl_file: Optional[str] = None,
        caption_extension: Optional[str] = None,
        resolution: Optional[List[int]] = None,
        target_frames: Optional[List[int]] = None,
        frame_extraction: Optional[str] = "head",
        frame_stride: Optional[int] = None,
        frame_sample: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_bucket: Optional[bool] = None,
        bucket_no_upscale: Optional[bool] = None,
        cache_directory: Optional[str] = None,
    ):
        super().__init__("video")
        self.video_directory = video_directory
        self.video_jsonl_file = video_jsonl_file
        self.caption_extension = caption_extension
        self.resolution = resolution
        self.target_frames = target_frames
        self.frame_extraction = frame_extraction
        self.frame_stride = frame_stride
        self.frame_sample = frame_sample
        self.batch_size = batch_size
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.cache_directory = cache_directory

    def to_toml(self) -> str:
        dataset_config = OrderedDict()
        
        if self.video_directory:
            dataset_config['video_directory'] = self.video_directory
        if self.video_jsonl_file:
            dataset_config['video_jsonl_file'] = self.video_jsonl_file
        if self.caption_extension:
            dataset_config['caption_extension'] = self.caption_extension
        if self.resolution:
            dataset_config['resolution'] = self.resolution
        if self.target_frames:
            dataset_config['target_frames'] = self.target_frames
        if self.frame_extraction:
            dataset_config['frame_extraction'] = self.frame_extraction
        if self.frame_stride:
            dataset_config['frame_stride'] = self.frame_stride
        if self.frame_sample:
            dataset_config['frame_sample'] = self.frame_sample
        if self.batch_size is not None:
            dataset_config['batch_size'] = self.batch_size
        if self.enable_bucket is not None:
            dataset_config['enable_bucket'] = self.enable_bucket
        if self.bucket_no_upscale is not None:
            dataset_config['bucket_no_upscale'] = self.bucket_no_upscale
        if self.cache_directory:
            dataset_config['cache_directory'] = self.cache_directory
            
        return dataset_config


class DatasetConfig:
    def __init__(self, general: GeneralConfig, datasets: List[Dataset]):
        self.general = general
        self.datasets = datasets

    def to_toml(self) -> OrderedDict:
        config = self.general.to_toml()
        for dataset in self.datasets:
            config['datasets'].append(dataset.to_toml())
        return config
      
    def save(self, path) -> None:
      
      config = self.to_toml()
      
      dataset_config_file = f"dataset_config.toml"
      dataset_config_path_full = os.path.join(path, dataset_config_file)
      with open(dataset_config_path_full, "w") as f:
          toml.dump(config, f)


class TrainingConfig:
    def __init__(
        self,
        dit_path: str,
        vae_path: str,
        llm_path: str,
        clip_path: str,
        dataset_config: str,
        output_dir: str,
        output_name: str,
        log_dir: str = "./log",
        mixed_precision: str = "bf16",
        optimizer_type: str = "adamw8bit",
        learning_rate: float = 1e-3,
        gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 4,
        max_data_loader_n_workers: int = 2,
        persistent_data_loader_workers: bool = True,
        network_module: str = "networks.lora",
        network_dim: int = 32,
        network_alpha: int = 16,
        timestep_sampling: str = "sigmoid",
        discrete_flow_shift: float = 1.0,
        max_train_epochs: int = 16,
        save_every_n_epochs: int = 1,
        seed: int = 42,
        num_cpu_threads_per_process: int = 1,
        fp8_base: bool = True,
        enable_lowvram: bool = False,
        blocks_to_swap: int = 20,
        attention: str = "sdpa",
    ):
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.llm_path = llm_path
        self.clip_path = clip_path
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.output_name = output_name
        self.mixed_precision = mixed_precision
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_data_loader_n_workers = max_data_loader_n_workers
        self.persistent_data_loader_workers = persistent_data_loader_workers
        self.network_module = network_module
        self.network_dim = network_dim
        self.network_alpha = network_alpha
        self.timestep_sampling = timestep_sampling
        self.discrete_flow_shift = discrete_flow_shift
        self.max_train_epochs = max_train_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.seed = seed
        self.log_dir = log_dir
        self.num_cpu_threads_per_process = num_cpu_threads_per_process
        self.fp8_base = fp8_base
        self.enable_lowvram = enable_lowvram
        self.blocks_to_swap = blocks_to_swap
        self.attention = attention


    def generate_command(self) -> str:
        
        flags = [
            f"echo \"Running cache_latents.py..\" &&",
            f"python cache_latents.py --dataset_config {self.dataset_config} ",
            f"--vae {self.vae_path} ",
            f"--vae_chunk_size 32 --vae_tiling && "
            f"echo \"Running cache_text_encoder_outputs.py...\" && "
            f"python cache_text_encoder_outputs.py --dataset_config {self.dataset_config} "
            f"--text_encoder1 {self.llm_path} "
            f"--text_encoder2 {self.clip_path} --batch_size 16 && "
            f"echo \"Running the training command...\" && "
            f"accelerate launch --num_cpu_threads_per_process {self.num_cpu_threads_per_process}",
            f"--mixed_precision {self.mixed_precision}",
            f"hv_train_network.py",
            f"--dit {self.dit_path}",
            f"--dataset_config {self.dataset_config}",
            "--sdpa" if self.attention == "sdpa" else "--sage_attn" if self.attention == "sage_attn" else "--flash_attn" if self.attention == "flash_attn" else "--xformers" if self.attention == "xformers" else "--sdpa",
            "--fp8_base" if self.fp8_base else "",
            f"--blocks_to_swap {self.blocks_to_swap} --fp8_llm" if self.enable_lowvram else "",
            f"--optimizer_type {self.optimizer_type}",
            f"--learning_rate {self.learning_rate}",
            "--gradient_checkpointing" if self.gradient_checkpointing else "",
            f"--gradient_accumulation_steps {self.gra}",
            f"--max_data_loader_n_workers {self.max_data_loader_n_workers}",
            "--persistent_data_loader_workers" if self.persistent_data_loader_workers else "",
            f"--network_module {self.network_module}",
            f"--network_dim {self.network_dim}",
            f"--network_alpha {self.network_alpha}",
            f"--timestep_sampling {self.timestep_sampling}",
            f"--discrete_flow_shift {self.discrete_flow_shift}",
            f"--max_train_epochs {self.max_train_epochs}",
            f"--save_every_n_epochs {self.save_every_n_epochs}",
            f"--seed {self.seed}",
            f"--logging_dir {self.log_dir}",
            f"--log_with \"tensorboard\"",
            f"--output_dir {self.output_dir}",
            f"--output_name {self.output_name}",
        ]
        # Filter out empty flags
        flags = [flag for flag in flags if flag]
        return " ".join(flags)
      
    def save(self, path) -> None:
      
      training_config = {
          "dit_path": self.dit_path,
          "dataset_config": self.dataset_config,
          "output_dir": self.output_dir,
          "output_name": self.output_name,
          "mixed_precision": self.mixed_precision,
          "optimizer_type": self.optimizer_type,
          "learning_rate": self.learning_rate,
          "gradient_checkpointing": self.gradient_checkpointing,
          "max_data_loader_n_workers": self.max_data_loader_n_workers,
          "persistent_data_loader_workers": self.persistent_data_loader_workers,
          "network_module": self.network_module,
          "network_dim": self.network_dim,
          "timestep_sampling": self.timestep_sampling,
          "discrete_flow_shift": self.discrete_flow_shift,
          "max_train_epochs": self.max_train_epochs,
          "save_every_n_epochs": self.save_every_n_epochs,
          "seed": self.seed,
          "num_cpu_threads_per_process": self.num_cpu_threads_per_process,
          "fp8_base": self.fp8_base,
          "attention": self.attention
      }
      
      training_config_file = f"training_command.toml"
      training_config_path_full = os.path.join(path, training_config_file)
      with open(training_config_path_full, "w") as f:
          toml.dump(training_config, f)
