

import torch
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import gradio as gr
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    num_images: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None

class ModelCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model_id: str, load_func: callable, cache_name: str) -> Any:
        try:
            logger.info(f"Loading {cache_name}")
            return load_func(model_id)
        except Exception as e:
            logger.error(f"Error loading model {cache_name}: {str(e)}")
            raise

class EnhancedBanglaSDGenerator:
    def __init__(
        self,
        banglaclip_weights_path: str,
        cache_dir: str,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.cache = ModelCache(Path(cache_dir))
        self._initialize_models(banglaclip_weights_path)
        self._load_context_data()

    def _initialize_models(self, banglaclip_weights_path: str):
        try:
            # Initialize translation models
            self.bn2en_model_name = "Helsinki-NLP/opus-mt-bn-en"
            self.translator = self.cache.load_model(
                self.bn2en_model_name,
                MarianMTModel.from_pretrained,
                "translator"
            ).to(self.device)
            self.trans_tokenizer = MarianTokenizer.from_pretrained(self.bn2en_model_name)

            # Initialize CLIP models
            self.clip_model_name = "openai/clip-vit-base-patch32"
            self.bangla_text_model = "csebuetnlp/banglabert"
            self.banglaclip_model = self._load_banglaclip_model(banglaclip_weights_path)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.bangla_text_model)

            # Initialize Stable Diffusion
            self._initialize_stable_diffusion()

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def _initialize_stable_diffusion(self):
        """Initialize Stable Diffusion pipeline with optimized settings."""
        self.pipe = self.cache.load_model(
            "runwayml/stable-diffusion-v1-5",
            lambda model_id: StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None
            ),
            "stable_diffusion"
        )

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
        self.pipe = self.pipe.to(self.device)

        # Memory optimization
        self.pipe.enable_attention_slicing()
        if torch.cuda.is_available():
            self.pipe.enable_sequential_cpu_offload()

    def _load_banglaclip_model(self, weights_path: str) -> CLIPModel:
        try:
            if not Path(weights_path).exists():
                raise FileNotFoundError(f"BanglaCLIP weights not found at {weights_path}")

            clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            state_dict = torch.load(weights_path, map_location=self.device)

            cleaned_state_dict = {
                k.replace('module.', '').replace('clip.', ''): v
                for k, v in state_dict.items()
                if k.replace('module.', '').replace('clip.', '').startswith(('text_model.', 'vision_model.'))
            }

            clip_model.load_state_dict(cleaned_state_dict, strict=False)
            return clip_model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load BanglaCLIP model: {str(e)}")
            raise

    def _load_context_data(self):
        """Load location and scene context data."""
        self.location_contexts = {
            'কক্সবাজার': 'Cox\'s Bazar beach, longest natural sea beach in the world, sandy beach',
            'সেন্টমার্টিন': 'Saint Martin\'s Island, coral island, tropical paradise',
            'সুন্দরবন': 'Sundarbans mangrove forest, Bengal tigers, riverine forest'
        }

        self.scene_contexts = {
            'সৈকত': 'beach, seaside, waves, sandy shore, ocean view',
            'সমুদ্র': 'ocean, sea waves, deep blue water, horizon',
            'পাহাড়': 'mountains, hills, valleys, scenic landscape'
        }

    def _translate_text(self, bangla_text: str) -> str:
        """Translate Bangla text to English."""
        inputs = self.trans_tokenizer(bangla_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.translator.generate(**inputs)

        translated = self.trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def _get_text_embedding(self, text: str):
        """Get text embedding from BanglaCLIP model."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.banglaclip_model.get_text_features(**inputs)

        return outputs

    def generate_image(
        self,
        bangla_text: str,
        config: Optional[GenerationConfig] = None
    ) -> Tuple[List[Any], str]:
        if not bangla_text.strip():
            raise ValueError("Empty input text")

        config = config or GenerationConfig()

        try:
            if config.seed is not None:
                torch.manual_seed(config.seed)

            enhanced_prompt = self._enhance_prompt(bangla_text)
            negative_prompt = self._get_negative_prompt()

            with torch.autocast(self.device.type):
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=config.num_images,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale
                )

            return result.images, enhanced_prompt

        except Exception as e:
            logger.error(f"Error during image generation: {str(e)}")
            raise

    def _enhance_prompt(self, bangla_text: str) -> str:
        """Enhance prompt with context and style information."""
        translated_text = self._translate_text(bangla_text)

        # Gather contexts
        contexts = []
        contexts.extend(context for loc, context in self.location_contexts.items() if loc in bangla_text)
        contexts.extend(context for scene, context in self.scene_contexts.items() if scene in bangla_text)

        # Add photo style
        photo_style = [
            "professional photography",
            "high resolution",
            "4k",
            "detailed",
            "realistic",
            "beautiful composition"
        ]

        # Combine all parts
        all_parts = [translated_text] + contexts + photo_style
        return ", ".join(dict.fromkeys(all_parts))

    def _get_negative_prompt(self) -> str:
        return (
            "blurry, low quality, pixelated, cartoon, anime, illustration, "
            "painting, drawing, artificial, fake, oversaturated, undersaturated"
        )

    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'banglaclip_model'):
            del self.banglaclip_model
        if hasattr(self, 'translator'):
            del self.translator
        torch.cuda.empty_cache()
        gc.collect()

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    cache_dir = Path("model_cache")
    generator = None

    def initialize_generator():
        nonlocal generator
        if generator is None:
            generator = EnhancedBanglaSDGenerator(
                banglaclip_weights_path="banglaclip_model_epoch_10_quantized.pth",
                cache_dir=str(cache_dir)
            )
        return generator

    def cleanup_generator():
        nonlocal generator
        if generator is not None:
            generator.cleanup()
            generator = None

    def generate_images(text: str, num_images: int, steps: int, guidance_scale: float, seed: Optional[int]) -> Tuple[List[Any], str]:
        if not text.strip():
            return None, "দয়া করে কিছু টেক্সট লিখুন"

        try:
            gen = initialize_generator()
            config = GenerationConfig(
                num_images=int(num_images),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                seed=int(seed) if seed else None
            )

            images, prompt = gen.generate_image(text, config)
            cleanup_generator()
            return images, prompt

        except Exception as e:
            logger.error(f"Error in Gradio interface: {str(e)}")
            cleanup_generator()
            return None, f"ছবি তৈরি ব্যর্থ হয়েছে: {str(e)}"

    # Create Gradio interface
    demo = gr.Interface(
        fn=generate_images,
        inputs=[
            gr.Textbox(
                label="বাংলা টেক্সট লিখুন",
                placeholder="যেকোনো বাংলা টেক্সট লিখুন...",
                lines=3
            ),
            gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=1,
                label="ছবির সংখ্যা"
            ),
            gr.Slider(
                minimum=20,
                maximum=100,
                step=1,
                value=50,
                label="স্টেপস"
            ),
            gr.Slider(
                minimum=1.0,
                maximum=20.0,
                step=0.5,
                value=7.5,
                label="গাইডেন্স স্কেল"
            ),
            gr.Number(
                label="সীড (ঐচ্ছিক)",
                precision=0
            )
        ],
        outputs=[
            gr.Gallery(label="তৈরি করা ছবি"),
            gr.Textbox(label="ব্যবহৃত প্রম্পট")
        ],
        title="বাংলা টেক্সট থেকে ছবি তৈরি",
        description="যেকোনো বাংলা টেক্সট দিয়ে উচ্চমানের ছবি তৈরি করুন"
    )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    # Fixed queue configuration for newer Gradio versions
    demo.queue().launch(share=True)
