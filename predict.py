# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from typing import Union

import os
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import PIL.Image
import re
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_CACHE = "model_cache"
# BASE_URL = f"https://weights.replicate.delivery/default/StepVideo/{MODEL_CACHE}/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

os.environ["HF_API_TOKEN"] = "SET-HERE" # You'll need to go gv-hf/paligemma2-10b-mix-448 and accept the terms of service to get a token

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model and necessary components into memory"""
        # Get Hugging Face token from environment variable
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable is required")

        self.model_id = "gv-hf/paligemma2-10b-mix-448"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pass token for authentication
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            token=hf_token
        ).eval().to(self.device)
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id, token=hf_token)
        
        # Load segmentation parameters
        self._MODEL_PATH = 'vae-oid.npz'  # Ensure this file is in the model directory
        # Regular expression to parse segmentation output
        self._SEGMENT_DETECT_RE = re.compile(
            r'(.*?)' +
            r'<loc(\d{4})>' * 4 + r'\s*' +
            '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
            r'\s*([^;<>]+)? ?(?:; )?',
        )
        # Load the VAE parameters for mask reconstruction
        self.params = self._load_params()
        self.reconstruct_masks_func = self._get_reconstruct_masks()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        # task: str = Input(description="Task to perform", choices=["generate_text", "segment"], default="generate_text"),
        text: str = Input(description="Input text for generation or segmentation", default=""),
        max_new_tokens: int = Input(description="Max new tokens for text generation", default=20),
    ) -> str:
        """Run a single prediction on the model."""
        input_image = PIL.Image.open(image).convert("RGB")
        result = self.infer(input_image, text, max_new_tokens)
        return result
        # elif task == "segment":
        #     output_image_path = self.parse_segmentation(input_image, text)
        #     return output_image_path

    def infer(
        self,
        image: PIL.Image.Image,
        text: str,
        max_new_tokens: int
    ) -> str:
        """Perform text generation inference."""
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return result[0][len(text):].lstrip("\n")

    def parse_segmentation(self, input_image, input_text) -> Path:
        """Parse segmentation output tokens into masks and create annotated image."""
        out = self.infer(input_image, input_text, max_new_tokens=200)
        objs = self.extract_objs(out.lstrip("\n"), input_image.size[0], input_image.size[1], unique_labels=True)
        labels = set(obj.get('name') for obj in objs if obj.get('name'))
        color_map = {l: self.color_palette[i % len(self.color_palette)] for i, l in enumerate(labels)}
        
        # Create an annotated image with segmentation masks
        annotated_img = self.create_annotated_image(input_image, objs, color_map)
        output_path = Path("output.png")
        annotated_img.save(output_path)
        return output_path

    def extract_objs(self, text, width, height, unique_labels=False):
        """Returns objs for a string with "<loc>" and "<seg>" tokens."""
        objs = []
        seen = set()
        while text:
            m = self._SEGMENT_DETECT_RE.match(text)
            if not m:
                break
            gs = list(m.groups())
            before = gs.pop(0)
            name = gs.pop()
            y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
            
            y1, x1, y2, x2 = map(round, (y1*height, x1*width, y2*height, x2*width))
            seg_indices = gs[4:20]
            if seg_indices[0] is None:
                mask = None
            else:
                seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
                m64, = self.reconstruct_masks_func(seg_indices[None])[..., 0]
                m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
                m64 = PIL.Image.fromarray((m64 * 255).astype('uint8'))
                mask = np.zeros([height, width])
                if y2 > y1 and x2 > x1:
                    mask[y1:y2, x1:x2] = np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0

            content = m.group()
            if before:
                objs.append(dict(content=before))
                content = content[len(before):]
            while unique_labels and name in seen:
                name = (name or '') + "'"
            seen.add(name)
            objs.append(dict(
                content=content, xyxy=(x1, y1, x2, y2), mask=mask, name=name))
            text = text[len(before) + len(content):]

        if text:
            objs.append(dict(content=text))

        return objs

    def create_annotated_image(self, image, objs, color_map):
        """Create an annotated image with segmentation masks and bounding boxes."""
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for obj in objs:
            color = color_map.get(obj.get('name'), '#ffffff')
            if obj.get('mask') is not None:
                mask = obj['mask']
                ax.imshow(mask, cmap='jet', alpha=0.5)
                if obj.get('name'):
                    ax.text(0, 0, obj['name'], bbox=dict(facecolor=color, alpha=0.5), transform=ax.transAxes)
            elif obj.get('xyxy') is not None:
                x1, y1, x2, y2 = obj['xyxy']
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                if obj.get('name'):
                    ax.text(x1, y1 - 10, obj['name'], color=color, fontsize=12, fontweight='bold')

        plt.axis('off')
        # Save the figure to a PIL image
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        annotated_image = PIL.Image.frombytes('RGB', (width, height), fig.canvas.tostring_rgb())
        plt.close(fig)
        return annotated_image

    def _load_params(self):
        """Load parameters for mask reconstruction from the VAE model."""
        def _get_params(checkpoint):
            """Convert PyTorch checkpoint to Flax params."""
            def transp(kernel):
                return np.transpose(kernel, (2, 3, 1, 0))
            def conv(name):
                return {
                    'bias': checkpoint[name + '.bias'],
                    'kernel': transp(checkpoint[name + '.weight']),
                }
            def resblock(name):
                return {
                    'Conv_0': conv(name + '.0'),
                    'Conv_1': conv(name + '.2'),
                    'Conv_2': conv(name + '.4'),
                }
            return {
                '_embeddings': checkpoint['_vq_vae._embedding'],
                'Conv_0': conv('decoder.0'),
                'ResBlock_0': resblock('decoder.2.net'),
                'ResBlock_1': resblock('decoder.3.net'),
                'ConvTranspose_0': conv('decoder.4'),
                'ConvTranspose_1': conv('decoder.6'),
                'ConvTranspose_2': conv('decoder.8'),
                'ConvTranspose_3': conv('decoder.10'),
                'Conv_1': conv('decoder.12'),
            }
        with open(self._MODEL_PATH, 'rb') as f:
            params = _get_params(dict(np.load(f)))
        return params

    def _get_reconstruct_masks(self):
        """Reconstruct masks from codebook indices."""
        params = self.params

        class ResBlock(nn.Module):
            features: int

            @nn.compact
            def __call__(self, x):
                original_x = x
                x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
                x = nn.relu(x)
                x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
                x = nn.relu(x)
                x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
                return x + original_x

        class Decoder(nn.Module):
            """Upscale quantized vectors to mask."""

            @nn.compact
            def __call__(self, x):
                num_res_blocks = 2
                dim = 128
                num_upsample_layers = 4

                x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
                x = nn.relu(x)

                for _ in range(num_res_blocks):
                    x = ResBlock(features=dim)(x)

                for _ in range(num_upsample_layers):
                    x = nn.ConvTranspose(
                        features=dim,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding=2,
                        transpose_kernel=True,
                    )(x)
                    x = nn.relu(x)
                    dim //= 2

                x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)
                return x

        def reconstruct_masks(codebook_indices):
            quantized = self._quantized_values_from_codebook_indices(
                codebook_indices, params['_embeddings']
            )
            return Decoder().apply({'params': params}, quantized)

        return jax.jit(reconstruct_masks, backend='cpu')

    def _quantized_values_from_codebook_indices(self, codebook_indices, embeddings):
        batch_size, num_tokens = codebook_indices.shape
        assert num_tokens == 16, codebook_indices.shape
        _, embedding_dim = embeddings.shape

        encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
        encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
        return encodings
