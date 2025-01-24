import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from lavis.models import load_model_and_preprocess
from src.utils import load_bin_file, load_id2image_file, result_format, top_k_unique_in_order


class BLIPImageEncoder:
    def __init__(self, pretrained: str = 'pretrain_vitL'):
        self.model_name = "blip2_feature_extractor"
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model, self.vis_processors, _ = load_model_and_preprocess(
                name=self.model_name, model_type=pretrained, is_eval=True, device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}' with weights '{pretrained}': {e}")
    
    def process_image(self, image_path: str):
        """Process a single image and return a tensor."""
        try:
            with Image.open(image_path) as img:
                image = self.vis_processors['eval'](img).unsqueeze(0).to(self.device)
            return image
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")
            return None
    
    def encode_image(self, image_path_list: List[str]):
        id_to_image_dict = {}
        encoding_list = []

        for id, image_path in tqdm(enumerate(image_path_list), desc="Processing images", total=len(image_path_list)):
            id_to_image_dict[id] = image_path
            image_tensor = self.process_image(image_path)
            if image_tensor is None:
                continue
            
            try:
                features = self.model.extract_features({"image": image_tensor}, mode="image")
                image_feature = features.image_embeds_proj
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                encoding_list.extend(image_feature.squeeze(0).cpu().numpy().astype(np.float32).tolist())
            except Exception as e:
                print(f"Failed to extract features for image {image_path}: {e}")

        return id_to_image_dict, encoding_list


class BLIPSearchEngine:
    def __init__(self, bin_file: str, metadata: str):
        # Set device and load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.vis_processors, self.txt_processors = self.load_model_and_preprocess()
        
        # Load index and metadata
        self.index = load_bin_file(bin_file)
        self.id2image_fps = load_id2image_file(metadata)
    
    def load_model_and_preprocess(self):
        """Load model and necessary processors."""
        return load_model_and_preprocess(
            name='blip2_feature_extractor',
            model_type='pretrain_vitL',
            is_eval=True,
            device=self.device
        )
    
    def encode_text(self, query_text: str):
        """Extract features from the query text."""
        # Preprocess text and extract features
        processed_text = self.txt_processors["eval"](query_text)
        text_features = self.model.extract_features(
            {"text_input": [processed_text]},
            mode="text"
        )
        
        # Normalize the text features
        text_features = text_features.text_embeds_proj[:, 0, :]
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def text_search(self, query_text: str, top_k: int):
        """
        Perform image search based on the query text.
        """
        # Extract text features
        text_features = self.encode_text(query_text)
        
        # Perform search in the index
        scores, idx_image = self.index.search(text_features, 32 * top_k)
        
        # Process image indices
        idx_image = np.floor(idx_image.flatten() / 32).astype(np.int64)
        idx_image = top_k_unique_in_order(idx_image, top_k)
        
        # Retrieve image paths from index
        image_paths = [self.id2image_fps.get(idx) for idx in idx_image]
        
        return result_format(image_paths, scores.flatten())
