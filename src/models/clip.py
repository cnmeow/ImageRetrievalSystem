import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from src.utils import load_bin_file, load_id2image_file, result_format


class CLIPImageEncoder:
    def __init__(self, pretrained: str = 'laion400m_e32'):
        self.model_name = 'ViT-L-14'
        self.pretrained = pretrained
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained
            )
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name} with weights {pretrained}: {e}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image_path_list: List[str], batch_size: int):
        id_to_image_dict = {id: path for id, path in enumerate(image_path_list)}  # Mapping ID to image path
        encoding_list = []

        # Process and encode images in batches
        for start_index in tqdm(range(0, len(image_path_list), batch_size), desc="Encoding images"):
            batch_paths = image_path_list[start_index:start_index + batch_size]
            image_list = []

            for image_path in batch_paths:
                try:
                    with Image.open(image_path) as img:
                        image = self.preprocess(img).unsqueeze(0)  # Add batch dimension
                        image_list.append(image)
                except Exception as e:
                    print(f"Failed to process image {image_path}: {e}")

            if not image_list:
                print(f"No valid images to encode in batch starting at index {start_index}.")
                continue

            image_tensor = torch.cat(image_list, dim=0).to(self.device)

            # Perform encoding
            with torch.inference_mode():
                image_features = self.model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                encoding_list.extend(image_features.cpu().numpy().astype(np.float32))

        if not encoding_list:
            raise ValueError("No valid images were successfully encoded.")

        return id_to_image_dict, encoding_list

class CLIPSearchEngine:
    def __init__(self, bin_file: str, metadata: str):
        # Load the index and metadata
        self.index = load_bin_file(bin_file)
        self.id2image_fps = load_id2image_file(metadata)
        
        # Set the device for computation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the model and tokenizer
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='laion400m_e32', device=self.device
        )
        self.model.eval()

    def encode_text(self, query_text: str) -> np.ndarray:
        """
        Extracts features from the input text.
        """
        text_tokens = self.tokenizer([query_text]).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy().astype(np.float32)

    def text_search(self, query_text: str, top_k: int) -> dict:
        """
        Searches for the most relevant images based on the query text.
        """
        # Extract text features
        text_features = self.encode_text(query_text)
        
        # Perform the search using Faiss
        scores, idx_image = self.index.search(text_features, top_k)
        
        # Get the image paths corresponding to the indices
        idx_image = idx_image.flatten()
        image_paths = [self.id2image_fps.get(idx) for idx in idx_image]
        
        # Return the search results in the desired format
        return result_format(image_paths, scores.flatten())
