import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from torchvision import transforms
from transformers import XLMRobertaTokenizer
from torchvision.transforms.functional import InterpolationMode
from repos.unilm.beit3.modeling_finetune import beit3_base_patch16_224_retrieval
from src.utils import load_bin_file, load_id2image_file, result_format

WEIGHT_DIR = './data/weights'

class BEITImageEncoder:
    def __init__(self, model_path=f'{WEIGHT_DIR}/beit3_base_itc_patch16_224.pth', image_size: int = 224):
        self.model_path = model_path
        self.image_size = image_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = beit3_base_patch16_224_retrieval(pretrained=True)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def process_image(self, image_path: str):
        """Transform a single image."""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        try:
            with Image.open(image_path).convert('RGB') as img:
                return transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")
            return None

    def encode_image(self, image_path_list: List[str], batch_size: int):
        id_to_image_dict = {}
        encoding_list = []

        # Dictionary for mapping IDs to image paths
        for idx, image_path in enumerate(image_path_list):
            id_to_image_dict[idx] = image_path

        # Process and encode images batch by batch
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(image_path_list), batch_size), desc="Processing and encoding images"):
                batch_paths = image_path_list[start_idx:start_idx + batch_size]
                batch_tensors = []

                # Preprocess images in the batch
                for image_path in batch_paths:
                    try:
                        image_tensor = self.process_image(image_path)
                        if image_tensor is not None:
                            batch_tensors.append(image_tensor)
                    except Exception as e:
                        print(f"Failed to process image {image_path}: {e}")

                if not batch_tensors:
                    print(f"No valid images in batch {start_idx}-{start_idx + batch_size}. Skipping.")
                    continue

                # Stack tensors and move to device
                batch_images = torch.cat(batch_tensors, dim=0).to(self.device)

                # Encode images
                try:
                    image_features, _ = self.model(image=batch_images, only_infer=True)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    encoding_list.extend(image_features.cpu().numpy().astype(np.float32))
                except Exception as e:
                    print(f"Error during encoding batch {start_idx}-{start_idx + batch_size}: {e}")

        return id_to_image_dict, encoding_list


class BEITSearchEngine:
    def __init__(self, bin_file: str, metadata: str):
        """
        Initializes the BEITSearchEngine with the model, tokenizer, and index.
        
        Args:
            bin_file (str): Path to the BEiT binary file containing the index.
            metadata (str): Path to the metadata file mapping IDs to image paths.
        """
        # Load the index and metadata for image retrieval
        self.index = load_bin_file(bin_file)
        self.id2image_fps = load_id2image_file(metadata)
        
        # Set device for model inference
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize tokenizer and model
        self.tokenizer = XLMRobertaTokenizer(f'{WEIGHT_DIR}/beit3.spm')
        self.model = beit3_base_patch16_224_retrieval(pretrained=True)
        
        # Load model weights from checkpoint
        checkpoint = torch.load(f'{WEIGHT_DIR}/beit3_base_itc_patch16_224.pth')
        self.model.load_state_dict(checkpoint['model'])

    def encode_text(self, query_text: str):
        """
        Encode the query text into feature vectors using the BEiT model.
        
        Args:
            query_text (str): The text query to be encoded.
        
        Returns:
            np.ndarray: The text features as a normalized numpy array.
        """
        # Tokenize the input text
        text_tokens = self.tokenizer(
            text=query_text,
            return_tensors='pt'
        )["input_ids"]
        text_tokens = text_tokens.to(self.device)

        # Extract features using the BEiT model
        with torch.no_grad():
            _, text_features = self.model(
                text_description=text_tokens,
                only_infer=True
            )
            
            # Normalize the text features
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype(np.float32)
    
    def embedding_search(self, embedding, top_k: int) -> dict:
        # Perform the search using Faiss
        scores, idx_image = self.index.search(embedding, top_k)
        
        # Get the image paths corresponding to the indices
        idx_image = idx_image.flatten()
        image_paths = [self.id2image_fps.get(idx) for idx in idx_image]
        
        # Return the search results in the desired format
        return result_format(image_paths, scores.flatten())
    
    def text_search(self, query_text: str, top_k: int):
        """
        Perform a text-based search to find the top `k` most relevant images.
        
        Args:
            query_text (str): The text query to search for.
            image_path_subset (list): A list of image paths to search from.
            top_k (int): The number of top results to return.
        
        Returns:
            dict: A dictionary with image paths and corresponding similarity scores.
        """
        # Get the text features
        text_features = self.encode_text(query_text)
        
        # Search for the top `k` most similar images in the index
        scores, idx_image = self.index.search(text_features, top_k)
        
        # Process the retrieved image indices
        idx_image = idx_image.flatten()
        image_paths = list(map(self.id2image_fps.get, list(idx_image)))
        
        return result_format(image_paths, scores.flatten())