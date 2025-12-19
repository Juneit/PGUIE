import os
import pickle
import numpy as np
import torch
import faiss
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FeatureStorage:
    
    def __init__(self, 
                 save_dir: str = "./feature_database",
                 feature_levels: List[str] = ["down_0", "down_1", "down_2", "down_3"],
                 normalize_features: bool = True,
                 lazy_load: bool = False):
        self.save_dir = Path(save_dir)
        self.feature_levels = feature_levels
        self.normalize_features = normalize_features
        self.lazy_load = lazy_load
        
        self.indices = {}
        self.feature_shapes = {}
        self.image_names = {}
        self.feature_metadata = {}
        
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if not lazy_load:
            self._initialize_indices()
        else:
            # 延迟初始化：只设置基本结构，不加载索引
            for level in self.feature_levels:
                self.indices[level] = None
                self.feature_shapes[level] = None
                self.image_names[level] = []
                self.feature_metadata[level] = {
                    'feature_shape': None,
                    'image_names': [],
                    'total_features': 0
                }
            logger.info(f"特征库延迟加载模式已启用: {save_dir}")
    
    def _initialize_indices(self):
        for level in self.feature_levels:
            level_dir = self.save_dir / level
            level_dir.mkdir(exist_ok=True)

            index_path = level_dir / "faiss_index.bin"
            metadata_path = level_dir / "metadata.json"
            
            if index_path.exists() and metadata_path.exists():
                self.indices[level] = faiss.read_index(str(index_path))
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_shapes[level] = metadata['feature_shape']
                    self.image_names[level] = metadata['image_names']
                    self.feature_metadata[level] = metadata
                logger.info(f"加载现有索引: {level}, 特征数量: {self.indices[level].ntotal}")
            else:
                self.indices[level] = None  # 延迟初始化
                self.feature_shapes[level] = None
                self.image_names[level] = []
                self.feature_metadata[level] = {
                    'feature_shape': None,
                    'image_names': [],
                    'total_features': 0
                }
                logger.info(f"创建新索引: {level}")
    
    def _prepare_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        processed_features = {}
        
        for level, tensor in features.items():
            if level not in self.feature_levels:
                continue
                
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu()

            if self.feature_shapes[level] is None:
                self.feature_shapes[level] = tensor.shape

            batch_size = tensor.shape[0]
            tensor_2d = tensor.view(batch_size, -1)

            feature_array = tensor_2d.numpy().astype(np.float32)

            if self.normalize_features:
                faiss.normalize_L2(feature_array)
            
            processed_features[level] = feature_array
            
        return processed_features
    
    def _initialize_index_for_level(self, level: str, feature_dim: int):
        if self.indices[level] is None:

            self.indices[level] = faiss.IndexFlatIP(feature_dim)
            logger.info(f"初始化Flat索引 {level}: 维度={feature_dim}")
            

    def _load_index_for_level(self, level: str):
        if self.indices[level] is not None:
            return
            
        level_dir = self.save_dir / level
        index_path = level_dir / "faiss_index.bin"
        metadata_path = level_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            self.indices[level] = faiss.read_index(str(index_path))
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_shapes[level] = metadata['feature_shape']
                self.image_names[level] = metadata['image_names']
                self.feature_metadata[level] = metadata
            logger.info(f"延迟加载索引: {level}, 特征数量: {self.indices[level].ntotal}")
        else:
            logger.warning(f"索引文件不存在: {level}")
            self.indices[level] = None
            self.feature_shapes[level] = None
            self.image_names[level] = []
            self.feature_metadata[level] = {
                'feature_shape': None,
                'image_names': [],
                'total_features': 0
            }
    
    def add_features(self, 
                    image_names: List[str], 
                    features: Dict[str, torch.Tensor],
                    batch_size: int = 32) -> None:
        processed_features = self._prepare_features(features)
        
        for level, feature_array in processed_features.items():
            if level not in self.feature_levels:
                continue
                
            feature_dim = feature_array.shape[1]
            
            self._initialize_index_for_level(level, feature_dim)
            
            if self.indices[level].ntotal == 0:
                logger.info(f"Flat索引 {level} 无需训练，直接添加特征")
            
            self.indices[level].add(feature_array)
            
            self.image_names[level].extend(image_names)

            self.feature_metadata[level]['total_features'] = self.indices[level].ntotal
            self.feature_metadata[level]['feature_shape'] = self.feature_shapes[level]
            self.feature_metadata[level]['image_names'] = self.image_names[level]
            
            if self.indices[level].ntotal % 100 == 0:
                logger.info(f"添加特征到 {level}: {len(image_names)} 个图像, 总特征数: {self.indices[level].ntotal}")
    
    def search_features(self, 
                       query_features: Dict[str, torch.Tensor],
                       k: int = 5,
                       level_weights: Optional[Dict[str, float]] = None) -> Dict[str, List[Tuple[str, float]]]:

        if self.lazy_load:
            for level in query_features.keys():
                if level in self.feature_levels and self.indices[level] is None:
                    self._load_index_for_level(level)
        
        processed_query = self._prepare_features(query_features)
        results = {}
        
        for level, query_array in processed_query.items():
            if level not in self.feature_levels or self.indices[level] is None:
                continue

            similarities, indices = self.indices[level].search(query_array, k)
            
            level_results = []
            for i in range(len(indices)):
                image_results = []
                for j in range(len(indices[i])):
                    if indices[i][j] < len(self.image_names[level]):
                        image_name = self.image_names[level][indices[i][j]]
                        similarity = float(similarities[i][j])
                        image_results.append((image_name, similarity))
                level_results.append(image_results)
            
            results[level] = level_results
        
        return results
    
    def get_feature_by_name(self, image_name: str, level: str) -> Optional[torch.Tensor]:
        if level not in self.feature_levels or level not in self.image_names:
            return None
        
        if self.lazy_load and self.indices[level] is None:
            self._load_index_for_level(level)
            
        try:
            idx = self.image_names[level].index(image_name)
            feature_vector = self.indices[level].reconstruct(idx)
            
            original_shape = self.feature_shapes[level]
            if original_shape is not None:
                single_feature_shape = original_shape[1:]
                feature_tensor = torch.from_numpy(feature_vector).reshape(single_feature_shape)
                return feature_tensor
        except (ValueError, IndexError):
            logger.warning(f"未找到图像 {image_name} 在level {level}的特征")
            
        return None
    
    def save_database(self) -> None:
        for level in self.feature_levels:
            if self.indices[level] is not None:
                level_dir = self.save_dir / level
                
                index_path = level_dir / "faiss_index.bin"
                faiss.write_index(self.indices[level], str(index_path))
                
                metadata_path = level_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(self.feature_metadata[level], f, indent=2)
                
                logger.info(f"保存数据库 {level}: {self.indices[level].ntotal} 个特征")
    
    def load_database(self) -> None:
        self._initialize_indices()
    
    def get_database_info(self) -> Dict[str, Any]:
        info = {}
        for level in self.feature_levels:
            if self.indices[level] is not None:
                info[level] = {
                    'total_features': self.indices[level].ntotal,
                    'feature_shape': self.feature_shapes[level],
                    'image_count': len(self.image_names[level]),
                    'feature_dim': self.indices[level].d if hasattr(self.indices[level], 'd') else None
                }
        return info
    
    def clear_database(self) -> None:
        for level in self.feature_levels:
            self.indices[level] = None
            self.feature_shapes[level] = None
            self.image_names[level] = []
            self.feature_metadata[level] = {
                'feature_shape': None,
                'image_names': [],
                'total_features': 0
            }
        
        import shutil
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("数据库已清空")


class FeatureExtractor:

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def extract_features(self, 
                        images: torch.Tensor,
                        return_posterior: bool = False) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            images = images.to(self.device)
            
            if hasattr(self.model, 'encode'):
                if return_posterior:
                    posterior, features = self.model.encode(images, return_features=True)
                    return features
                else:
                    posterior = self.model.encode(images)
                    features = self.model.get_intermediate_features()
                    return features
            else:
                raise ValueError("模型必须具有encode方法")
    
    def extract_features_batch(self, 
                             image_paths: List[str],
                             transform_fn,
                             batch_size: int = 8) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        all_features = {level: [] for level in ["down_0", "down_1", "down_2", "down_3"]}
        image_names = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_names = []
            
            for path in batch_paths:
                try:
                    image = transform_fn(path)
                    batch_images.append(image)
                    batch_names.append(Path(path).name)
                except Exception as e:
                    logger.warning(f"处理图像 {path} 时出错: {e}")
                    continue
            
            if not batch_images:
                continue
                
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            features = self.extract_features(batch_tensor)
            
            for level, tensor in features.items():
                if level in all_features:
                    all_features[level].append(tensor.cpu())
            
            image_names.extend(batch_names)
        
        final_features = {}
        for level, tensors in all_features.items():
            if tensors:
                final_features[level] = torch.cat(tensors, dim=0)
        
        return image_names, final_features


def create_feature_database(model: torch.nn.Module,
                          image_paths: List[str],
                          transform_fn,
                          save_dir: str = "./feature_database",
                          batch_size: int = 8,
                          device: str = "cuda") -> FeatureStorage:
    extractor = FeatureExtractor(model, device)
    
    storage = FeatureStorage(save_dir=save_dir)
    
    image_names, features = extractor.extract_features_batch(
        image_paths, transform_fn, batch_size
    )
    
    storage.add_features(image_names, features)
    
    storage.save_database()
    
    return storage 