from PIL import Image
from pathlib import Path

import torch.utils.data as data


class IR_ImageFolderDataset(data.Dataset):
    def __init__(self, root_gt, root_deg=None, name="", degradation=None, transform=None):
        self.name = name
        self.degradation = degradation
        self.transform = transform
        self.root_gt = Path(root_gt)
        self.root_deg = None if root_deg is None else Path(root_deg)
        
        self.image_paths_gt = self._get_paths(self.root_gt, is_gt=True)
        self.image_paths_deg = None
        
        if self.root_deg:
            self.image_paths_deg = self._get_paths(self.root_deg, is_gt=False)
            
            if len(self.image_paths_gt) != len(self.image_paths_deg):
                raise ValueError(
                    f"Mismatch in image counts after split: GT ({len(self.image_paths_gt)}) "
                    f"vs Degradation ({len(self.image_paths_deg)})"
                )

    def _get_paths(self, root_path: Path, is_gt: bool):
        
        if not root_path.exists():
            raise ValueError(f"Root directory does not exist: {root_path}")
            
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP']
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(root_path.glob('**/' + ext)))
            
        image_paths = sorted(list(set(image_paths)))
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {root_path}")
                
        return image_paths

    def __getitem__(self, index):
        path_gt = self.image_paths_gt[index]
        
        try:
            img = Image.open(path_gt).convert("RGB")
        except Exception as e:
            print(f"Error loading GT image {path_gt}: {e}")
            img = Image.new('RGB', (512, 512))

        if self.transform is not None:
            img = self.transform(img)
            
        degraded_img = img # default
        _img = None        
        
        if self.image_paths_deg is not None:
            path_deg = self.image_paths_deg[index]
            
            try:
                degraded_img = Image.open(path_deg).convert("RGB")
            except Exception as e:
                print(f"Error loading Degraded image {path_deg}: {e}")
                degraded_img = Image.new('RGB', (512, 512)) 
                
            if self.transform is not None:
                degraded_img = self.transform(degraded_img)
                            
        elif self.degradation is not None: 
            degraded_img, _img = self.degradation(img)

        if _img is not None:
            img = _img

        return degraded_img, img

    def __len__(self):
        return len(self.image_paths_gt)
