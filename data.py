import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from glob import glob


class VQData(Dataset):

    def __init__(self, root):
        
        super().__init__()
        self.root = root
        self.imgs = glob(f'{self.root}/*/*.jpg') 
        print(len(self.imgs))
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(0.05),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        
        return len(self.imgs)

    def __getitem__(self, idx):
    
        try:
            img = Image.open(self.imgs[idx])
            img = self.transform(img)
            return img
        except:
            return self.__getitem__(idx - 1)




    