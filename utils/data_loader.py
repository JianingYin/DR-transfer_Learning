from torchvision import transforms
from torchvision.transforms import Resize
from utils.utils import image_preprocess

transform = transforms.Compose([
    Resize((224, 224)),  # 调整图像尺寸为224x224
    transforms.Lambda(lambda x: image_preprocess(x)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

apots_transform = transforms.Compose([
    Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])