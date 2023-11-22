import torchvision.transforms as transforms
from channel_aug import ChannelT, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,ChannelRandomErasing
#normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def test_transforms(H,W,normalize):
    transform_test = transforms.Compose( [
        transforms.ToPILImage(),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        normalize])
    return transform_test

def train_transforms_color1(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            #transforms.ColorJitter(hue=0.5),
            transforms.RandomGrayscale(p = 0.5),
            transforms.Pad(10),
            transforms.RandomCrop((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ])
    return transform_train

def train_transforms_color2(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray=2)
            ])
    return transform_train




def train_transforms_thermal1(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)
            ])
    return transform_train


def train_transforms_thermal2(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelT(probability =0.5)           
            ])
    return transform_train




       