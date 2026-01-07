from torchvision import transforms


def eval_transform(img_size):
    """Transforms used for evaluation / validation: deterministic resize + to-tensor."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def train_transform_default(img_size):
    """Basic train transform (no strong augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def train_transform_retfound_linear(img_size):
    """Stronger augmentations used for RETFound linear probing mode."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
    ])
