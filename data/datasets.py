import torch
import torch.utils.data as data


class SegmentationImageDataset(data.Dataset):
    def __init__(self, image_paths, mask_paths, image_reader, mask_reader):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_reader = image_reader
        self.mask_reader = mask_reader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_reader(self.image_paths[index])
        mask = self.mask_reader(self.mask_paths[index])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


class SegmentationPatchDataset(data.Dataset):
    def __init__(self, patches, masks):
        self.patches = patches
        self.masks = masks

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        image = self.patches[index]
        mask = self.masks[index]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


class UnlabeledImageDataset(data.Dataset):
    def __init__(self, image_paths, image_reader):
        self.image_paths = image_paths
        self.image_reader = image_reader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_reader(self.image_paths[index])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image


class UnlabeledPatchDataset(data.Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        image = self.patches[index]
        return torch.from_numpy(image.transpose(2, 0, 1)).float()
