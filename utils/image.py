import torchvision.transforms as transforms

__all__ = ['make_square']


def make_square(x):
    width, height = x.size
    big_axis = max(width, height)
    padder = transforms.Pad((int((big_axis - width) / 2.0), int((big_axis - height) / 2.0)))
    return padder(x)
