class CompositeTransform():
    def __init__(self, transforms, **_):
        """ Compose multiple data transforms sequentially. """
        self.transforms = transforms

    def __call__(self, data, **_):
        """ Call each transform separately. """
        for transform in self.transforms:
            data = transform(data)
        return data
