from GCL.augmentors.Augmentor import Graph, Augmentor


class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g
