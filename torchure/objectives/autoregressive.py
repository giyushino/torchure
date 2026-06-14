"""
normally for AR models we don't need to have
this objective wrapper, but if we want to expand
to different model archs / learning objectives, 
it's good to have this abstraction
"""
from torchure.loss.cross_entropy import cross_entropy_loss


class ARObjective:
    def __init__(self):
        pass




