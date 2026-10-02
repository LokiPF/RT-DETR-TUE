"""Image-level corruption detection inside a frozen RT-DETRv2-R18.

Six published baselines and the two-axis score of the early backbone channels: corruptions either flatten the
channels (fog, contrast, blur) or shift their level (noise), judged against clean training scenes like the image.
"""

__version__ = "2.0.0"
