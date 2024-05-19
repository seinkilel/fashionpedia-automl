import numpy as np
import pandas as pd
from PIL import Image
from azureml.fsspec import AzureMachineLearningFileSystem
from matplotlib import pyplot as plt
from random import randint
from azureml.dataprep.rslex import StreamInfo
from ultralytics.utils.plotting import Annotator, colors

class ImageReader:

    def read_image(self, path):
        pass


class AzureImageReader(ImageReader):

    def __init__(self, filesystem: AzureMachineLearningFileSystem, output_format: str):
        self.fs = filesystem
        self.format = output_format
        if self.format != "pil" and self.format != "numpy":
            raise ValueError("Format must be either 'pil' or 'numpy'")

    def read_image(self, path):
        f = None
        if isinstance(path, str):
            f = self.fs.open(path)
        if isinstance(path, StreamInfo):
            f = path.open()
        if f is None:
            raise ValueError("Path must be a valid path to a file, or a StreamInfo")
        image = Image.open(f)
        if self.format == "pil":
            f.close()
            return image
        img = np.array(image)
        f.close()
        return img


def plot_random_images(df: pd.DataFrame, image_reader: ImageReader, n_cols: int = 3, n_rows: int = 5,
                       fig_size=(18, 20)):
    figure = plt.figure(figsize=fig_size)
    for i in range(1, n_cols * n_rows + 1):
        n = randint(0, len(df) - 1)
        image_path = df.iloc[n]['image_url']
        image = image_reader.read_image(image_path)
        label = df.iloc[n]['label']
        figure.add_subplot(n_rows, n_cols, i)
        if type(label) == str:
            plt.title(f'Label: {label}', fontsize=14, fontweight='bold')
        plt.axis("off")
        if image.ndim == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.show()


def plot_image_with_boxes(df: pd.DataFrame, image_reader: ImageReader, index: int):
    row = df.iloc[index]
    image_path = row['image_url']
    image = image_reader.read_image(image_path)
    annotations = row['label']
    width = row["image_details"]['width']
    height = row["image_details"]['height']
    labels = []
    bounded_boxes = []
    targets = []
    i = 0
    for label in annotations:
        labels.append(label['label'])
        bounded_boxes.append(
            [label['topX'] * width, label['topY'] * height, label['bottomX'] * width, label['bottomY'] * height])
        targets.append(i)
        i = i + 1
    annotator = Annotator(image)
    for box, label, target in zip(bounded_boxes, labels, targets):
        annotator.box_label(box, label, color=colors(target))
    plt.axis("off")
    plt.figure(figsize=(18, 20))
    plt.imshow(annotator.im)
    plt.show()
