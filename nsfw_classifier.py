import os
import cv2
import pydload
import logging
import numpy as np
import onnxruntime


class Classifier:
    """
    Class for loading model and running predictions.
    For example on how to use take a look the if __name__ == '__main__' part.
    """

    nsfw_model = None

    def __init__(self):
        """
        model = Classifier()
        """
        url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, ".NudeNet/")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        model_path = os.path.join(model_folder, os.path.basename(url))
        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = onnxruntime.InferenceSession(model_path)

    def pred_nsfw(
        self,
        images=[],
        batch_size=4,
        categories=["unsafe", "safe"],
    ):
        """
        inputs:
            image_paths: list of image
            batch_size: batch_size for running predictions
            image_size: size to which the image needs to be resized
            categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if not isinstance(images, list):
            images = [images]
        for i in range(len(images)):
            h, w, c = images[i].shape
            if h != 256 or w != 256:
                images[i] = cv2.resize(images[i], (256, 256))

        model_preds = []
        while len(images):
            _model_preds = self.nsfw_model.run(
                [self.nsfw_model.get_outputs()[0].name],
                {self.nsfw_model.get_inputs()[0].name: images[:batch_size]},
            )[0]
            for pred in _model_preds:
                model_preds.append(pred[0])
            images = images[batch_size:]
        return model_preds
