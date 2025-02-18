
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class BinaryMNISTClassifier:
    """
    A classifier that loads binary classifiers for digits 0-9 and predicts
    if an image is a MNIST digit (0-9) or not.
    """

    def __init__(self, model_dir="/users/zzhan513/data/zzhan513/visual_reasoning/BENCHMARKING-VISUAL-GENERATIVE-REASONING/mnist_equation/mnist_classifier/models"):
        """
        Initializes the BinaryMNISTClassifier by loading models for digits 0-9.

        Args:
            model_dir (str): Directory where model checkpoints are saved.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.digits = list(range(10))
        self.model_dir = model_dir
        self._load_models()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def _load_models(self):
        """
        Loads the binary classifiers for digits 0-9.
        """
        for digit in self.digits:
            model_path = os.path.join(self.model_dir, f"binary_classifier_digit_{digit}.pt")
            if os.path.exists(model_path):
                model = self._build_model()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models[digit] = model
            else:
                raise FileNotFoundError(
                    f"Model for digit {digit} not found at {model_path}"
                )

    def _build_model(self):
        """
        Builds the model architecture matching the saved checkpoints.
        """
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.network = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.network(x)

        return Model()

    def predict(self, image):
        """
        Predicts if the image is one of the digits 0-9 or not a MNIST digit.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            int or None: Predicted digit if image is a MNIST digit, None otherwise.
        """
        # Apply transformations
        if isinstance(image, Image.Image):
            img = self.transform(image)
        elif isinstance(image, torch.Tensor):
            img = image
        else:
            raise TypeError("Input must be a PIL image or a torch Tensor.")

        # add batch dimension
        img = img.to(self.device).unsqueeze(0)

        # collect outputs from all models
        outputs = {}
        with torch.no_grad():
            for digit, model in self.models.items():
                output = model(img).item()
                outputs[digit] = output

        # find the digit with the highest confidence
        max_digit = max(outputs, key=outputs.get)
        max_confidence = outputs[max_digit]
        
        return max_digit, max_confidence
