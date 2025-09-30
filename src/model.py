import torch
import torch.nn as nn

class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) implementation."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout_rate: float = 0.0):
        """
        Initialize the MLP.

        Parameters:
        - input_size: The size of the input features.
        - hidden_size: The size of the hidden layers.
        - num_layers: The total number of layers (including the output layer).
        - dropout_rate: The dropout rate to apply after each hidden layer.
        """
        super(MLP, self).__init__()

        layers = []
        current_size = input_size

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, hidden_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

class ReCDA(nn.Module):
    """Model of ReCDA that combines an encoder and a head for representation learning."""

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        perturbation_rate: float = 0.6,
        encoder_depth: int = 2,
        head_depth: int = 2,
    ):
        """
        Initialize the ReCDA model.

        Parameters:
        - input_size: The size of the input features.
        - embedding_size: The size of the embedding output.
        - perturbation_rate: The fraction of input features to perturb.
        - encoder_depth: The depth of the encoder MLP.
        - head_depth: The depth of the head MLP.
        """
        super(ReCDA, self).__init__()

        self.encoder = MLP(input_size, embedding_size, encoder_depth)
        self.head = MLP(embedding_size, embedding_size, head_depth)

        self.perturbation_length = int(perturbation_rate * input_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the model."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, anchor_sample: torch.Tensor, random_sample: torch.Tensor) -> tuple:
        """
        Forward pass through the ReCDA model.

        Parameters:
        - anchor_sample: The anchor sample tensor.
        - random_sample: The tensor for generating perturbations.

        Returns:
        - A tuple of anchor and positive embeddings.
        """
        positive_sample = self._generate_perturbed_sample(anchor_sample, random_sample)

        anchor_embedding = self.head(self.encoder(anchor_sample))
        positive_embedding = self.head(self.encoder(positive_sample))

        return anchor_embedding, positive_embedding

    def _generate_perturbed_sample(self, anchor_sample: torch.Tensor, random_sample: torch.Tensor) -> torch.Tensor:
        """
        Generate a perturbed sample by replacing a fraction of the anchor sample.

        Parameters:
        - anchor_sample: The original anchor sample.
        - random_sample: The sample used for perturbations.

        Returns:
        - The perturbed sample.
        """
        batch_size, sequence_length = anchor_sample.size()
        perturbation_mask = torch.zeros_like(anchor_sample, dtype=torch.bool)

        for i in range(batch_size):
            idx = torch.randperm(sequence_length)[:self.perturbation_length]
            perturbation_mask[i, idx] = True

        return torch.where(perturbation_mask, random_sample, anchor_sample)

    def get_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the encoder.

        Parameters:
        - input: The input tensor to encode.

        Returns:
        - The resulting embeddings from the encoder.
        """
        return self.encoder(input)