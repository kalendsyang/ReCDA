import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """Implementation of the SimCLR loss function."""

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the SimCLR module.

        Parameters:
        - temperature: A scaling factor for the similarity scores.
        """
        super(SimCLR, self).__init__()
        self.temperature = temperature

    def compute_similarity(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity matrix between two sets of embeddings.

        Parameters:
        - z_i: Embeddings from the first batch.
        - z_j: Embeddings from the second batch.

        Returns:
        - A tensor containing the cosine similarity scores.
        """
        z = torch.cat([z_i, z_j], dim=0)
        return F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute the SimCLR loss.

        Parameters:
        - z_i: Embeddings from the augmented view of the first image.
        - z_j: Embeddings from the augmented view of the second image.

        Returns:
        - The calculated SimCLR loss.
        """
        batch_size = z_i.size(0)
        similarity = self.compute_similarity(z_i, z_j)

        # Ensure no NaN values in similarity
        assert not torch.any(torch.isnan(similarity)), "NaN values found in similarity matrix."

        # Extract positive pairs
        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Calculate the numerator and denominator for the loss
        numerator = torch.exp(positives / self.temperature)
        mask = (~torch.eye(batch_size * 2, dtype=torch.bool)).float().to(similarity.device)
        denominator = mask * torch.exp(similarity / self.temperature)

        # Calculate the loss
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss