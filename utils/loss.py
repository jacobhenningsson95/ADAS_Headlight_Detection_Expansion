from utils.discriminative import DiscriminativeLoss
import torch
# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class calc_loss(torch.nn.Module):
    def __init__(self, config):
        super(calc_loss, self).__init__()
        # Initialize discriminative loss, binary cross-entropy loss, and mean squared error loss
        self.discriminativeLoss = DiscriminativeLoss(0.5, 1.5, norm=2).to(device)
        self.crossEntropyLoss = torch.nn.BCEWithLogitsLoss().to(device)
        self.meanSquareErrorLoss = torch.nn.MSELoss().to(device)

        # Retrieve configuration parameters
        self.nstack = config['nstack']
        self.max_num_light = config['max_num_light']

    def forward(self, preds, instance_maps=None, instance_count=None, semantic_seg=None):
        """
        Forward pass to compute the combined loss.

        Args:
            preds (list): List containing predicted instance and semantic maps.
            instance_maps (torch.Tensor): Ground truth instance maps.
            instance_count (torch.Tensor): Number of instances in each image.
            semantic_seg (torch.Tensor): Ground truth semantic segmentation maps.

        Returns:
            list: List containing discriminative loss and cross-entropy loss.

        """
        # Separate predicted instance and semantic maps from the input
        instance_preds = preds[0]
        semantic_preds = preds[1]

        # Compute discriminative loss for each stack
        discriminative_loss = []
        for i in range(self.nstack):
            disc_loss = self.discriminativeLoss(instance_preds[:, i], instance_maps, instance_count, instance_maps.size(1)).to(device)
            discriminative_loss.append(disc_loss)
        discriminative_loss = torch.stack(discriminative_loss, dim=0)

        # Compute cross-entropy loss for each stack
        cross_entropy_loss = []
        _, ce_semantic_seg = torch.max(semantic_seg, 1)  # Convert semantic_seg to one-hot encoding
        for i in range(self.nstack):
            cross_entropy_loss.append(self.crossEntropyLoss(semantic_preds[:, i], semantic_seg.to(torch.float16)) )
        cross_entropy_loss = torch.stack(cross_entropy_loss, dim=0)

        return [discriminative_loss, cross_entropy_loss]