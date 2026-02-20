import torch
import torch.nn as nn
import torch.nn.functional as F

class RATBase(nn.Module):

    def __init__(self, extractor, feature_processor, in_dim=1024, head_nb=4):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        param head_nb: Number of heads for multi-head attention.
        """

        super().__init__()

        # Initialize the feature extractor and processor
        self.extractor = extractor
        self.feature_processor = feature_processor

        # Initialize the multi-head attention layer (GPT-J style)
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=head_nb)
        self.preattn_ln = nn.LayerNorm(in_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4), nn.ReLU(), nn.Linear(in_dim * 4, in_dim)
        )
        self.final_ln = nn.LayerNorm(in_dim)

        # Allow variable input dimension, mainly for base (768 features), large (1024 features) and extra-large (1920 features) models.
        self.layer1_in_dim = in_dim
        self.layer1_out_dim = in_dim // 2
        self.layer2_out_dim = self.layer1_out_dim // 2

        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(self.layer1_in_dim),  # Normalize
        #     nn.Linear(self.layer1_in_dim, self.layer1_out_dim),
        #     nn.GELU(),
        #     # nn.Dropout(0.1),
        #     nn.LayerNorm(self.layer1_out_dim),
        #     nn.Linear(self.layer1_out_dim, self.layer2_out_dim),
        #     nn.GELU(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(self.layer2_out_dim, 2),  # output 2 classes
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.layer1_in_dim, self.layer1_out_dim),
            nn.BatchNorm1d(self.layer1_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer1_out_dim, self.layer2_out_dim),
            nn.BatchNorm1d(self.layer2_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_out_dim, 2),  # output 2 classes
        )

    def forward(self, input_ref, input_test):
        raise NotImplementedError("Forward pass not implemented in the base class.")


class RAT(RATBase):
    """
    Reference-augmented training model
    """

    def __init__(self, extractor, feature_processor, in_dim=1024, head_nb=4):
        super().__init__(extractor, feature_processor, in_dim, head_nb)

    def forward(self, input_ref, input_test):
        """
        Forward pass of the model.

        Extract features from the test and reference data, compute cross-attention
        between the embeddings, enhance the test embedding with the cross-attention using residual connection,
        process the informed embedding and pass it to the classifier.

        param input_ref: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_test: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_ref = self.extractor.extract_features(input_ref)
        emb_test = self.extractor.extract_features(input_test)

        # Reshape so that we compute cross-attention per transformer layer
        layers, batches, time_ref, feature = emb_ref.shape
        time_test = emb_test.shape[2]

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        # (time_ref, layers * batches, feature)
        stacked_ref = emb_ref.view(layers * batches, time_ref, feature).transpose(0, 1)
        # (time_test, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(0, 1)

        # Norm after SSL extractor
        norm_ref = self.preattn_ln(stacked_ref)
        norm_test = self.preattn_ln(stacked_test)

        # Compute cross-attention
        attn_map, _ = self.attn(norm_test, norm_ref, norm_ref)

        # MLP
        mlp_out = self.attn_mlp(norm_test)

        # Combine
        combined = attn_map + mlp_out
        residual = stacked_test + combined
        final_out = self.final_ln(residual).transpose(0, 1).view(layers, batches, time_test, feature)
        
        # Process the features
        self.emb = self.feature_processor(final_out)

        out = self.classifier(self.emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class RAT_baseline(nn.Module):
    """
    RAT baseline, just extractor -> mean pooling -> classifier
    """

    def __init__(self, extractor, feature_processor, in_dim=1024, head_nb=4):
        super().__init__()

        # Initialize the feature extractor and processor
        self.extractor = extractor
        self.feature_processor = feature_processor

        self.layer1_in_dim = in_dim
        self.layer1_out_dim = in_dim // 2
        self.layer2_out_dim = self.layer1_out_dim // 2

        self.classifier = nn.Sequential(
            nn.Linear(self.layer1_in_dim, self.layer1_out_dim),
            nn.BatchNorm1d(self.layer1_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer1_out_dim, self.layer2_out_dim),
            nn.BatchNorm1d(self.layer2_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_out_dim, 2),  # output 2 classes
        )

    def forward(self, input_ref, input_test):
        emb_test = self.extractor.extract_features(input_test)

        # Process the features
        self.emb = self.feature_processor(emb_test)

        out = self.classifier(self.emb)
        prob = F.softmax(out, dim=1)

        return out, prob
    

class RAT_selfattn(RATBase):
    """
    RAT with self-attention instead of cross-attention
    """

    def __init__(self, extractor, feature_processor, in_dim=1024, head_nb=4):
        super().__init__(extractor, feature_processor, in_dim, head_nb)

    def forward(self, input_ref, input_test):
        emb_test = self.extractor.extract_features(input_test)

        # Reshape so that we compute self-attention per transformer layer
        layers, batches, time_test, feature = emb_test.shape

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        # (time_test, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(0, 1)

        # Norm after SSL extractor
        norm_test = self.preattn_ln(stacked_test)

        # Compute self-attention
        attn_map, _ = self.attn(norm_test, norm_test, norm_test)

        # MLP
        mlp_out = self.attn_mlp(norm_test)

        # Combine
        combined = attn_map + mlp_out
        residual = stacked_test + combined
        final_out = self.final_ln(residual).transpose(0, 1).view(layers, batches, time_test, feature)
        
        # Process the features
        self.emb = self.feature_processor(final_out)

        out = self.classifier(self.emb)
        prob = F.softmax(out, dim=1)

        return out, prob
    

class RAT_zeroref(RATBase):
    """
    RAT with zeroed reference input
    """

    def __init__(self, extractor, feature_processor, in_dim=1024, head_nb=4):
        super().__init__(extractor, feature_processor, in_dim, head_nb)

    def forward(self, input_ref, input_test):
        emb_test = self.extractor.extract_features(input_test)

        # Reshape so that we compute cross-attention per transformer layer
        layers, batches, time_test, feature = emb_test.shape

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        # (time_test, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(0, 1)

        # Norm after SSL extractor
        norm_test = self.preattn_ln(stacked_test)

        # Create zeroed reference
        zero_ref = torch.zeros_like(norm_test)

        # Compute cross-attention
        attn_map, _ = self.attn(norm_test, zero_ref, zero_ref)

        # MLP
        mlp_out = self.attn_mlp(norm_test)

        # Combine
        combined = attn_map + mlp_out
        residual = stacked_test + combined
        final_out = self.final_ln(residual).transpose(0, 1).view(layers, batches, time_test, feature)
        
        # Process the features
        self.emb = self.feature_processor(final_out)

        out = self.classifier(self.emb)
        prob = F.softmax(out, dim=1)

        return out, prob
