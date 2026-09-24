"""The MIRROR network.

Four stages, in the order :meth:`Mirror.forward` runs them:

    1. visit-history encoder   past admissions to one vector each, then a
                               selection of past visits to one patient vector
    2. current-visit fusion    the discharge note and laboratory values modulate
                               the patient vector (FiLM)
    3. drug knowledge graph    one vector per drug class from its text and
                               structure, refined over four drug relations
    4. drug scorer             one score per drug class, all classes at once

The classes below follow that order.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelConfig
from dataset import CohortArtifacts, DrugGraph


class AttentionPooling(nn.Module):
    """Weighted average over a set of code embeddings.

    A single linear layer scores every code, the scores are normalised over the
    codes present, and the codes are averaged with those weights.

    Args:
        embedding_dim: width of one code embedding.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool one batch of code sets.

        Args:
            embeddings: ``(batch, codes, embedding_dim)``.
            mask: ``(batch, codes)``, true where a code is present.

        Returns:
            ``(batch, embedding_dim)``. An admission with no codes pools to
            zeros rather than to undefined weights.
        """
        scores = self.score(embeddings).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)


class SinusoidalPosition(nn.Module):
    """Fixed position signal added to the admission vectors.

    Args:
        width: width of the vectors the signal is added to.
        dropout: dropout applied after the addition.
        max_length: longest sequence the table covers.
    """

    def __init__(self, width: int, dropout: float = 0.1, max_length: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float) * (-math.log(10000.0) / width)
        )
        signal = torch.zeros(max_length, width)
        signal[:, 0::2] = torch.sin(position * frequency)
        signal[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("signal", signal.unsqueeze(0))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Add the position signal to ``(batch, length, width)``."""
        return self.dropout(sequence + self.signal[:, : sequence.size(1), :])


class AdmissionSequenceEncoder(nn.Module):
    """Causal transformer encoder over admission vectors.

    Args:
        width: width of one admission vector.
        layers: number of transformer blocks.
        heads: attention heads per block.
        dropout: dropout inside the blocks and after the position signal.
    """

    def __init__(self, width: int, layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.position = SinusoidalPosition(width, dropout=dropout)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Contextualise each admission with the admissions before it.

        Args:
            sequence: ``(batch, admissions, width)``.
            lengths: ``(batch,)`` number of real admissions per patient.

        Returns:
            ``(batch, admissions, width)``.
        """
        length = sequence.size(1)
        device = sequence.device
        sequence = self.position(sequence)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )
        padding_mask = torch.arange(length, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        return self.blocks(sequence, mask=causal_mask, src_key_padding_mask=padding_mask)


class DrugContextEncoder(nn.Module):
    """Cross-attention to drug embeddings followed by the sequence encoder.

    Args:
        width: width of one admission vector.
        drug_embedding_dim: width of one drug description embedding.
        layers: transformer blocks in the sequence encoder.
        heads: attention heads, both here and in the sequence encoder.
        dropout: dropout inside the sequence encoder.
    """

    def __init__(
        self,
        width: int,
        drug_embedding_dim: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            kdim=drug_embedding_dim,
            vdim=drug_embedding_dim,
            batch_first=True,
        )
        self.sequence = AdmissionSequenceEncoder(
            width=width, layers=layers, heads=heads, dropout=dropout
        )

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        drug_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Add drug context to each admission, then encode the sequence.

        Args:
            sequence: ``(batch, admissions, width)``.
            lengths: ``(batch,)`` number of real admissions per patient.
            drug_embeddings: ``(drugs, drug_embedding_dim)`` frozen embeddings of
                the drug descriptions.

        Returns:
            ``(batch, admissions, width)``.
        """
        batch_size = sequence.size(0)
        drugs = drug_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(query=sequence, key=drugs, value=drugs)
        return self.sequence(sequence + attended, lengths)


class VisitHistoryEncoder(nn.Module):
    """Encode a sequence of past admissions into contextual vectors.

    Args:
        diagnosis_embeddings: ``(diagnosis_codes, embedding_dim)`` frozen
            embeddings of the diagnosis descriptions.
        procedure_embeddings: ``(procedure_codes, embedding_dim)`` frozen
            embeddings of the procedure descriptions.
        drug_embeddings: ``(drugs, embedding_dim)`` frozen embeddings of the drug
            descriptions, used for the drug summary and the drug context.
        hidden_dim: width of the admission vectors the encoder produces.
        layers: transformer blocks over the admission sequence.
        heads: attention heads in every attention block.
        dropout: dropout applied after pooling, after projection and inside the
            transformer.
        max_visits: largest admission count the position table covers.
    """

    def __init__(
        self,
        diagnosis_embeddings: torch.Tensor,
        procedure_embeddings: torch.Tensor,
        drug_embeddings: torch.Tensor,
        hidden_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        max_visits: int = 30,
    ):
        super().__init__()
        embedding_dim = diagnosis_embeddings.size(1)
        self.diagnosis = nn.Embedding.from_pretrained(diagnosis_embeddings, freeze=True)
        self.procedure = nn.Embedding.from_pretrained(procedure_embeddings, freeze=True)
        self.register_buffer("drug_embeddings", drug_embeddings)

        self.pooling = AttentionPooling(embedding_dim)
        self.pooling_dropout = nn.Dropout(dropout)
        self.to_hidden = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.visit_position = nn.Embedding(max_visits, hidden_dim)
        self.drug_summary = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.encoder = DrugContextEncoder(
            width=hidden_dim,
            drug_embedding_dim=embedding_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )

    def encode_admission(
        self,
        diagnosis_codes: torch.Tensor,
        procedure_codes: torch.Tensor,
        diagnosis_mask: torch.Tensor,
        procedure_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool one admission's diagnosis and procedure codes into one vector.

        Args:
            diagnosis_codes: ``(batch, diagnosis_slots)`` code indices.
            procedure_codes: ``(batch, procedure_slots)`` code indices.
            diagnosis_mask: ``(batch, diagnosis_slots)`` true where a code is present.
            procedure_mask: ``(batch, procedure_slots)`` true where a code is present.

        Returns:
            ``(batch, hidden_dim)``.
        """
        embeddings = torch.cat(
            [self.diagnosis(diagnosis_codes), self.procedure(procedure_codes)], dim=1
        )
        mask = torch.cat([diagnosis_mask, procedure_mask], dim=1)
        pooled = self.pooling_dropout(self.pooling(embeddings, mask))
        return self.to_hidden(pooled)

    def forward(
        self,
        diagnosis_codes: list[torch.Tensor],
        procedure_codes: list[torch.Tensor],
        diagnosis_mask: list[torch.Tensor],
        procedure_mask: list[torch.Tensor],
        lengths: torch.Tensor,
        drugs_per_visit: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the whole history.

        Args:
            diagnosis_codes: one ``(batch, slots)`` tensor per admission position.
            procedure_codes: one ``(batch, slots)`` tensor per admission position.
            diagnosis_mask: validity masks matching ``diagnosis_codes``.
            procedure_mask: validity masks matching ``procedure_codes``.
            lengths: ``(batch,)`` number of real admissions per patient.
            drugs_per_visit: ``(batch, admissions, drugs)`` binary prescriptions
                of each past admission.

        Returns:
            ``(batch, admissions, hidden_dim)`` one contextual vector per
            admission position.
        """
        admissions = len(diagnosis_codes)
        device = diagnosis_codes[0].device
        encoded = torch.stack(
            [
                self.encode_admission(
                    diagnosis_codes[position],
                    procedure_codes[position],
                    diagnosis_mask[position],
                    procedure_mask[position],
                )
                for position in range(admissions)
            ],
            dim=1,
        )

        positions = torch.arange(admissions, device=device).clamp(
            max=self.visit_position.num_embeddings - 1
        )
        encoded = encoded + self.visit_position(positions).unsqueeze(0)

        if drugs_per_visit.size(-1) != self.drug_embeddings.size(0):
            raise ValueError(
                f"The batch reports {drugs_per_visit.size(-1)} drug classes but the drug "
                f"embeddings cover {self.drug_embeddings.size(0)}. The records and the "
                "embedding file come from different preprocessing runs."
            )
        summary = drugs_per_visit[:, :admissions, :] @ self.drug_embeddings
        encoded = encoded + self.drug_summary(summary)

        return self.encoder(encoded, lengths, self.drug_embeddings)


MASKED_SCORE = -1e9


class VisitSelector(nn.Module):
    """Attention over past admissions with a learned keep or drop decision.

    Args:
        hidden_dim: width of the admission vectors.
        dropout: dropout applied to the attention weights.
        attention_temperature: divides the scores before the softmax; a larger
            value spreads the weight more evenly over the kept admissions.
        selection_temperature: temperature of the keep or drop sampling.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.3,
        attention_temperature: float = 20.0,
        selection_temperature: float = 0.6,
    ):
        super().__init__()
        self.attention_temperature = attention_temperature
        self.selection_temperature = selection_temperature
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Collapse ``(batch, admissions, hidden_dim)`` into ``(batch, hidden_dim)``.

        Args:
            sequence: contextual vectors from the history encoder.
            lengths: ``(batch,)`` number of real admissions per patient.

        Returns:
            ``(batch, hidden_dim)``. A patient with a single past admission
            returns that admission's vector, normalised.
        """
        batch_size, admissions, width = sequence.shape
        device = sequence.device
        rows = torch.arange(batch_size, device=device)
        most_recent = (lengths - 1).clamp(min=0)
        current = sequence[rows, most_recent]

        if bool((lengths == 1).all()):
            return self.norm(current)

        query = self.project(current).unsqueeze(1)
        keys = self.project(sequence)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(width)
        padding = torch.arange(admissions, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(padding, MASKED_SCORE)

        keep = self._keep_decisions(scores)
        keep[rows, most_recent] = 1.0

        weights = F.softmax(
            scores.masked_fill(keep == 0, MASKED_SCORE) / self.attention_temperature, dim=-1
        )
        weights = self.dropout(weights)
        selected = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return self.norm(selected)

    def _keep_decisions(self, scores: torch.Tensor) -> torch.Tensor:
        """Decide which admissions to keep.

        Training samples a binary decision per admission so the choice stays
        trainable through the sampled value; evaluation takes the decision with
        the higher score, which makes the selection deterministic.

        Args:
            scores: ``(batch, admissions)`` relevance scores.

        Returns:
            ``(batch, admissions)`` with one for kept admissions and zero otherwise.
        """
        pair = torch.stack([scores, torch.zeros_like(scores)], dim=-1)
        if self.training:
            decisions = F.gumbel_softmax(pair, tau=self.selection_temperature, hard=True)
        else:
            decisions = F.one_hot(pair.argmax(dim=-1), num_classes=2).float()
        return decisions[..., 0]


SCALE_RANGE = 0.5


class ModalityFusion(nn.Module):
    """Gate the note and laboratory channels, then modulate the patient vector.

    Args:
        hidden_dim: width of the patient vector.
        note_dim: width of one note vector, or zero when notes are not used.
        lab_dim: width of one laboratory vector, or zero when labs are not used.
        note_projection_dim: width of the note channel after projection.
        lab_projection_dim: width of the laboratory channel after projection;
            defaults to a quarter of ``hidden_dim``.
        dropout: dropout applied inside both projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        note_dim: int,
        lab_dim: int,
        note_projection_dim: int = 64,
        lab_projection_dim: int | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.uses_notes = note_dim > 0
        self.uses_labs = lab_dim > 0
        self.note_projection_dim = note_projection_dim if self.uses_notes else 0
        self.lab_projection_dim = (
            (lab_projection_dim or max(16, hidden_dim // 4)) if self.uses_labs else 0
        )

        if self.uses_notes:
            self.note_projection = nn.Sequential(
                nn.Linear(note_dim, self.note_projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.note_gate = nn.Linear(
                hidden_dim + self.note_projection_dim, self.note_projection_dim
            )
            nn.init.zeros_(self.note_gate.bias)
        if self.uses_labs:
            self.lab_projection = nn.Sequential(
                nn.Linear(lab_dim, self.lab_projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.lab_gate = nn.Linear(
                hidden_dim + self.lab_projection_dim, self.lab_projection_dim
            )
            nn.init.zeros_(self.lab_gate.bias)

        modulation_input = hidden_dim + self.note_projection_dim + self.lab_projection_dim
        self.scale = nn.Linear(modulation_input, hidden_dim)
        self.shift = nn.Linear(modulation_input, hidden_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.xavier_normal_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def _gated(
        self,
        patient: torch.Tensor,
        channel: torch.Tensor,
        projection: nn.Module,
        gate: nn.Module,
        available: torch.Tensor,
    ) -> torch.Tensor:
        """Project one channel, gate it against the patient vector and mask it.

        Admissions without the channel are multiplied by zero, so a missing note
        or missing measurements contribute nothing rather than a projected zero
        vector that the gate could still shift.
        """
        projected = projection(channel)
        weights = torch.sigmoid(gate(torch.cat([patient, projected], dim=1)))
        return weights * projected * available.unsqueeze(1)

    def forward(
        self,
        patient: torch.Tensor,
        note_vector: torch.Tensor,
        lab_vector: torch.Tensor,
        has_note: torch.Tensor,
        has_labs: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate the patient vector with the available channels.

        Args:
            patient: ``(batch, hidden_dim)``.
            note_vector: ``(batch, note_dim)``, already mean-centred.
            lab_vector: ``(batch, lab_dim)``.
            has_note: ``(batch,)`` one where a note was found.
            has_labs: ``(batch,)`` one where measurements were found.

        Returns:
            ``(batch, hidden_dim)``.
        """
        channels = [patient]
        if self.uses_notes:
            channels.append(
                self._gated(
                    patient, note_vector, self.note_projection, self.note_gate, has_note
                )
            )
        if self.uses_labs:
            channels.append(
                self._gated(patient, lab_vector, self.lab_projection, self.lab_gate, has_labs)
            )

        combined = torch.cat(channels, dim=1)
        scale = 1.0 + torch.tanh(self.scale(combined)) * SCALE_RANGE
        return scale * patient + self.shift(combined)


NORMALISATION_LIMIT = 1e4


class RelationalGraphLayer(nn.Module):
    """Sum of per-relation neighbour averages, projected and added back.

    Every relation is normalised by the square root of its own node degrees. The
    relations share one projection, which keeps the layer small; the relations
    differ through their edges and weights rather than through their weights
    matrices.

    Args:
        width: width of one node vector.
        dropout: dropout applied to the projected message.
    """

    def __init__(self, width: int, dropout: float = 0.3):
        super().__init__()
        self.project = nn.Linear(width, width)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relation: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Pass messages once over every relation.

        Args:
            nodes: ``(drugs, width)`` current node vectors.
            edge_index: ``(2, edges)`` source and target of each edge.
            edge_relation: ``(edges,)`` relation identifier of each edge.
            edge_weight: ``(edges,)`` weight of each edge.

        Returns:
            ``(drugs, width)`` updated node vectors.
        """
        node_count, width = nodes.shape
        sources, targets = edge_index[0], edge_index[1]
        messages = torch.zeros(node_count, width, device=nodes.device, dtype=nodes.dtype)

        for relation in edge_relation.unique():
            selected = edge_relation == relation
            relation_sources = sources[selected]
            relation_targets = targets[selected]
            weights = edge_weight[selected]

            source_degree = torch.zeros(node_count, device=nodes.device).index_add_(
                0, relation_sources, weights
            )
            target_degree = torch.zeros(node_count, device=nodes.device).index_add_(
                0, relation_targets, weights
            )
            source_scale = source_degree.pow(-0.5).clamp(max=NORMALISATION_LIMIT)
            target_scale = target_degree.pow(-0.5).clamp(max=NORMALISATION_LIMIT)
            source_scale[source_degree == 0] = 0.0
            target_scale[target_degree == 0] = 0.0

            normalised = weights * source_scale[relation_sources] * target_scale[relation_targets]
            messages.index_add_(
                0, relation_targets, nodes[relation_sources] * normalised.unsqueeze(-1)
            )

        updated = self.dropout(self.activation(self.project(messages)))
        return self.norm(nodes + updated)


class DrugGraphEncoder(nn.Module):
    """Message passing over drug classes.

    Args:
        drug_embeddings: ``(drugs, embedding_dim)`` frozen embeddings of the drug
            descriptions, mean-centred by the caller.
        molecular_fingerprints: ``(drugs, fingerprint_bits)`` binary structural
            fingerprints.
        hidden_dim: width of the drug vectors the encoder produces.
        layers: number of message-passing layers.
        dropout: dropout in the input projection and in every layer.
    """

    def __init__(
        self,
        drug_embeddings: torch.Tensor,
        molecular_fingerprints: torch.Tensor,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.register_buffer("drug_embeddings", drug_embeddings)
        self.register_buffer("molecular_fingerprints", molecular_fingerprints)
        feature_width = drug_embeddings.size(1) + molecular_fingerprints.size(1)
        self.input_projection = nn.Sequential(
            nn.Linear(feature_width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            RelationalGraphLayer(width=hidden_dim, dropout=dropout) for _ in range(layers)
        )

    def forward(self, graph: DrugGraph) -> torch.Tensor:
        """Encode every drug class.

        Args:
            graph: the cohort's drug graph, already on the right device.

        Returns:
            ``(drugs, hidden_dim)``.
        """
        nodes = self.input_projection(
            torch.cat([self.drug_embeddings, self.molecular_fingerprints], dim=1)
        )
        for layer in self.layers:
            nodes = layer(nodes, graph.edge_index, graph.edge_relation, graph.edge_weight)
        return nodes


class DrugScorer(nn.Module):
    """Attention-based scorer over the whole drug vocabulary.

    Args:
        hidden_dim: width of the drug and patient vectors.
        heads: attention heads in both attention blocks.
        dropout: dropout inside the attention blocks and the feed-forward block.
    """

    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.drug_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.drug_dropout = nn.Dropout(dropout)

        self.history_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.history_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward_dropout = nn.Dropout(dropout)

        self.continuation_query = nn.Linear(hidden_dim, hidden_dim)
        self.continuation_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        patient: torch.Tensor,
        drug_vectors: torch.Tensor,
        admission_sequence: torch.Tensor,
        lengths: torch.Tensor,
        drug_history: torch.Tensor,
    ) -> torch.Tensor:
        """Score every drug class for every patient in the batch.

        Args:
            patient: ``(batch, hidden_dim)`` the modulated patient vector.
            drug_vectors: ``(drugs, hidden_dim)`` from the graph encoder.
            admission_sequence: ``(batch, admissions, hidden_dim)`` contextual
                admission vectors.
            lengths: ``(batch,)`` number of real admissions per patient.
            drug_history: ``(batch, drugs)`` one for classes prescribed at any
                earlier admission.

        Returns:
            ``(batch, drugs)`` unnormalised scores.
        """
        batch_size = patient.size(0)
        device = patient.device
        admissions = admission_sequence.size(1)
        padding_mask = torch.arange(admissions, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        drugs = drug_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.drug_attention(drugs, drugs, drugs, need_weights=False)
        contextual = self.drug_norm(drugs + self.drug_dropout(attended))

        attended, _ = self.history_attention(
            contextual,
            admission_sequence,
            admission_sequence,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        contextual = self.history_norm(contextual + self.history_dropout(attended))
        contextual = self.feed_forward_norm(
            contextual + self.feed_forward_dropout(self.feed_forward(contextual))
        )

        new_score = self._dot(contextual, patient)
        continuation_score = self._continuation_score(
            contextual, drug_vectors, patient, drug_history
        )
        gate = torch.sigmoid(self.continuation_gate(contextual)).squeeze(-1)
        gate = gate * (drug_history.sum(dim=-1, keepdim=True) > 0).float()
        return new_score * (1 - gate) + continuation_score * gate

    def _dot(self, drug_vectors: torch.Tensor, patient: torch.Tensor) -> torch.Tensor:
        """Score each drug by its agreement with the patient vector."""
        return (drug_vectors * patient.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)

    def _continuation_score(
        self,
        contextual: torch.Tensor,
        drug_vectors: torch.Tensor,
        patient: torch.Tensor,
        drug_history: torch.Tensor,
    ) -> torch.Tensor:
        """Score each drug against the classes the patient already receives.

        Attention is restricted to the classes present in the history, so this
        branch can only argue from the current regimen. Patients with no history
        receive zero weight through the gate in :meth:`forward`.
        """
        batch_size = contextual.size(0)
        query = self.continuation_query(contextual)
        scores = query @ drug_vectors.t().unsqueeze(0) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(drug_history.unsqueeze(1) == 0, float("-inf"))
        weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        summarised = torch.bmm(weights, drug_vectors.unsqueeze(0).expand(batch_size, -1, -1))
        return self._dot(summarised, patient)


MINIMUM_TEMPERATURE = 0.2
COPY_SCALE_RANGE = (0.5, 5.0)


class LabScoringHead(nn.Module):
    """Scores drug classes from the laboratory vector.

    Args:
        lab_dim: width of one laboratory vector.
        hidden_dim: width of the drug vectors the scores are taken against.
        dropout: dropout inside the projection.
    """

    def __init__(self, lab_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        projection_dim = max(16, hidden_dim // 4)
        self.project = nn.Sequential(
            nn.Linear(lab_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_drug_space = nn.Linear(projection_dim, hidden_dim, bias=False)

    def forward(
        self,
        lab_vector: torch.Tensor,
        drug_vectors: torch.Tensor,
        has_labs: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(batch, drugs)`` scores, zero where measurements are missing."""
        projected = self.to_drug_space(self.project(lab_vector))
        scores = (projected @ drug_vectors.t()) / temperature
        return scores * has_labs.unsqueeze(1)


class NoteScoringHead(nn.Module):
    """Scores drug classes from the discharge-note vector.

    The second projection starts as the identity, so the head begins by scoring
    drugs with the note vector as the language model produced it and learns a
    rotation towards the drug space from there.

    Args:
        note_dim: width of one note vector.
        hidden_dim: width of the drug vectors the scores are taken against.
        dropout: dropout inside the projection.
    """

    def __init__(self, note_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(note_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_drug_space = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.to_drug_space.weight)

    def forward(
        self,
        note_vector: torch.Tensor,
        drug_vectors: torch.Tensor,
        has_note: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(batch, drugs)`` scores, zero where the note is missing."""
        projected = self.to_drug_space(self.project(note_vector))
        scores = (projected @ drug_vectors.t()) / temperature
        return scores * has_note.unsqueeze(1)


class CopyHead(nn.Module):
    """Scores drug classes from the prescriptions of individual past admissions.

    Past admissions are attended over with the patient vector as the query, so a
    class that was prescribed at the admissions most relevant now scores higher
    than one prescribed long ago. One gate per patient decides how much of this
    reaches the final score.

    Args:
        drug_count: size of the drug vocabulary.
        hidden_dim: width of the patient vector.
        max_visits: largest admission count the position table covers.
    """

    def __init__(self, drug_count: int, hidden_dim: int, max_visits: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.visit_projection = nn.Linear(drug_count, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.visit_position = nn.Embedding(max_visits, hidden_dim)

    def forward(
        self,
        patient: torch.Tensor,
        drugs_per_visit: torch.Tensor,
        drug_history: torch.Tensor,
        lengths: torch.Tensor,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score continuation of the existing regimen.

        Args:
            patient: ``(batch, hidden_dim)``.
            drugs_per_visit: ``(batch, admissions, drugs)`` binary prescriptions.
            drug_history: ``(batch, drugs)`` union of past prescriptions.
            lengths: ``(batch,)`` number of real admissions.
            temperature: shared scalar temperature.

        Returns:
            The ``(batch, drugs)`` contribution and the ``(batch, 1)`` gate, the
            latter reported for diagnostics.
        """
        device = patient.device
        admissions = drugs_per_visit.size(1)
        gate = torch.sigmoid(self.gate(patient))
        gate = gate * (drug_history.sum(dim=-1, keepdim=True) > 0).float()

        positions = torch.arange(admissions, device=device).clamp(
            max=self.visit_position.num_embeddings - 1
        )
        visits = self.visit_projection(drugs_per_visit)
        visits = visits + self.visit_position(positions).unsqueeze(0)
        scores = (self.query(patient).unsqueeze(1) * self.key(visits)).sum(dim=-1)
        scores = scores / math.sqrt(self.hidden_dim)
        padding = torch.arange(admissions, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(padding, float("-inf"))

        weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        distribution = (weights.unsqueeze(-1) * drugs_per_visit).sum(dim=1)
        contribution = distribution * self.scale.clamp(*COPY_SCALE_RANGE) / temperature
        return gate * contribution, gate


class PredictionHead(nn.Module):
    """Adds the main scorer and the three auxiliary heads into one score per drug.

    Args:
        drug_count: size of the drug vocabulary.
        hidden_dim: width of the patient and drug vectors.
        note_dim: width of one note vector, or zero to leave the note head out.
        lab_dim: width of one laboratory vector, or zero to leave the lab head out.
        heads: attention heads inside the main scorer.
        dropout: dropout inside every block.
        use_copy_head: build the copy head.
        max_visits: largest admission count the copy head's position table covers.
    """

    def __init__(
        self,
        drug_count: int,
        hidden_dim: int = 128,
        note_dim: int = 0,
        lab_dim: int = 0,
        heads: int = 4,
        dropout: float = 0.3,
        use_copy_head: bool = True,
        max_visits: int = 30,
    ):
        super().__init__()
        self.scorer = DrugScorer(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
        self.note_head = NoteScoringHead(note_dim, hidden_dim, dropout) if note_dim else None
        self.lab_head = LabScoringHead(lab_dim, hidden_dim, dropout) if lab_dim > 0 else None
        self.copy_head = (
            CopyHead(drug_count, hidden_dim, max_visits) if use_copy_head else None
        )

        self.scorer_weight = nn.Parameter(torch.tensor(10.0))
        self.note_weight = nn.Parameter(torch.tensor(0.3))
        self.lab_weight = nn.Parameter(torch.tensor(0.2))
        self.raw_temperature = nn.Parameter(torch.tensor(-1.5))

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature shared by the dot-product heads."""
        return F.softplus(self.raw_temperature) + MINIMUM_TEMPERATURE

    def head_weights(self) -> dict[str, float]:
        """Current weight of each head, for logging."""
        weights = {"scorer": torch.sigmoid(self.scorer_weight).item()}
        if self.note_head is not None:
            weights["note"] = torch.sigmoid(self.note_weight).item()
        if self.lab_head is not None:
            weights["lab"] = torch.sigmoid(self.lab_weight).item()
        return weights

    def forward(
        self,
        patient: torch.Tensor,
        drug_vectors: torch.Tensor,
        admission_sequence: torch.Tensor,
        lengths: torch.Tensor,
        drug_history: torch.Tensor,
        drugs_per_visit: torch.Tensor,
        note_vector: torch.Tensor,
        lab_vector: torch.Tensor,
        has_note: torch.Tensor,
        has_labs: torch.Tensor,
        return_contributions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        """Produce one score per drug class.

        Args:
            return_contributions: also return what each head put into the score.
                The four terms sum to the score exactly, which is what makes them
                an account of the decision rather than an estimate of it.

        Returns:
            The ``(batch, drugs)`` scores and the ``(batch, 1)`` copy gate, which
            is zero when the copy head is absent. With ``return_contributions``,
            a third value maps each head's name to its ``(batch, drugs)`` share.
        """
        temperature = self.temperature
        contributions: dict[str, torch.Tensor] = {}

        contributions["history"] = torch.sigmoid(self.scorer_weight) * (
            self.scorer(patient, drug_vectors, admission_sequence, lengths, drug_history)
            / temperature
        )
        scores = contributions["history"]

        if self.note_head is not None:
            contributions["note"] = torch.sigmoid(self.note_weight) * self.note_head(
                note_vector, drug_vectors, has_note, temperature
            )
            scores = scores + contributions["note"]
        if self.lab_head is not None:
            contributions["labs"] = torch.sigmoid(self.lab_weight) * self.lab_head(
                lab_vector, drug_vectors, has_labs, temperature
            )
            scores = scores + contributions["labs"]

        copy_gate = torch.zeros(patient.size(0), 1, device=patient.device)
        if self.copy_head is not None:
            contributions["regimen"], copy_gate = self.copy_head(
                patient, drugs_per_visit, drug_history, lengths, temperature
            )
            scores = scores + contributions["regimen"]

        if return_contributions:
            return scores, copy_gate, contributions
        return scores, copy_gate


def _centred(embeddings: torch.Tensor) -> torch.Tensor:
    """Subtract the mean embedding.

    Language-model embeddings of clinical text share a large common component.
    Removing the mean leaves what distinguishes one code, drug or note from the
    others, which is what the model needs.
    """
    return embeddings - embeddings.mean(dim=0, keepdim=True)


class Mirror(nn.Module):
    """Medication recommendation from coded history, discharge notes and labs.

    Args:
        artifacts: the cohort's embeddings, vocabulary sizes and note statistics.
        config: model sizes and the input-channel switches.
    """

    def __init__(self, artifacts: CohortArtifacts, config: ModelConfig):
        super().__init__()
        self.config = config
        self.drug_count = artifacts.drug_count
        self.uses_notes = config.use_notes and artifacts.note_embeddings is not None
        self.uses_labs = config.use_labs and artifacts.lab_vectors is not None
        note_dim = artifacts.note_dim if self.uses_notes else 0
        lab_dim = artifacts.lab_dim if self.uses_labs else 0

        drug_embeddings = _centred(artifacts.drug_embeddings)
        self.history_encoder = VisitHistoryEncoder(
            diagnosis_embeddings=_centred(artifacts.diagnosis_embeddings),
            procedure_embeddings=_centred(artifacts.procedure_embeddings),
            drug_embeddings=drug_embeddings,
            hidden_dim=config.hidden_dim,
            layers=config.encoder_layers,
            heads=config.attention_heads,
            dropout=config.dropout,
            max_visits=config.max_visits,
        )
        self.visit_selector = (
            VisitSelector(
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                attention_temperature=config.attention_temperature,
                selection_temperature=config.selection_temperature,
            )
            if config.use_visit_selection
            else None
        )
        self.fusion = (
            ModalityFusion(
                hidden_dim=config.hidden_dim,
                note_dim=note_dim,
                lab_dim=lab_dim,
                note_projection_dim=config.note_projection_dim,
                lab_projection_dim=config.lab_projection_dim,
                dropout=config.dropout,
            )
            if note_dim or lab_dim
            else None
        )
        self.drug_graph_encoder = DrugGraphEncoder(
            drug_embeddings=drug_embeddings,
            molecular_fingerprints=artifacts.molecular_fingerprints,
            hidden_dim=config.hidden_dim,
            layers=config.graph_layers,
            dropout=config.dropout,
        )
        self.prediction_head = PredictionHead(
            drug_count=artifacts.drug_count,
            hidden_dim=config.hidden_dim,
            note_dim=note_dim,
            lab_dim=lab_dim,
            heads=config.attention_heads,
            dropout=config.dropout,
            use_copy_head=config.use_copy_head,
            max_visits=config.max_visits,
        )

        note_mean = artifacts.note_mean
        self.register_buffer(
            "note_mean",
            torch.zeros(note_dim) if note_mean is None else torch.from_numpy(note_mean).float(),
        )

    def forward(
        self, batch: dict, graph: DrugGraph, return_contributions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        """Score every drug class for every prediction instance in the batch.

        Args:
            batch: the output of ``collate_instances``, already on the device.
            graph: the cohort's drug graph, already on the device.
            return_contributions: also return what each scoring head put into
                each score.

        Returns:
            The ``(batch, drugs)`` scores and the ``(batch, 1)`` copy gate, and
            with ``return_contributions`` the per-head shares as well.
        """
        lengths = batch["history_length"]
        admission_sequence = self.history_encoder(
            batch["diagnosis_codes"],
            batch["procedure_codes"],
            batch["diagnosis_mask"],
            batch["procedure_mask"],
            lengths,
            batch["drugs_per_visit"],
        )

        if self.visit_selector is not None:
            patient = self.visit_selector(admission_sequence, lengths)
        else:
            rows = torch.arange(admission_sequence.size(0), device=admission_sequence.device)
            patient = admission_sequence[rows, (lengths - 1).clamp(min=0)]

        note_vector = batch["note_vector"]
        has_note = batch["has_note"]
        lab_vector = batch["lab_vector"]
        has_labs = batch["has_labs"]
        if self.uses_notes:
            note_vector = note_vector - self.note_mean
        else:
            has_note = torch.zeros_like(has_note)
        if not self.uses_labs:
            has_labs = torch.zeros_like(has_labs)

        if self.fusion is not None:
            patient = self.fusion(patient, note_vector, lab_vector, has_note, has_labs)

        drug_vectors = self.drug_graph_encoder(graph)
        return self.prediction_head(
            patient=patient,
            drug_vectors=drug_vectors,
            admission_sequence=admission_sequence,
            lengths=lengths,
            drug_history=batch["drug_history"],
            drugs_per_visit=batch["drugs_per_visit"],
            note_vector=note_vector,
            lab_vector=lab_vector,
            has_note=has_note,
            has_labs=has_labs,
            return_contributions=return_contributions,
        )

    def parameter_counts(self) -> dict[str, dict[str, int]]:
        """Count parameters per named part, plus a total.

        The parts are the model's direct children, so the per-part numbers add up
        to the total.
        """
        counts: dict[str, dict[str, int]] = {}
        for name, module in self.named_children():
            parameters = list(module.parameters())
            counts[name] = {
                "total": sum(p.numel() for p in parameters),
                "trainable": sum(p.numel() for p in parameters if p.requires_grad),
            }
        counts["total"] = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return counts
