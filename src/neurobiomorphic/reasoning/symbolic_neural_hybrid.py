"""
Symbolic-Neural Hybrid Reasoning System

Implements a novel architecture that combines symbolic reasoning with neural networks
for compositional generalization and systematic reasoning. Based on recent advances
in neuro-symbolic AI and program synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import sympy as sp
from sympy import symbols, sympify, lambdify
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class SymbolicOperation(Enum):
    """Enumeration of supported symbolic operations."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    LOG = "log"
    EXP = "exp"
    SIN = "sin"
    COS = "cos"
    COMPOSE = "compose"
    CONDITIONAL = "conditional"


@dataclass
class SymbolicExpression:
    """Represents a symbolic expression with neural embeddings."""
    expression: str
    variables: List[str]
    operation: SymbolicOperation
    embedding: Optional[torch.Tensor] = None
    
    def to_sympy(self) -> sp.Expr:
        """Convert to SymPy expression."""
        return sympify(self.expression)
    
    def evaluate(self, values: Dict[str, float]) -> float:
        """Evaluate the expression with given values."""
        expr = self.to_sympy()
        return float(expr.subs(values))
    
    def to_callable(self) -> callable:
        """Convert to callable function."""
        expr = self.to_sympy()
        var_symbols = [symbols(var) for var in self.variables]
        return lambdify(var_symbols, expr, 'numpy')


class NeuralSymbolicEncoder(nn.Module):
    """Encodes symbolic expressions into neural embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_proj = nn.Linear(embedding_dim, hidden_dim)
        
    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode symbolic expression tokens into embeddings.
        
        Args:
            token_ids: [batch_size, seq_len] Token IDs
            attention_mask: [batch_size, seq_len] Attention mask
            
        Returns:
            [batch_size, hidden_dim] Expression embeddings
        """
        embeddings = self.token_embedding(token_ids)
        
        seq_len = embeddings.size(1)
        position_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0)
        position_embeddings = self.token_embedding(position_ids % self.token_embedding.num_embeddings)
        embeddings = embeddings + position_embeddings
        
        embeddings = embeddings.transpose(0, 1)
        
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask
        
        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        if attention_mask is not None:
            mask_expanded = (~attention_mask).unsqueeze(-1).float()
            encoded = (encoded.transpose(0, 1) * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            encoded = encoded.transpose(0, 1).mean(1)
        
        return self.output_proj(encoded)


class NeuralSymbolicDecoder(nn.Module):
    """Decodes neural embeddings back to symbolic expressions."""
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        embedding_dim: int,
        max_length: int = 64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.input_proj = nn.Linear(hidden_dim, embedding_dim)
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        self.output_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(
        self,
        encoded_expr: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode neural embedding to symbolic expression tokens.
        
        Args:
            encoded_expr: [batch_size, hidden_dim] Encoded expression
            target_tokens: [batch_size, seq_len] Target tokens (for training)
            
        Returns:
            [batch_size, seq_len, vocab_size] Token logits
        """
        batch_size = encoded_expr.size(0)
        device = encoded_expr.device
        
        memory = self.input_proj(encoded_expr).unsqueeze(0)
        
        if target_tokens is not None:
            seq_len = target_tokens.size(1)
            tgt_embeddings = self.token_embedding(target_tokens)
            tgt_embeddings = tgt_embeddings.transpose(0, 1)
            
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)
            
            decoded = self.transformer(tgt_embeddings, memory, tgt_mask=tgt_mask)
            output = self.output_head(decoded.transpose(0, 1))
        else:
            output_tokens = []
            hidden_state = memory
            
            # Start with BOS token
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            
            for _ in range(self.max_length):
                tgt_embeddings = self.token_embedding(current_token).transpose(0, 1)
                decoded = self.transformer(tgt_embeddings, hidden_state)
                token_logits = self.output_head(decoded[-1])  # [batch, vocab_size]
                
                # Sample next token
                next_token = torch.argmax(token_logits, dim=-1, keepdim=True)
                output_tokens.append(token_logits.unsqueeze(1))
                current_token = next_token
                
                # Stop if EOS token is generated
                if (next_token == 1).all():  # Assuming EOS token ID is 1
                    break
            
            output = torch.cat(output_tokens, dim=1) if output_tokens else torch.zeros(batch_size, 1, self.vocab_size, device=device)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class SymbolicOperationModule(nn.Module):
    """Neural module for executing symbolic operations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, operation: SymbolicOperation):
        super().__init__()
        self.operation = operation
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Operation-specific networks
        if operation == SymbolicOperation.ADD:
            self.network = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        elif operation == SymbolicOperation.MULTIPLY:
            self.network = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        elif operation == SymbolicOperation.COMPOSE:
            self.network = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        elif operation == SymbolicOperation.CONDITIONAL:
            # Condition network
            self.condition_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            # Branch networks
            self.true_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            self.false_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        else:
            # Generic operation network
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Execute the symbolic operation."""
        if self.operation == SymbolicOperation.ADD:
            assert len(inputs) == 2, "ADD operation requires exactly 2 inputs"
            combined = torch.cat(inputs, dim=-1)
            return self.network(combined)
        
        elif self.operation == SymbolicOperation.MULTIPLY:
            assert len(inputs) == 2, "MULTIPLY operation requires exactly 2 inputs"
            combined = torch.cat(inputs, dim=-1)
            return self.network(combined)
        
        elif self.operation == SymbolicOperation.COMPOSE:
            assert len(inputs) == 2, "COMPOSE operation requires exactly 2 inputs"
            combined = torch.cat(inputs, dim=-1)
            return self.network(combined)
        
        elif self.operation == SymbolicOperation.CONDITIONAL:
            assert len(inputs) == 3, "CONDITIONAL operation requires exactly 3 inputs (condition, true_branch, false_branch)"
            condition_input, true_input, false_input = inputs
            
            # Compute condition probability
            condition_prob = self.condition_net(condition_input)
            
            # Compute branch outputs
            true_output = self.true_branch(true_input)
            false_output = self.false_branch(false_input)
            
            # Weighted combination based on condition
            return condition_prob * true_output + (1 - condition_prob) * false_output
        
        else:
            # Single input operations
            assert len(inputs) == 1, f"{self.operation} operation requires exactly 1 input"
            return self.network(inputs[0])


class SymbolicNeuralHybridSystem(nn.Module):
    """
    Main hybrid system that combines symbolic and neural reasoning.
    
    This system can:
    - Parse symbolic expressions into neural representations
    - Perform neural-symbolic computation
    - Generate new symbolic expressions
    - Learn compositional rules from data
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        max_expression_length: int = 64,
        supported_operations: List[SymbolicOperation] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_expression_length = max_expression_length
        
        if supported_operations is None:
            supported_operations = [
                SymbolicOperation.ADD,
                SymbolicOperation.MULTIPLY,
                SymbolicOperation.COMPOSE,
                SymbolicOperation.CONDITIONAL
            ]
        
        # Core components
        self.encoder = NeuralSymbolicEncoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = NeuralSymbolicDecoder(hidden_dim, vocab_size, embedding_dim, max_expression_length)
        
        # Operation modules
        self.operation_modules = nn.ModuleDict({
            op.value: SymbolicOperationModule(hidden_dim, hidden_dim, op)
            for op in supported_operations
        })
        
        # Operation classifier (to choose which operation to apply)
        self.operation_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(supported_operations)),
            nn.Softmax(dim=-1)
        )
        
        # Rule memory for storing learned compositional rules
        self.rule_memory = nn.Parameter(torch.randn(100, hidden_dim))  # 100 rules max
        self.rule_usage = nn.Parameter(torch.zeros(100))
        
    def encode_expression(self, expression: str, tokenizer) -> torch.Tensor:
        """Encode a symbolic expression into neural representation."""
        tokens = tokenizer.encode(expression)
        token_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        if token_ids.device != next(self.parameters()).device:
            token_ids = token_ids.to(next(self.parameters()).device)
        
        return self.encoder(token_ids)
    
    def decode_expression(self, embedding: torch.Tensor, tokenizer) -> str:
        """Decode neural representation back to symbolic expression."""
        token_logits = self.decoder(embedding)
        token_ids = torch.argmax(token_logits, dim=-1).squeeze(0)
        return tokenizer.decode(token_ids.tolist())
    
    def apply_operation(
        self,
        operation: SymbolicOperation,
        operands: List[torch.Tensor]
    ) -> torch.Tensor:
        """Apply a symbolic operation to neural representations."""
        if operation.value not in self.operation_modules:
            raise ValueError(f"Operation {operation} not supported")
        
        operation_module = self.operation_modules[operation.value]
        return operation_module(*operands)
    
    def forward(
        self,
        expressions: List[torch.Tensor],
        target_expression: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for symbolic-neural reasoning.
        
        Args:
            expressions: List of encoded input expressions
            target_expression: Target output expression (for training)
            
        Returns:
            Tuple of (output_expression_embedding, operation_probs)
        """
        # Combine input expressions
        if len(expressions) == 1:
            combined = expressions[0]
        else:
            combined = torch.stack(expressions, dim=0).mean(dim=0)
        
        # Predict operation to apply
        operation_probs = self.operation_classifier(combined)
        
        # Apply operations with soft attention
        operation_outputs = []
        for op_name, op_module in self.operation_modules.items():
            if len(expressions) >= 2:
                output = op_module(*expressions[:2])  # Use first 2 expressions
            else:
                output = op_module(expressions[0])
            operation_outputs.append(output)
        
        # Weighted combination of operation outputs
        operation_outputs = torch.stack(operation_outputs, dim=1)  # [batch, n_ops, hidden_dim]
        
        # Apply soft attention over operations
        attended_output = torch.bmm(
            operation_probs.unsqueeze(1),  # [batch, 1, n_ops]
            operation_outputs  # [batch, n_ops, hidden_dim]
        ).squeeze(1)  # [batch, hidden_dim]
        
        return attended_output, operation_probs
    
    def compositional_generalization_loss(
        self,
        input_expressions: List[torch.Tensor],
        target_expressions: torch.Tensor,
        target_operations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for compositional generalization."""
        
        # Forward pass
        output_embedding, operation_probs = self.forward(input_expressions)
        
        # Reconstruction loss
        predicted_tokens = self.decoder(output_embedding)
        if target_expressions.dim() == 2:  # If target is token sequence
            reconstruction_loss = F.cross_entropy(
                predicted_tokens.view(-1, predicted_tokens.size(-1)),
                target_expressions.view(-1)
            )
        else:  # If target is embedding
            reconstruction_loss = F.mse_loss(output_embedding, target_expressions)
        
        # Operation prediction loss
        operation_loss = torch.tensor(0.0, device=output_embedding.device)
        if target_operations is not None:
            operation_loss = F.cross_entropy(operation_probs, target_operations)
        
        # Compositional consistency loss (encourage systematic generalization)
        consistency_loss = self._compositional_consistency_loss(input_expressions, output_embedding)
        
        # Rule regularization (encourage learning of reusable rules)
        rule_loss = self._rule_regularization_loss()
        
        total_loss = (
            reconstruction_loss +
            0.1 * operation_loss +
            0.05 * consistency_loss +
            0.01 * rule_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'operation_loss': operation_loss,
            'consistency_loss': consistency_loss,
            'rule_loss': rule_loss
        }
    
    def _compositional_consistency_loss(
        self,
        input_expressions: List[torch.Tensor],
        output_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Encourage compositional consistency in representations."""
        if len(input_expressions) < 2:
            return torch.tensor(0.0, device=output_embedding.device)
        
        # Compute pairwise similarities between inputs
        input_similarities = []
        for i in range(len(input_expressions)):
            for j in range(i + 1, len(input_expressions)):
                sim = F.cosine_similarity(input_expressions[i], input_expressions[j], dim=-1)
                input_similarities.append(sim)
        
        if not input_similarities:
            return torch.tensor(0.0, device=output_embedding.device)
        
        # Encourage output to reflect input structure
        input_sim_mean = torch.stack(input_similarities).mean()
        consistency_target = input_sim_mean * 0.5  # Scale down for output
        
        # This is a simplified consistency loss - can be made more sophisticated
        consistency_loss = F.mse_loss(
            torch.norm(output_embedding, dim=-1).mean().unsqueeze(0),
            consistency_target.unsqueeze(0)
        )
        
        return consistency_loss
    
    def _rule_regularization_loss(self) -> torch.Tensor:
        """Regularize rule memory to encourage learning of reusable rules."""
        # Encourage sparsity in rule usage
        usage_sparsity = torch.sum(torch.sigmoid(self.rule_usage))
        
        # Encourage diversity in rule representations
        rule_similarities = F.cosine_similarity(
            self.rule_memory.unsqueeze(1),
            self.rule_memory.unsqueeze(0),
            dim=-1
        )
        # Penalize high similarities (off-diagonal elements)
        diversity_loss = torch.sum(torch.triu(rule_similarities, diagonal=1) ** 2)
        
        return 0.1 * usage_sparsity + 0.01 * diversity_loss
    
    def generate_new_expression(
        self,
        seed_expressions: List[str],
        tokenizer,
        max_iterations: int = 5
    ) -> List[str]:
        """Generate new expressions by combining existing ones."""
        generated_expressions = []
        
        # Encode seed expressions
        encoded_seeds = [self.encode_expression(expr, tokenizer) for expr in seed_expressions]
        
        for iteration in range(max_iterations):
            # Sample pairs of expressions
            if len(encoded_seeds) >= 2:
                expr1, expr2 = np.random.choice(encoded_seeds, 2, replace=False)
                
                # Apply random operation
                output_embedding, _ = self.forward([expr1, expr2])
                
                # Decode to symbolic expression
                new_expression = self.decode_expression(output_embedding, tokenizer)
                generated_expressions.append(new_expression)
                
                # Add to pool for next iteration
                encoded_seeds.append(output_embedding)
        
        return generated_expressions