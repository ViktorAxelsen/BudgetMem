from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import module descriptions
try:
    from module_description import (
        Module1FilterDescription,
        Module2EntityRelationDescription,
        Module3TemporalRelationDescription,
        Module4SummaryDescription,
        Module5TopicRelationDescription
    )
except ImportError:
    # Fallback if module_description.py is not available
    Module1FilterDescription = "Module 1 (Filter): Selects the top-k memories that are most relevant to the query intent by scoring semantic relevance, reducing noise and limiting downstream computation to high-signal memories."
    Module2EntityRelationDescription = "Module 2 (Entity Relation): Extracts query-relevant entities and their relationships from the filtered memories, focusing on factual, structured relations while removing redundant or weakly supported information."
    Module3TemporalRelationDescription = "Module 3 (Temporal Relation): Identifies and organizes temporal information from the filtered memories, extracting time constraints, event orderings, and temporal dependencies that affect how entities and relations should be interpreted."
    Module4SummaryDescription = "Module 4 (Summary): Synthesizes filtered memories together with extracted entity, temporal, and topic relations into a coherent knowledge summary, and provides explicit reasoning steps that a later language model can follow to answer the query, without directly answering the query itself."
    Module5TopicRelationDescription = "Module 5 (Topic Relation): Analyzes thematic connections and topic transitions within the filtered memories, identifying main discussion topics, topic shifts, and thematic relationships that provide context for understanding the conversation flow and intent."


# ============================================================================
# MLP Decision Network with Critic (Actor-Critic architecture) - Unified MLP for 4 modules
# ============================================================================

class ActorCriticNetwork(nn.Module):
    """
    Unified Actor-Critic Network - Single MLP for all 5 modules

    Uses module description embeddings to distinguish different modules:
    - Module 1: query + initial_memories + module1_desc → action (low/mid/high)
    - Module 2: query + filtered_memories + module2_desc → action
    - Module 3: query + filtered_memories + module3_desc → action
    - Module 5: query + filtered_memories + module5_desc → action
    - Module 4: query + aggregated_memory_emb + module4_desc → action

    Key changes from 4-agent to 1-agent:
    - Single shared encoder and action head (instead of 4 separate ones)
    - Module description embeddings added to input to differentiate modules
    - Projection layers to handle different input dimensions
    """
    def __init__(
        self,
        query_dim: int = 768,
        memory_dim: int = 768,
        desc_dim: int = 768,  # dimension of module description embeddings
        hidden_dim: int = 256,
        projection_dim: int = 256,  # dimension after projection
        num_actions_per_module: int = 3,  # low/mid/high
        module_descriptions: Optional[Dict[str, str]] = None,  # module descriptions (not used, kept for compatibility)
        desc_encoder=None  # Optional encoder for initializing description embeddings from text
    ):
        super().__init__()

        self.num_modules = 5
        self.num_actions_per_module = num_actions_per_module
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.desc_dim = desc_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim

        # Shared projection layer for all 4 modules (they all have same input dimension)
        # Input: query(768) + memory(768) = 1536 → projection_dim
        self.shared_proj = nn.Linear(query_dim + memory_dim, projection_dim)

        # Module description projection layer
        # Projects module description embedding to projection_dim
        self.module_desc_proj = nn.Linear(desc_dim, projection_dim)

        # Unified shared encoder (input: projected_features + projected_desc = 2 * projection_dim)
        unified_input_dim = projection_dim * 2
        self.shared_encoder = nn.Sequential(
            nn.Linear(unified_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Unified action head (shared by all 4 modules)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions_per_module)
        )

        # Critic: Value estimation head (based on initial state)
        # In 1-step bandit setting, critic estimates V(s0) not V(s4)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize module description embeddings from text descriptions
        # These are learnable parameters that encode the identity/role of each module
        self.module_desc_embeddings = self._init_module_desc_embeddings(desc_encoder)

        # Initialize weights
        self._init_weights()

    def _init_module_desc_embeddings(self, desc_encoder=None):
        """
        Initialize module description embeddings from text descriptions.
        
        Args:
            desc_encoder: Optional encoder model for generating embeddings from text.
                         If None, will use random initialization.
        
        Returns:
            nn.ParameterDict with initialized embeddings
        """
        descriptions = {
            'module1': Module1FilterDescription,
            'module2': Module2EntityRelationDescription,
            'module3': Module3TemporalRelationDescription,
            'module5': Module5TopicRelationDescription,
            'module4': Module4SummaryDescription
        }
        
        embeddings_dict = {}
        
        if desc_encoder is not None:
            # Use provided encoder to generate embeddings from text
            try:
                # Try to use the encoder (could be SentenceTransformer or similar)
                if hasattr(desc_encoder, 'encode'):
                    # SentenceTransformer style
                    desc_texts = [descriptions[f'module{i}'] for i in [1, 2, 3, 5, 4]]
                    desc_embs = desc_encoder.encode(
                        desc_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    for i, module_key in enumerate(['module1', 'module2', 'module3', 'module5', 'module4']):
                        emb = torch.from_numpy(desc_embs[i]).float()
                        # Ensure dimension matches
                        if emb.shape[0] != self.desc_dim:
                            # Project to correct dimension if needed
                            if emb.shape[0] > self.desc_dim:
                                emb = emb[:self.desc_dim]
                            else:
                                padding = torch.zeros(self.desc_dim - emb.shape[0])
                                emb = torch.cat([emb, padding])
                        embeddings_dict[module_key] = nn.Parameter(emb)
                else:
                    # Fallback to random initialization
                    print("Warning: desc_encoder provided but doesn't have 'encode' method, using random initialization")
                    for module_key in ['module1', 'module2', 'module3', 'module5', 'module4']:
                        embeddings_dict[module_key] = nn.Parameter(torch.randn(self.desc_dim) * 0.01)
            except Exception as e:
                print(f"Warning: Failed to initialize description embeddings from encoder: {e}")
                print("Falling back to random initialization")
                for module_key in ['module1', 'module2', 'module3', 'module5', 'module4']:
                    embeddings_dict[module_key] = nn.Parameter(torch.randn(self.desc_dim) * 0.01)
        else:
            # Random initialization (original behavior)
            for module_key in ['module1', 'module2', 'module3', 'module5', 'module4']:
                embeddings_dict[module_key] = nn.Parameter(torch.randn(self.desc_dim) * 0.01)
        
        return nn.ParameterDict(embeddings_dict)

    def _init_weights(self):
        """Orthogonal initialization (recommended for PPO)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Value head uses smaller initialization
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)

    def _forward_unified(
        self,
        features: torch.Tensor,
        module_desc_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward pass for any module (all modules now use shared projection)

        Args:
            features: [batch_size, feature_dim] - concatenated features (query + memory embeddings)
            module_desc_emb: [batch_size, desc_dim] - module description embedding

        Returns:
            action_logits: [batch_size, num_actions]
            encoded_features: [batch_size, hidden_dim]
        """
        # Project features to unified dimension (all modules use shared_proj)
        projected_features = self.shared_proj(features)  # [batch_size, projection_dim]

        # Project module description
        projected_desc = self.module_desc_proj(module_desc_emb)  # [batch_size, projection_dim]

        # Concatenate projected features and description
        combined = torch.cat([projected_features, projected_desc], dim=-1)  # [batch_size, 2*projection_dim]

        # Pass through shared encoder
        encoded = self.shared_encoder(combined)  # [batch_size, hidden_dim]

        # Get action logits from shared action head
        action_logits = self.action_head(encoded)  # [batch_size, num_actions]

        return action_logits, encoded

    def forward_module1(self, query_emb: torch.Tensor, initial_memory_emb: torch.Tensor):
        """
        Module 1 decision: based on query and initial memories

        Args:
            query_emb: [batch_size, query_dim]
            initial_memory_emb: [batch_size, memory_dim] - aggregated embedding of initial memories

        Returns:
            module1_logits: [batch_size, num_actions]
            features1: [batch_size, hidden_dim] - for later value function computation
        """
        # Clear GPU cache before forward pass to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_size = query_emb.shape[0]
        combined = torch.cat([query_emb, initial_memory_emb], dim=-1)

        # Get module1 description embedding
        module_desc_emb = self.module_desc_embeddings['module1'].unsqueeze(0).expand(batch_size, -1)

        module1_logits, features1 = self._forward_unified(combined, module_desc_emb)
        
        # Clear GPU cache after forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return module1_logits, features1

    def forward_module2(self, query_emb: torch.Tensor, filtered_memory_emb: torch.Tensor):
        """
        Module 2 decision: based on query and filtered memories (Module1 output)

        Args:
            query_emb: [batch_size, query_dim]
            filtered_memory_emb: [batch_size, memory_dim] - aggregated filtered memories

        Returns:
            module2_logits: [batch_size, num_actions]
            features2: [batch_size, hidden_dim]
        """
        batch_size = query_emb.shape[0]
        combined = torch.cat([query_emb, filtered_memory_emb], dim=-1)

        # Get module2 description embedding
        module_desc_emb = self.module_desc_embeddings['module2'].unsqueeze(0).expand(batch_size, -1)

        module2_logits, features2 = self._forward_unified(combined, module_desc_emb)
        return module2_logits, features2

    def forward_module3(self, query_emb: torch.Tensor, filtered_memory_emb: torch.Tensor):
        """
        Module 3 decision: based on query and filtered memories (Module1 output)

        Args:
            query_emb: [batch_size, query_dim]
            filtered_memory_emb: [batch_size, memory_dim] - aggregated filtered memories

        Returns:
            module3_logits: [batch_size, num_actions]
            features3: [batch_size, hidden_dim]
        """
        batch_size = query_emb.shape[0]
        combined = torch.cat([query_emb, filtered_memory_emb], dim=-1)

        # Get module3 description embedding
        module_desc_emb = self.module_desc_embeddings['module3'].unsqueeze(0).expand(batch_size, -1)

        module3_logits, features3 = self._forward_unified(combined, module_desc_emb)
        return module3_logits, features3

    def forward_module5(self, query_emb: torch.Tensor, filtered_memory_emb: torch.Tensor):
        """
        Module 5 decision: based on query and filtered memories (Module1 output)

        Args:
            query_emb: [batch_size, query_dim]
            filtered_memory_emb: [batch_size, memory_dim] - aggregated filtered memories

        Returns:
            module5_logits: [batch_size, num_actions]
            features5: [batch_size, hidden_dim]
        """
        batch_size = query_emb.shape[0]
        combined = torch.cat([query_emb, filtered_memory_emb], dim=-1)

        # Get module5 description embedding
        module_desc_emb = self.module_desc_embeddings['module5'].unsqueeze(0).expand(batch_size, -1)

        module5_logits, features5 = self._forward_unified(combined, module_desc_emb)
        return module5_logits, features5

    def forward_module4(self, query_emb: torch.Tensor, aggregated_memory_emb: torch.Tensor):
        """
        Module 4 decision: based on query and aggregated memory embeddings (entity + temporal)

        Args:
            query_emb: [batch_size, query_dim]
            aggregated_memory_emb: [batch_size, memory_dim] - aggregated embedding of entity/temporal relations

        Returns:
            module4_logits: [batch_size, num_actions]
            features4: [batch_size, hidden_dim]
        """
        batch_size = query_emb.shape[0]
        combined = torch.cat([query_emb, aggregated_memory_emb], dim=-1)

        # Get module4 description embedding
        module_desc_emb = self.module_desc_embeddings['module4'].unsqueeze(0).expand(batch_size, -1)

        module4_logits, features4 = self._forward_unified(combined, module_desc_emb)
        return module4_logits, features4

    def forward(
        self,
        query_emb: torch.Tensor,
        initial_memory_emb: torch.Tensor,
        filtered_memory_emb: torch.Tensor,
        aggregated_memory_emb: torch.Tensor
    ):
        """
        Complete forward pass (for batch computation during training)

        Args:
            query_emb: [batch_size, query_dim]
            initial_memory_emb: [batch_size, memory_dim]
            filtered_memory_emb: [batch_size, memory_dim]
            aggregated_memory_emb: [batch_size, memory_dim] - aggregated from entity/temporal embeddings

        Returns:
            module1_logits, module2_logits, module3_logits, module5_logits, module4_logits, state_value
        """
        # Sequentially compute logits for each module
        module1_logits, features1 = self.forward_module1(query_emb, initial_memory_emb)
        module2_logits, _ = self.forward_module2(query_emb, filtered_memory_emb)
        module3_logits, _ = self.forward_module3(query_emb, filtered_memory_emb)
        module5_logits, _ = self.forward_module5(query_emb, filtered_memory_emb)
        module4_logits, _ = self.forward_module4(query_emb, aggregated_memory_emb)

        # Value function based on initial state (Module1 features, corresponds to V(s0))
        # In 1-step bandit setting, this is the correct baseline
        state_value = self.value_head(features1).squeeze(-1)

        return module1_logits, module2_logits, module3_logits, module5_logits, module4_logits, state_value

    def _get_action_from_logits(self, logits: torch.Tensor, action: torch.Tensor = None, deterministic: bool = False):
        """Helper function: get action, log_prob and entropy from logits"""
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def get_module1_action(self, query_emb: torch.Tensor, initial_memory_emb: torch.Tensor,
                           action: torch.Tensor = None, deterministic: bool = False):
        """Step 1: Get Module1 action and initial state value"""
        logits, features = self.forward_module1(query_emb, initial_memory_emb)
        action_result = self._get_action_from_logits(logits, action, deterministic)
        # Compute initial state value (V(s0))
        state_value = self.value_head(features).squeeze(-1)
        return (*action_result, state_value)

    def get_module2_action(self, query_emb: torch.Tensor, filtered_memory_emb: torch.Tensor,
                           action: torch.Tensor = None, deterministic: bool = False):
        """Step 2: Get Module2 action (requires Module1 output)"""
        logits, _ = self.forward_module2(query_emb, filtered_memory_emb)
        return self._get_action_from_logits(logits, action, deterministic)

    def get_module3_action(self, query_emb: torch.Tensor, filtered_memory_emb: torch.Tensor,
                           action: torch.Tensor = None, deterministic: bool = False):
        """Step 3: Get Module3 action (requires Module1 output)"""
        logits, _ = self.forward_module3(query_emb, filtered_memory_emb)
        return self._get_action_from_logits(logits, action, deterministic)

    def get_module5_action(self, query_emb: torch.Tensor, filtered_memory_emb: torch.Tensor,
                           action: torch.Tensor = None, deterministic: bool = False):
        """Step 5: Get Module5 action (requires Module1 output)"""
        logits, _ = self.forward_module5(query_emb, filtered_memory_emb)
        return self._get_action_from_logits(logits, action, deterministic)

    def get_module4_action(self, query_emb: torch.Tensor, aggregated_memory_emb: torch.Tensor,
                           action: torch.Tensor = None, deterministic: bool = False):
        """Step 4: Get Module4 action (requires aggregated memory embedding from Module2&3 output)"""
        logits, _ = self.forward_module4(query_emb, aggregated_memory_emb)
        return self._get_action_from_logits(logits, action, deterministic)

    def get_action_and_value(
        self,
        query_emb: torch.Tensor,
        initial_memory_emb: torch.Tensor,
        filtered_memory_emb: torch.Tensor,
        aggregated_memory_emb: torch.Tensor,
        actions: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False
    ):
        """
        Batch computation version - calculate log_probs for given embeddings during PPO training
        
        Args:
            aggregated_memory_emb: [batch_size, memory_dim] - aggregated from entity/temporal embeddings
        """
        m1_logits, m2_logits, m3_logits, m5_logits, m4_logits, state_value = self.forward(
            query_emb, initial_memory_emb, filtered_memory_emb, aggregated_memory_emb
        )

        m1_probs = F.softmax(m1_logits, dim=-1)
        m2_probs = F.softmax(m2_logits, dim=-1)
        m3_probs = F.softmax(m3_logits, dim=-1)
        m5_probs = F.softmax(m5_logits, dim=-1)
        m4_probs = F.softmax(m4_logits, dim=-1)

        m1_dist = torch.distributions.Categorical(m1_probs)
        m2_dist = torch.distributions.Categorical(m2_probs)
        m3_dist = torch.distributions.Categorical(m3_probs)
        m5_dist = torch.distributions.Categorical(m5_probs)
        m4_dist = torch.distributions.Categorical(m4_probs)

        # NOTE: In PPO update, MUST pass sampled actions to prevent re-sampling
        # When actions is provided, use them directly instead of sampling
        if actions is None:
            if deterministic:
                m1_action = torch.argmax(m1_probs, dim=-1)
                m2_action = torch.argmax(m2_probs, dim=-1)
                m3_action = torch.argmax(m3_probs, dim=-1)
                m5_action = torch.argmax(m5_probs, dim=-1)
                m4_action = torch.argmax(m4_probs, dim=-1)
            else:
                m1_action = m1_dist.sample()
                m2_action = m2_dist.sample()
                m3_action = m3_dist.sample()
                m5_action = m5_dist.sample()
                m4_action = m4_dist.sample()
        else:
            # CRITICAL: Use actions from rollout, do NOT re-sample
            m1_action, m2_action, m3_action, m5_action, m4_action = actions

        # CRITICAL: Return average entropy instead of sum to match single-agent scale
        # Sum of 5 entropies would make exploration term 5x larger, causing training instability
        avg_entropy = (m1_dist.entropy() + m2_dist.entropy() + m3_dist.entropy() + m5_dist.entropy() + m4_dist.entropy()) / 5.0

        return (
            [m1_action, m2_action, m3_action, m5_action, m4_action],
            [m1_dist.log_prob(m1_action), m2_dist.log_prob(m2_action),
             m3_dist.log_prob(m3_action), m5_dist.log_prob(m5_action), m4_dist.log_prob(m4_action)],
            avg_entropy,  # Average entropy to match single-agent exploration scale
            state_value
        )


# ============================================================================
# Experience Buffer for PPO - 4 modules version
# ============================================================================

@dataclass
class Experience:
    """Single experience - supports 5-module sequential decision making"""
    query_emb: torch.Tensor
    memory_embs: torch.Tensor  # Keep for compatibility
    actions: List[int]  # [m1_action, m2_action, m3_action, m5_action, m4_action]
    log_probs: List[float]  # [m1_log_prob, m2_log_prob, m3_log_prob, m5_log_prob, m4_log_prob]
    reward: float
    cost: float
    value: float
    done: bool = True
    # Intermediate embeddings - for sequential decision PPO training
    # CRITICAL: These must match exactly what was used during action selection
    initial_memory_emb: Optional[torch.Tensor] = None
    filtered_memory_emb: Optional[torch.Tensor] = None
    entity_emb: Optional[torch.Tensor] = None
    temporal_emb: Optional[torch.Tensor] = None
    topic_emb: Optional[torch.Tensor] = None
    aggregated_memory_emb: Optional[torch.Tensor] = None  # CRITICAL: for Module4 action selection (aggregated from entity/temporal/topic)


class ExperienceBuffer:
    """
    Experience buffer - supports 4 modules

    Stores all experiences in an episode for PPO update
    """
    def __init__(self):
        self.experiences: List[Experience] = []

    def add(self, exp: Experience):
        self.experiences.append(exp)

    def clear(self):
        self.experiences.clear()

    def __len__(self):
        return len(self.experiences)

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        reward_weight: float = 1.0,  # Reward weight (default 1.0, no scaling)
        cost_weight: float = 0.0  # Cost penalty weight (0 means no cost penalty)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages

        Note: Each experience is an independent episode (done=True),
        so use MC return instead of GAE, or compute GAE separately for each episode.

        reward = reward_weight * f1_reward + cost_weight * cost
        """
        # Compute combined reward (reward_weight * F1 reward + cost efficiency bonus)
        rewards = [reward_weight * exp.reward + cost_weight * exp.cost for exp in self.experiences]
        values = [exp.value for exp in self.experiences]
        dones = [exp.done for exp in self.experiences]

        n = len(rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        # Check if all experiences are independent episodes (done=True)
        all_done = all(dones)

        if all_done:
            # All experiences are independent episodes, use MC return
            # return = reward (since done=True, no future rewards)
            # advantage = reward - value
            for t in range(n):
                returns[t] = rewards[t]  # MC return: directly use reward
                advantages[t] = rewards[t] - values[t]  # Advantage = reward - value
        else:
            # Mixed case: some done=True, some done=False
            # Use GAE, but handle done=True correctly
            gae = 0
            for t in reversed(range(n)):
                if dones[t]:
                    # Current episode ends, no future rewards
                    next_value = 0
                    # TD error
                    delta = rewards[t] - values[t]
                    # GAE (episode ends, don't accumulate)
                    gae = delta
                else:
                    # Current episode not ended, has future rewards
                    if t == n - 1:
                        next_value = 0  # Last one, no next
                    else:
                        next_value = values[t + 1] if not dones[t + 1] else 0

                    # TD error
                    delta = rewards[t] + gamma * next_value - values[t]
                    # GAE
                    gae = delta + gamma * gae_lambda * gae

                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

        # Normalize advantages
        # Note: Cannot normalize with single sample (std=0), but will be resolved after batch update
        if normalize_advantages and len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:  # Ensure std is not 0
                advantages = (advantages - adv_mean) / adv_std
            else:
                # If std too small, only center
                advantages = advantages - adv_mean

        return returns, advantages

    def get_batch_tensors(self, device: torch.device, reward_weight: float = 1.0, cost_weight: float = 0.0) -> Dict[str, torch.Tensor]:
        """Get batch tensors - includes intermediate embeddings"""
        returns, advantages = self.compute_returns_and_advantages(reward_weight=reward_weight, cost_weight=cost_weight)

        # Extract 5 modules' actions and log_probs
        # CRITICAL: old_log_probs must come from rollout-time policy and strictly correspond to [m1, m2, m3, m5, m4]
        # The order MUST be [m1, m2, m3, m5, m4] to match the order in pipeline.execute() return

        # CRITICAL: Validate actions and log_probs length/order to prevent silent misalignment
        for i, exp in enumerate(self.experiences):
            if len(exp.actions) != 5:
                raise ValueError(
                    f"Experience {i} has {len(exp.actions)} actions, expected 5. "
                    f"Actions: {exp.actions}. This will cause PPO to learn incorrectly."
                )
            if len(exp.log_probs) != 5:
                raise ValueError(
                    f"Experience {i} has {len(exp.log_probs)} log_probs, expected 5. "
                    f"Log_probs: {exp.log_probs}. This will cause PPO to learn incorrectly."
                )
        
        m1_actions = torch.tensor([exp.actions[0] for exp in self.experiences], dtype=torch.long)
        m2_actions = torch.tensor([exp.actions[1] for exp in self.experiences], dtype=torch.long)
        m3_actions = torch.tensor([exp.actions[2] for exp in self.experiences], dtype=torch.long)
        m5_actions = torch.tensor([exp.actions[3] for exp in self.experiences], dtype=torch.long)
        m4_actions = torch.tensor([exp.actions[4] for exp in self.experiences], dtype=torch.long)

        # Extract log_probs in strict order [m1, m2, m3, m5, m4] matching rollout-time order
        m1_log_probs = torch.tensor([exp.log_probs[0] for exp in self.experiences], dtype=torch.float32)
        m2_log_probs = torch.tensor([exp.log_probs[1] for exp in self.experiences], dtype=torch.float32)
        m3_log_probs = torch.tensor([exp.log_probs[2] for exp in self.experiences], dtype=torch.float32)
        m5_log_probs = torch.tensor([exp.log_probs[3] for exp in self.experiences], dtype=torch.float32)
        m4_log_probs = torch.tensor([exp.log_probs[4] for exp in self.experiences], dtype=torch.float32)

        # Extract intermediate embeddings (if exist)
        def get_emb(exp, attr, default_dim=768):
            emb = getattr(exp, attr, None)
            return emb if emb is not None else torch.zeros(default_dim)

        # Aggregate entity_emb, temporal_emb, and topic_emb for Module4 (average pooling to match Module1-5 dimension)
        # CRITICAL: Fallback logic must match rollout logic in pipeline.execute()
        # In rollout: if entity, temporal, and topic all empty, use filtered_memory_emb
        # Here: if aggregated_memory_emb missing, check if all three are zero, then use filtered_memory_emb
        aggregated_memory_embs = []
        for exp in self.experiences:
            entity_emb = get_emb(exp, 'entity_emb')
            temporal_emb = get_emb(exp, 'temporal_emb')
            topic_emb = get_emb(exp, 'topic_emb')
            filtered_memory_emb = get_emb(exp, 'filtered_memory_emb')

            # If aggregated_memory_emb exists, use it directly
            if hasattr(exp, 'aggregated_memory_emb') and exp.aggregated_memory_emb is not None:
                aggregated_memory_embs.append(exp.aggregated_memory_emb)
            else:
                # Fallback: reconstruct aggregated_memory_emb
                # Check if entity_emb, temporal_emb, and topic_emb are all zero vectors (empty relations)
                # NOTE: This checks the embeddings, not the original relations lists
                # The rollout logic checks len(relations) == 0, but here we check if embeddings are zero
                # This handles the case where relations exist but embedding conversion failed
                entity_is_zero = torch.allclose(entity_emb, torch.zeros_like(entity_emb), atol=1e-6)
                temporal_is_zero = torch.allclose(temporal_emb, torch.zeros_like(temporal_emb), atol=1e-6)
                topic_is_zero = torch.allclose(topic_emb, torch.zeros_like(topic_emb), atol=1e-6)

                if entity_is_zero and temporal_is_zero and topic_is_zero:
                    # All empty: use filtered_memory_emb (matches rollout logic in pipeline.execute())
                    aggregated_memory_embs.append(filtered_memory_emb)
                else:
                    # At least one has content: average pool entity, temporal, and topic embeddings
                    valid_embs = []
                    if not entity_is_zero:
                        valid_embs.append(entity_emb)
                    if not temporal_is_zero:
                        valid_embs.append(temporal_emb)
                    if not topic_is_zero:
                        valid_embs.append(topic_emb)

                    if valid_embs:
                        aggregated = torch.mean(torch.stack(valid_embs), dim=0)
                        aggregated_memory_embs.append(aggregated)
                    else:
                        aggregated_memory_embs.append(filtered_memory_emb)
        
        batch = {
            'query_embs': torch.stack([exp.query_emb for exp in self.experiences]).to(device),
            'memory_embs': torch.stack([exp.memory_embs for exp in self.experiences]).to(device),
            # Intermediate embeddings
            # CRITICAL: These must match exactly what was used during action selection
            'initial_memory_embs': torch.stack([get_emb(exp, 'initial_memory_emb') for exp in self.experiences]).to(device),
            'filtered_memory_embs': torch.stack([get_emb(exp, 'filtered_memory_emb') for exp in self.experiences]).to(device),
            'aggregated_memory_embs': torch.stack(aggregated_memory_embs).to(device),  # CRITICAL: for Module4
            # Actions and log_probs
            'actions': [
                m1_actions.to(device), m2_actions.to(device),
                m3_actions.to(device), m5_actions.to(device), m4_actions.to(device)
            ],
            'old_log_probs': [
                m1_log_probs.to(device), m2_log_probs.to(device),
                m3_log_probs.to(device), m5_log_probs.to(device), m4_log_probs.to(device)
            ],
            'returns': returns.to(device),
            'advantages': advantages.to(device),
            'values': torch.tensor([exp.value for exp in self.experiences], dtype=torch.float32).to(device),
            'costs': torch.tensor([exp.cost for exp in self.experiences], dtype=torch.float32).to(device),
        }

        return batch


# ============================================================================
# PPO Trainer - 5 modules version
# ============================================================================

class PPOTrainer:
    """
    PPO Trainer - supports 5 modules

    Features:
    1. Supports 5-module action selection
    2. Supports cost reward penalty
    3. Adds entropy regularization
    4. Correct advantage function computation
    """

    def __init__(
        self,
        actor_critic: ActorCriticNetwork,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        reward_weight: float = 1.0,  # Reward weight (default 1.0, no scaling)
        cost_weight: float = 0.0,  # Cost penalty weight (0 means no cost penalty)
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=1e-5)
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.reward_weight = reward_weight
        self.cost_weight = cost_weight
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, buffer: ExperienceBuffer) -> Dict[str, float]:
        """
        Update policy using PPO

        Args:
            buffer: Experience buffer

        Returns:
            Training metrics dictionary
        """
        if len(buffer) == 0:
            return {}

        # Get batch data
        batch = buffer.get_batch_tensors(self.device, reward_weight=self.reward_weight, cost_weight=self.cost_weight)

        # Record metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        total_grad_norm = 0
        num_updates = 0
        approx_kl = 0

        self.actor_critic.train()

        # PPO multiple rounds of updates
        for _ in range(self.ppo_epochs):
            # Recalculate current policy's log probabilities and values (using intermediate embeddings)
            # CRITICAL: Must pass ALL embeddings that were used during action selection
            # CRITICAL: Must pass actions from rollout to prevent re-sampling
            _, new_log_probs, entropy, new_values = (
                self.actor_critic.get_action_and_value(
                    batch["query_embs"],
                    batch["initial_memory_embs"],
                    batch["filtered_memory_embs"],
                    batch["aggregated_memory_embs"],  # CRITICAL: aggregated for Module4
                    actions=batch["actions"],  # CRITICAL: Use actions from rollout, not re-sample
                )
            )

            # Compute joint action ratio using sum of log probabilities
            # Use Σ logπ_i: ratio = exp((Σ new_logprob_i) - (Σ old_logprob_i))
            # Do NOT compute ratio separately for each of the 5 modules
            new_total_log_prob = sum(new_log_probs)  # new_log_probs is a list of tensors [m1, m2, m3, m5, m4]
            old_total_log_prob = sum(batch["old_log_probs"])  # old_log_probs is a list of tensors [m1, m2, m3, m5, m4]

            # Ensure is tensor (prevent being Python scalar)
            if not isinstance(new_total_log_prob, torch.Tensor):
                new_total_log_prob = torch.tensor(new_total_log_prob, device=self.device)
            if not isinstance(old_total_log_prob, torch.Tensor):
                old_total_log_prob = torch.tensor(old_total_log_prob, device=self.device)

            # Compute joint ratio: exp(Σ new_logprob - Σ old_logprob)
            log_ratio = new_total_log_prob - old_total_log_prob
            log_ratio_clamped = log_ratio.clamp(-10, 10)  # Prevent numerical overflow
            ratio = torch.exp(log_ratio_clamped)

            # Compute approximate KL divergence (for monitoring)
            # CRITICAL: Use clamped log_ratio to be consistent with ratio calculation
            # This ensures monitoring value reflects the actual KL used in loss computation
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio_clamped).mean().item()
                # Clip fraction: proportion of samples being clipped
                clip_fraction = ((ratio < (1 - self.clip_epsilon)) | (ratio > (1 + self.clip_epsilon))).float().mean().item()

            # PPO Clipped Objective
            advantages = batch["advantages"]
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (add clipping for stability)
            # Standard PPO implementation: take max for each sample first, then average (per-sample clipping)
            value_pred_clipped = batch["values"] + torch.clamp(
                new_values - batch["values"],
                -self.clip_epsilon,
                self.clip_epsilon
            )
            # Compute mse per sample, then take max, then average
            value_loss1 = (new_values - batch["returns"]) ** 2
            value_loss2 = (value_pred_clipped - batch["returns"]) ** 2
            value_loss = torch.max(value_loss1, value_loss2).mean()

            # Entropy loss (encourage exploration)
            entropy_loss = entropy.mean()

            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
            total_clip_fraction += clip_fraction
            total_grad_norm += grad_norm.item()
            num_updates += 1

            # Early stopping if KL divergence too large
            # if approx_kl > 0.02:
            #     print(f"Early stopping at epoch due to high KL: {approx_kl:.4f}")
            #     break

        # Compute average cost
        avg_cost = batch["costs"].mean().item()

        # Compute explained variance (important metric for value function quality)
        # Use updated network to recompute values to reflect current network performance
        with torch.no_grad():
            _, _, _, final_new_values = (
                self.actor_critic.get_action_and_value(
                    batch["query_embs"],
                    batch["initial_memory_embs"],
                    batch["filtered_memory_embs"],
                    batch["aggregated_memory_embs"],  # CRITICAL: aggregated for Module4
                    actions=batch["actions"],
                )
            )
            y_pred = final_new_values  # Use updated network's values
            y_true = batch["returns"]
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
            explained_var = explained_var.item()

        # Return average metrics
        metrics = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": approx_kl,
            "clip_fraction": total_clip_fraction / num_updates,
            "grad_norm": total_grad_norm / num_updates,
            "explained_variance": explained_var,
            "num_updates": num_updates,
            "avg_cost": avg_cost,
        }

        return metrics


# ============================================================================
# Legacy Support - Compatible with old 2-module API
# ============================================================================

