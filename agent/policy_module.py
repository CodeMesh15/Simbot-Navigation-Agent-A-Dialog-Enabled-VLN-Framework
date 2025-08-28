import torch
import torch.nn as nn

class PolicyModule(nn.Module):
    """
    The core decision-making module (the "brain") of the agent.
    It's a recurrent model that decides the next action to take.
    """
    def __init__(self, vision_dim, lang_dim, hidden_dim, num_actions, dropout=0.5):
        super(PolicyModule, self).__init__()
        
        # Dimensions
        self.vision_dim = vision_dim
        self.lang_dim = lang_dim
        self.hidden_dim = hidden_dim
        
        # LSTMCell to maintain the agent's state/memory over time
        # The input to the LSTM will be the combined vision and language features
        self.lstm_cell = nn.LSTMCell(vision_dim + lang_dim, hidden_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # The final layer that predicts the action
        self.action_predictor = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, vision_features, lang_features, hidden_state, cell_state):
        """
        Performs a single step of the policy.
        
        Args:
            vision_features (torch.Tensor): Features from the current visual input.
            lang_features (torch.Tensor): Embedding of the full instruction.
            hidden_state (torch.Tensor): The previous hidden state of the LSTM.
            cell_state (torch.Tensor): The previous cell state of the LSTM.
            
        Returns:
            tuple: (action_logits, new_hidden_state, new_cell_state)
        """
        # Combine the vision and language features
        combined_features = torch.cat((vision_features, lang_features), dim=1)
        
        # Update the LSTM state
        new_hidden, new_cell = self.lstm_cell(combined_features, (hidden_state, cell_state))
        
        # Apply dropout for regularization
        new_hidden_dropped = self.dropout(new_hidden)
        
        # Predict the action logits from the new hidden state
        action_logits = self.action_predictor(new_hidden_dropped)
        
        return action_logits, new_hidden, new_cell

if __name__ == '__main__':
    # --- Example Usage ---
    # Define some dimensions for the agent
    VISION_DIM = 2048 # From ResNet-50
    LANG_DIM = 768   # From BERT
    HIDDEN_DIM = 512
    NUM_ACTIONS = 4  # e.g., move_forward, turn_left, turn_right, stop
    BATCH_SIZE = 1   # Usually 1 during inference

    policy_agent = PolicyModule(VISION_DIM, LANG_DIM, HIDDEN_DIM, NUM_ACTIONS)
    
    # Create dummy inputs
    dummy_vision = torch.randn(BATCH_SIZE, VISION_DIM)
    dummy_lang = torch.randn(BATCH_SIZE, LANG_DIM)
    dummy_hidden = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
    dummy_cell = torch.zeros(BATCH_SIZE, HIDDEN_DIM)

    print("Performing one step with the policy agent...")
    logits, new_h, new_c = policy_agent(dummy_vision, dummy_lang, dummy_hidden, dummy_cell)
    
    print(f"Policy Module initialized successfully.")
    print(f"Output action logits shape: {logits.shape}")
