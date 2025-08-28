import torch
import torch.nn.functional as F

class DialogManager:
    """
    A simple rule-based manager to decide when the agent should ask for help.
    """
    def __init__(self, uncertainty_threshold=0.6, patience=3):
        """
        Args:
            uncertainty_threshold (float): Confidence level below which the agent is "uncertain".
            patience (int): Number of consecutive uncertain steps before asking for help.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.patience = patience
        self.uncertain_steps = 0

    def check_for_ambiguity(self, action_logits):
        """
        Checks if the agent's confidence is low for too many steps.
        
        Args:
            action_logits (torch.Tensor): The output logits from the policy module.
            
        Returns:
            tuple: (ask_question (bool), message (str))
        """
        # Calculate confidence by applying softmax to logits
        action_probs = F.softmax(action_logits, dim=1)
        max_confidence = torch.max(action_probs).item()
        
        if max_confidence < self.uncertainty_threshold:
            self.uncertain_steps += 1
            print(f"[Dialog Manager] Low confidence ({max_confidence:.2f}). Uncertain step count: {self.uncertain_steps}")
        else:
            # If confidence is high, reset the counter
            self.uncertain_steps = 0
            
        # If patience has run out, ask for help
        if self.uncertain_steps >= self.patience:
            self.uncertain_steps = 0 # Reset after asking
            clarification_question = "I'm not sure where to go next. Can you provide more specific directions?"
            return True, clarification_question
        
        return False, ""

if __name__ == '__main__':
    # --- Example Usage ---
    dialog_mgr = DialogManager(uncertainty_threshold=0.6, patience=3)
    
    print("--- Simulating agent steps ---")
    
    # Step 1: Confident action
    confident_logits = torch.tensor([[0.1, 3.5, 0.2, 0.1]])
    ask, msg = dialog_mgr.check_for_ambiguity(confident_logits)
    print(f"Confident step -> Ask for help? {ask}\n")
    
    # Step 2-4: Series of uncertain actions
    uncertain_logits = torch.tensor([[0.5, 0.6, 0.4, 0.3]])
    for i in range(1, 4):
        print(f"--- Step {i+1} ---")
        ask, msg = dialog_mgr.check_for_ambiguity(uncertain_logits)
        print(f"Uncertain step -> Ask for help? {ask}")
        if ask:
            print(f"Agent asks: '{msg}'")
        print()
