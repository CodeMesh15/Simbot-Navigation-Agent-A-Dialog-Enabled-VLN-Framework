
import torch
from transformers import BertTokenizer, BertModel

class LanguageModule:
    """
    Processes natural language instructions using a pre-trained BERT model.
    """
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        # Set the model to evaluation mode and move it to the device
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode_instruction(self, instruction_text):
        """
        Takes a text instruction and returns a sentence embedding.
        
        Args:
            instruction_text (str): The natural language instruction.
            
        Returns:
            torch.Tensor: A sentence embedding vector.
        """
        # Tokenize the instruction
        inputs = self.tokenizer(instruction_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Pass the tokens through the BERT model
        outputs = self.model(**inputs)
        
        # We use the embedding of the [CLS] token as the representation for the whole sentence
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding.squeeze()

if __name__ == '__main__':
    # --- Example Usage ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_processor = LanguageModule(device)
    
    sample_instruction = "Walk forward past the red chair and stop in front of the desk."
    
    print(f"Encoding instruction: '{sample_instruction}'")
    instruction_embedding = language_processor.encode_instruction(sample_instruction)
    
    print(f"Language Module loaded successfully on '{device}'.")
    print(f"Output embedding vector shape: {instruction_embedding.shape}")
