import re
from typing import List, Tuple, Optional

class PromptTemplate:
    """
    A utility class to build structured chat prompts using the <|im_start|> and
    <|im_end|> token format (e.g., Qwen/Llama 3 style). It manages conversation
    turn-taking with validation.
    """
    def __init__(self, system_prompt=None):
        """
        Initializes the template with an optional system prompt and empty message lists.
        """
        self.system_prompt = system_prompt
        self.user_messages = []
        self.model_replies = []

    def __str__(self):
        """
        Returns the fully built prompt string when the object is converted to string.
        """
        return self.build_prompt()

    def add_user_message(self, message: str, return_prompt=True):
        """
        Adds a user message and validates that it starts a new turn.
        
        Args:
            message: The user's input message.
            return_prompt: If True, returns the complete, updated prompt string.
        """
        self.user_messages.append(message)
        if len(self.user_messages) != len(self.model_replies) + 1:
            raise ValueError(
                "Error: Expected len(user_messages) = len(model_replies) + 1. Add a new user message!"
            )
        if return_prompt:
            return self.build_prompt()

    def add_model_reply(self, reply: str, includes_history=True, return_reply=False):
        """
        Adds a model's reply, validating that the turn is complete.
        
        Args:
            reply: The model's raw reply string.
            includes_history: If True, strips the generated prompt history from the reply.
            return_reply: If True, returns the processed reply (without history).
        """
        reply_ = reply.replace(self.build_prompt(), "") if includes_history else reply
        self.model_replies.append(reply_)
        if len(self.user_messages) != len(self.model_replies):
            raise ValueError(
                "Number of user messages does not equal number of system replies."
            )
        if return_reply:
            return reply_

    def get_user_messages(self, strip=True) -> List[str]:
        """Retrieves the list of user messages."""
        return [x.strip() for x in self.user_messages] if strip else self.user_messages

    def get_model_replies(self, strip=True) -> List[str]:
        """Retrieves the list of model replies."""
        return [x.strip() for x in self.model_replies] if strip else self.model_replies

    def build_prompt(self) -> str:
        """
        Constructs the final chat prompt string in the required format.
        The prompt is always structured to wait for the next assistant reply.
        """
        # Ensure the conversation state is valid for generating a new prompt
        if len(self.user_messages) != len(self.model_replies) + 1:
            raise ValueError(
                "Error: Expected len(user_messages) = len(model_replies) + 1. Add a new user message!"
            )

        # 1. System Prompt (Optional)
        if self.system_prompt is not None:
            SYS = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        else:
            SYS = ""
        
        # 2. History of full turns (User -> Assistant)
        CONVO = ""
        for i in range(len(self.model_replies)):
            user_message, model_reply = self.user_messages[i], self.model_replies[i]
            conversation_ = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{model_reply}<|im_end|>\n"
            CONVO += conversation_
        
        # 3. Current turn (User prompt waiting for Assistant reply)
        CONVO += f"<|im_start|>user\n{self.user_messages[-1]}<|im_end|>\n<|im_start|>assistant\n"
        
        return SYS + CONVO

if __name__ == '__main__':
    # Example usage:
    initial_prompt = """Classify the given log entries into normal and abnormal categories.
scsi(0): LIP reset occurred."""
    
    # Instantiate with system prompt
    pt = PromptTemplate(system_prompt="You are a helpful and professional log analysis assistant.")
    
    # Start the conversation turn
    full_prompt_for_model = pt.add_user_message(initial_prompt)
    
    print("--- Built Prompt ---")
    print(full_prompt_for_model)

    # Simulate model response
    model_raw_output = full_prompt_for_model + "<think>I will classify this as a normal system event.</think><answer>Normal</answer>"
    
    # Add model's reply (stripping the history)
    model_reply = pt.add_model_reply(model_raw_output, includes_history=True, return_reply=True)
    
    print("\n--- Processed Model Reply ---")
    print(model_reply)
    
    # Add a follow-up message
    follow_up_prompt = "What about 'Error reading disk'?"
    pt.add_user_message(follow_up_prompt, return_prompt=False)
    
    print("\n--- Updated Conversation Prompt ---")
    print(pt.build_prompt())
