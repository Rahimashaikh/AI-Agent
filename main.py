from dotenv import load_dotenv
import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
load_dotenv()

def main():
    MODEL_NAME = "gemini-2.0-flash"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    external_client = AsyncOpenAI(
        api_key = GEMINI_API_KEY,
        base_url ="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = OpenAIChatCompletionsModel(
        model = "gemini-2.0-flash",
        openai_client = external_client,
        )
    config = RunConfig(
         model = model,
        model_provider = external_client,
        tracing_disabled = True,  
    )
    agent = Agent (
        name = "Assistant",
        instructions = "A helpful assistant in all things.",
        model = model,    
    )
    result = Runner.run_sync(
        agent,
        "how to gain weight tell me in 100 words?",
        run_config = config,
    )
    print(result.final_output)
if __name__ == "__main__":
    main()
