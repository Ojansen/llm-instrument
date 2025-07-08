from pydantic_ai import Agent


class Inference():
    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def run(self, prompt: str) -> str:
        response = self.agent.run_sync(prompt)
        return response.output
