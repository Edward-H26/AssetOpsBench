"""A plain chat endpoint for JSON tool steps, without a model-specific parser."""

from openai import AsyncOpenAI


class ChatModel:
    """Send each request once; the caller supplies the model and endpoint."""

    def __init__(self, model, base_url, api_key, temperature, *, supports_images=False):
        self.model, self.temperature = model, temperature
        self.supports_images = supports_images
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=None)

    async def __call__(self, messages, *, response_format=None):
        extra = {"response_format": response_format} if response_format is not None else {}
        response = await self.client.chat.completions.create(model=self.model, messages=messages, temperature=self.temperature, **extra)
        choice = response.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None}

    async def close(self):
        await self.client.close()
