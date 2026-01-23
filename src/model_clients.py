import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
from mistralai import Mistral
from openai import OpenAI


@dataclass
class ModelResponse:
    """Standardized response from any model"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any = None


class ModelClient:
    """Base class for all model clients"""

    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.provider = model_config['provider']
        self.model_name = model_config['model_name']
        self.max_tokens = model_config.get('max_tokens', 4000)
        self.temperature = model_config.get('temperature', 0.7)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_async(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        """Async generation - to be implemented by subclasses"""
        raise NotImplementedError

    def generate_sync(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        """Sync generation wrapper"""
        return asyncio.run(self.generate_async(prompt, system_prompt))


class GroqClient(ModelClient):
    """Client for Groq models"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        # Groq uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    async def generate_async(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                raw_response=response
            )
        except Exception as e:
            print(f"Groq API error: {e}")
            raise


class MistralClient(ModelClient):
    """Client for Mistral models"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in environment")

        self.client = Mistral(api_key=api_key)

    async def generate_async(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.to_thread(
                self.client.chat.complete,
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                raw_response=response
            )
        except Exception as e:
            print(f"Mistral API error: {e}")
            raise


# REMOVE or COMMENT OUT these classes since we're not using them:
# class OpenRouterClient(ModelClient):
# class GoogleClient(ModelClient):


class ModelFactory:
    """Factory to create model clients"""

    @staticmethod
    def create_client(model_key: str, config: Dict[str, Any]) -> ModelClient:
        """Create appropriate client based on model key"""
        model_config = config['models'][model_key]

        if model_config['provider'] == 'groq':
            return GroqClient(model_config)
        elif model_config['provider'] == 'mistral':
            return MistralClient(model_config)
        else:
            raise ValueError(f"Unknown provider: {model_config['provider']}")


# Async batch processing
class BatchProcessor:
    """Process multiple LLM calls in parallel"""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(self, tasks: List[tuple]) -> List[Any]:
        """Process batch of tasks with rate limiting"""

        async def process_with_semaphore(task):
            async with self.semaphore:
                client, prompt, system_prompt = task
                return await client.generate_async(prompt, system_prompt)

        return await asyncio.gather(*[
            process_with_semaphore(task) for task in tasks
        ])


# Cache layer to save API calls
import hashlib
import pickle
from pathlib import Path


class ResponseCache:
    """Simple disk cache for LLM responses"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, model: str, prompt: str, system_prompt: str = "") -> str:
        """Generate cache key from request parameters"""
        content = f"{model}:{system_prompt}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, model: str, prompt: str, system_prompt: str = "") -> Optional[ModelResponse]:
        """Get cached response if exists"""
        cache_key = self._get_cache_key(model, prompt, system_prompt)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def set(self, model: str, prompt: str, system_prompt: str, response: ModelResponse):
        """Cache a response"""
        cache_key = self._get_cache_key(model, prompt, system_prompt)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)