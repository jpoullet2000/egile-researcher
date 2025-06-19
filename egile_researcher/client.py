"""
Azure OpenAI Client for research automation with managed identity support.

This client implements research-specific features on top of Azure OpenAI:
- Managed Identity authentication (preferred)
- Service Principal fallback for CI/CD
- Comprehensive error handling with retry logic
- Research-specific prompts and templates
- Paper summarization and analysis
- Research trend identification
"""

import os
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime, timedelta
import asyncio

from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    ClientSecretCredential,
)
from azure.keyvault.secrets import SecretClient
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import structlog
import httpx

from .exceptions import AzureOpenAIError, AuthenticationError, RetryableError
from .config import AzureOpenAIConfig


logger = structlog.get_logger(__name__)


class AzureOpenAIClient:
    """
    Azure OpenAI client optimized for research automation.

    Features:
    - Managed Identity authentication (Azure-hosted environments)
    - Service Principal authentication (CI/CD pipelines)
    - Key Vault integration for secure credential storage
    - Automatic retry with exponential backoff
    - Request/response logging and monitoring
    - Research-specific prompt templates
    - Paper summarization and analysis
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        max_retries: int = 3,
        timeout: int = 30,
        use_managed_identity: bool = True,
    ):
        """
        Initialize Azure OpenAI client for research automation.

        Args:
            config: Optional configuration object
            endpoint: Azure OpenAI endpoint URL
            api_version: API version to use
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            use_managed_identity: Whether to use managed identity for auth
        """
        self.config = config or AzureOpenAIConfig.from_environment()
        self.endpoint = endpoint or self.config.endpoint
        self.api_version = api_version
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_managed_identity = use_managed_identity

        self._client: Optional[AsyncAzureOpenAI] = None
        self._credential = None
        self._last_token_refresh = None
        self._token_cache = None

        logger.info(
            "Initializing Azure OpenAI client for research automation",
            endpoint=self.endpoint,
            api_version=self.api_version,
            use_managed_identity=self.use_managed_identity,
        )

    async def _get_credential(self):
        """Get Azure credential with fallback strategy."""
        if self._credential is None:
            try:
                if self.use_managed_identity:
                    # Preferred: Use Managed Identity in Azure-hosted environments
                    self._credential = ManagedIdentityCredential()
                    logger.info("Using Managed Identity for authentication")
                else:
                    # Fallback: Use Default Azure Credential chain
                    self._credential = DefaultAzureCredential()
                    logger.info("Using Default Azure Credential for authentication")

            except Exception as e:
                logger.error("Failed to initialize Azure credential", error=str(e))
                raise AuthenticationError(
                    f"Failed to initialize Azure credential: {e}",
                    auth_method="managed_identity"
                    if self.use_managed_identity
                    else "default",
                )

        return self._credential

    async def _get_api_key_from_keyvault(self) -> Optional[str]:
        """Retrieve API key from Azure Key Vault if configured."""
        if not self.config.key_vault_url or not self.config.api_key_secret_name:
            return None

        try:
            credential = await self._get_credential()
            secret_client = SecretClient(
                vault_url=self.config.key_vault_url, credential=credential
            )

            secret = secret_client.get_secret(self.config.api_key_secret_name)
            logger.info("Retrieved API key from Key Vault")
            return secret.value

        except Exception as e:
            logger.error("Failed to retrieve API key from Key Vault", error=str(e))
            raise AuthenticationError(f"Failed to retrieve API key from Key Vault: {e}")

    async def _initialize_client(self):
        """Initialize the OpenAI client with proper authentication."""
        if self._client is not None:
            return self._client

        try:
            api_key = (
                os.getenv("AZURE_OPENAI_API_KEY") or self.config.api_key_secret_name
            )
            # Try to get API key from Key Vault first
            if not api_key:
                api_key = await self._get_api_key_from_keyvault()
            if api_key:
                # Use API key authentication
                self._client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    timeout=httpx.Timeout(self.timeout),
                    max_retries=self.max_retries,
                )
                logger.info("Initialized client with API key authentication")
            else:
                # Use Azure AD token authentication
                credential = await self._get_credential()
                token = credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )

                self._client = AsyncAzureOpenAI(
                    azure_ad_token=token.token,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    timeout=httpx.Timeout(self.timeout),
                    max_retries=self.max_retries,
                )

                self._token_cache = token
                self._last_token_refresh = datetime.now()
                logger.info("Initialized client with Azure AD token authentication")

            return self._client

        except Exception as e:
            logger.error("Failed to initialize Azure OpenAI client", error=str(e))
            raise AzureOpenAIError(f"Failed to initialize client: {e}")

    async def _refresh_token_if_needed(self):
        """Refresh Azure AD token if it's close to expiration."""
        if not self._token_cache or not self._last_token_refresh:
            return

        # Refresh token if it expires within 5 minutes
        if (datetime.now() - self._last_token_refresh) > timedelta(minutes=55):
            try:
                credential = await self._get_credential()
                token = credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )

                # Update client with new token
                if hasattr(self._client, "_azure_ad_token"):
                    self._client._azure_ad_token = token.token

                self._token_cache = token
                self._last_token_refresh = datetime.now()
                logger.info("Refreshed Azure AD token")

            except Exception as e:
                logger.warning("Failed to refresh Azure AD token", error=str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RetryableError),
    )
    async def chat_completion(
        self,
        messages: List[ChatCompletionMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Create a chat completion with retry logic and error handling.

        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Chat completion response

        Raises:
            AzureOpenAIError: For API errors
            RetryableError: For retryable errors
        """
        # Use default model from config if none specified
        if model is None:
            model = self.config.default_model

        client = await self._initialize_client()
        await self._refresh_token_if_needed()

        request_start = datetime.now()
        request_id = f"req_{int(request_start.timestamp())}"

        logger.info(
            "Starting chat completion request",
            request_id=request_id,
            model=model,
            message_count=len(messages),
            temperature=temperature,
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            duration = (datetime.now() - request_start).total_seconds()

            logger.info(
                "Chat completion request completed",
                request_id=request_id,
                duration_seconds=duration,
                usage=response.usage.model_dump() if response.usage else None,
            )

            return response

        except Exception as e:
            duration = (datetime.now() - request_start).total_seconds()

            logger.error(
                "Chat completion request failed",
                request_id=request_id,
                duration_seconds=duration,
                error=str(e),
            )

            # Determine if error is retryable
            if hasattr(e, "status_code"):
                status_code = e.status_code
                if status_code in [429, 500, 502, 503, 504]:
                    raise RetryableError(
                        f"Retryable error {status_code}: {e}",
                        retry_after=getattr(e, "retry_after", None),
                    )
                else:
                    raise AzureOpenAIError(
                        f"API error {status_code}: {e}",
                        status_code=status_code,
                        request_id=request_id,
                    )
            else:
                raise AzureOpenAIError(f"Unexpected error: {e}", request_id=request_id)

    async def summarize_paper(
        self,
        paper_content: str,
        paper_metadata: Dict[str, Any],
        summary_type: str = "comprehensive",
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Summarize a research paper using optimized prompts.

        Args:
            paper_content: Full text content of the paper
            paper_metadata: Paper metadata (title, authors, etc.)
            summary_type: Type of summary (brief, comprehensive, technical)
            model: Model to use for summarization

        Returns:
            Generated paper summary
        """
        # Use default model from config if none specified
        if model is None:
            model = self.config.default_model

        system_prompt = f"""You are an expert research analyst specializing in academic paper summarization. 
Generate a {summary_type} summary that extracts key information in a structured format:

For {summary_type} summaries, include:
- Main research question and objectives
- Methodology and approach
- Key findings and results
- Significance and implications
- Limitations (if applicable)

Maintain academic rigor while making the content accessible."""

        user_prompt = f"""Please summarize this research paper:

Title: {paper_metadata.get("title", "Unknown")}
Authors: {paper_metadata.get("authors", "Unknown")}
Publication: {paper_metadata.get("journal", paper_metadata.get("venue", "Unknown"))}

Content:
{paper_content}

Generate a {summary_type} summary following the structured format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.chat_completion(
            messages=messages, model=model, temperature=0.3, **kwargs
        )

        return response.choices[0].message.content

    async def analyze_research_trends(
        self,
        papers_data: List[Dict[str, Any]],
        topic_focus: str,
        time_period: str = "recent",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze research trends from a collection of papers.

        Args:
            papers_data: List of papers with metadata and summaries
            topic_focus: Main research area or topic to focus on
            time_period: Time period for analysis
            model: Model to use for analysis

        Returns:
            Analysis results with trends, patterns, and insights
        """
        # Use default model from config if none specified
        if model is None:
            model = self.config.default_model

        system_prompt = """You are a research trend analyst with expertise in identifying patterns, 
emerging themes, and research directions from academic literature. Analyze the provided papers 
and identify key trends, methodological patterns, and research gaps."""

        papers_summary = "\n\n".join(
            [
                f"Paper {i + 1}: {paper.get('title', 'Unknown')}\n"
                f"Year: {paper.get('year', 'Unknown')}\n"
                f"Summary: {paper.get('summary', 'No summary available')}"
                for i, paper in enumerate(
                    papers_data[:20]
                )  # Limit to avoid token limits
            ]
        )

        user_prompt = f"""Analyze these {len(papers_data)} research papers on {topic_focus} from {time_period}:

{papers_summary}

Please provide:
1. Major research trends and themes
2. Methodological approaches being used
3. Emerging research directions
4. Key research gaps or opportunities
5. Notable patterns in findings or conclusions

Format your analysis in a structured way with clear sections."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.chat_completion(
            messages=messages, model=model, temperature=0.5
        )

        return {
            "analysis": response.choices[0].message.content,
            "papers_analyzed": len(papers_data),
            "topic_focus": topic_focus,
            "time_period": time_period,
            "analyzed_at": datetime.now().isoformat(),
        }

    async def compare_papers(
        self,
        papers: List[Dict[str, Any]],
        comparison_aspects: List[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple research papers across different aspects.

        Args:
            papers: List of papers to compare
            comparison_aspects: Specific aspects to compare
            model: Model to use for comparison

        Returns:
            Comparative analysis results
        """
        # Use default model from config if none specified
        if model is None:
            model = self.config.default_model

        if comparison_aspects is None:
            comparison_aspects = [
                "methodology",
                "findings",
                "contributions",
                "limitations",
                "future_work",
            ]

        system_prompt = """You are a research analyst specializing in comparative analysis of academic papers. 
Compare the provided papers systematically across the specified aspects, highlighting similarities, 
differences, and relationships between the works."""

        papers_text = "\n\n".join(
            [
                f"Paper {i + 1}:\nTitle: {paper.get('title', 'Unknown')}\n"
                f"Summary: {paper.get('summary', 'No summary available')}"
                for i, paper in enumerate(papers)
            ]
        )

        user_prompt = f"""Compare these {len(papers)} research papers across these aspects: {", ".join(comparison_aspects)}

{papers_text}

Provide a structured comparison highlighting:
- Similarities and differences
- Complementary findings
- Conflicting results or approaches
- Overall synthesis and insights

Format your comparison clearly with sections for each aspect."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.chat_completion(
            messages=messages, model=model, temperature=0.4
        )

        return {
            "comparison": response.choices[0].message.content,
            "papers_compared": len(papers),
            "comparison_aspects": comparison_aspects,
            "compared_at": datetime.now().isoformat(),
        }

    async def close(self):
        """Clean up resources."""
        if self._client:
            await self._client.close()
        logger.info("Azure OpenAI client closed")
