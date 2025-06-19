"""
Custom exceptions for the Egile Researcher package.

These exceptions provide specific error handling for research automation
and Azure OpenAI integration scenarios.
"""

from typing import Optional, Dict, Any


class EgileResearcherError(Exception):
    """Base exception for all Egile Researcher errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AzureOpenAIError(EgileResearcherError):
    """Exception for Azure OpenAI API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(EgileResearcherError):
    """Exception for authentication failures."""

    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.auth_method = auth_method


class RetryableError(EgileResearcherError):
    """Exception for errors that can be retried."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ResearchAgentError(EgileResearcherError):
    """Exception for research agent operation errors."""

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.agent_name = agent_name
        self.operation = operation


class PaperFetchError(EgileResearcherError):
    """Exception for paper fetching failures."""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.source = source
        self.query = query


class ContentProcessingError(EgileResearcherError):
    """Exception for content processing failures."""

    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        paper_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.content_type = content_type
        self.paper_id = paper_id


class SummarizationError(EgileResearcherError):
    """Exception for paper summarization failures."""

    def __init__(
        self,
        message: str,
        paper_title: Optional[str] = None,
        summary_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.paper_title = paper_title
        self.summary_type = summary_type


class TrendAnalysisError(EgileResearcherError):
    """Exception for trend analysis failures."""

    def __init__(
        self,
        message: str,
        topic: Optional[str] = None,
        time_period: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.topic = topic
        self.time_period = time_period


class ExportError(EgileResearcherError):
    """Exception for export operation failures."""

    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.export_format = export_format
        self.file_path = file_path
