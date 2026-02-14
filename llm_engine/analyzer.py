"""
AceML Studio – LLM Analyzer
==============================
Use an LLM (OpenAI / Azure OpenAI / Anthropic) to analyze data-quality
reports, suggest feature engineering, explain model results, and recommend
next steps.  Reads API keys from config.py.
"""

import json
import time
import logging
from config import Config

logger = logging.getLogger("aceml.llm_analyzer")


class LLMAnalyzer:
    """AI-powered analysis powered by configurable LLM backend."""

    SYSTEM_PROMPT = (
        "You are AceML Studio AI Assistant — an expert machine-learning engineer. "
        "You analyse datasets, suggest data cleaning strategies, recommend feature engineering, "
        "explain model evaluation results, and help users build production-ready ML pipelines. "
        "Be concise, specific, and actionable. Use bullet points. "
        "When suggesting code, use Python with pandas/scikit-learn idioms."
    )

    # ------------------------------------------------------------------ #
    #  Public helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def analyze_data_quality(cls, quality_report: dict, data_summary: dict) -> str:
        logger.info("LLM: analyzing data quality (report keys=%s)", list(quality_report.keys()))
        prompt = (
            "Analyze the following data-quality report and provide:\n"
            "1. A summary of the key issues found\n"
            "2. Prioritized recommendations for data cleaning\n"
            "3. Suggested feature engineering opportunities\n"
            "4. Any warnings about potential modeling pitfalls\n\n"
            f"**Dataset Summary:**\n```json\n{json.dumps(data_summary, indent=2, default=str)}\n```\n\n"
            f"**Quality Report:**\n```json\n{json.dumps(quality_report, indent=2, default=str)}\n```"
        )
        return cls._call_llm(prompt)

    @classmethod
    def suggest_cleaning(cls, issues: list, column_info: dict) -> str:
        logger.info("LLM: suggesting cleaning strategies for %d issues", len(issues))
        prompt = (
            "Based on the data issues below, suggest specific cleaning steps. "
            "For each issue, recommend the best strategy and explain why.\n\n"
            f"**Issues:**\n```json\n{json.dumps(issues, indent=2, default=str)}\n```\n\n"
            f"**Column Info:**\n```json\n{json.dumps(column_info, indent=2, default=str)}\n```"
        )
        return cls._call_llm(prompt)

    @classmethod
    def suggest_features(cls, column_info: dict, target: str | None = None) -> str:
        logger.info("LLM: suggesting features (target=%s, cols=%d)",
                    target, len(column_info.get('column_names', [])))
        prompt = (
            "Suggest feature engineering ideas for the dataset described below. "
            "Include date features, interaction terms, transformations, and new derived features.\n\n"
            f"**Columns:**\n```json\n{json.dumps(column_info, indent=2, default=str)}\n```\n"
            f"**Target variable:** {target or 'Not specified'}\n"
        )
        return cls._call_llm(prompt)

    @classmethod
    def explain_evaluation(cls, eval_results: dict, task: str) -> str:
        logger.info("LLM: explaining evaluation results (task=%s)", task)
        prompt = (
            f"Explain the following {task} model evaluation results in plain language. "
            "Highlight strengths, weaknesses, signs of overfitting/underfitting, "
            "and actionable next steps to improve performance.\n\n"
            f"**Results:**\n```json\n{json.dumps(eval_results, indent=2, default=str)}\n```"
        )
        return cls._call_llm(prompt)

    @classmethod
    def suggest_tuning(cls, model_key: str, current_params: dict, current_metrics: dict) -> str:
        logger.info("LLM: suggesting tuning for model='%s'", model_key)
        prompt = (
            f"The model '{model_key}' was trained with the following parameters and achieved these metrics. "
            "Suggest hyperparameter tuning strategies and specific parameter ranges to try.\n\n"
            f"**Current Params:**\n```json\n{json.dumps(current_params, indent=2, default=str)}\n```\n\n"
            f"**Current Metrics:**\n```json\n{json.dumps(current_metrics, indent=2, default=str)}\n```"
        )
        return cls._call_llm(prompt)

    @classmethod
    def general_question(cls, question: str, context: dict | None = None) -> str:
        logger.info("LLM: general question (len=%d, has_context=%s)", len(question), bool(context))
        prompt = question
        if context:
            prompt += f"\n\n**Context:**\n```json\n{json.dumps(context, indent=2, default=str)}\n```"
        return cls._call_llm(prompt)

    # ------------------------------------------------------------------ #
    #  Multi-turn Chat with Rich Context
    # ------------------------------------------------------------------ #
    CHAT_SYSTEM_PROMPT = (
        "You are AceML Studio Chat Assistant — a friendly, expert ML guide "
        "designed for business users with minimal technical knowledge.\n\n"
        "RULES:\n"
        "• Explain everything in simple, plain language.  Avoid jargon unless the user uses it first.\n"
        "• When recommending actions, map them to the exact UI buttons/sections in AceML Studio "
        "(Upload Data, Data Quality, Data Cleaning, Feature Engineering, Transformations, "
        "Reduce Dimensions, Train Models, Evaluation, Tuning, Experiments, AI Insights).\n"
        "• Use bullet points and short paragraphs for readability.\n"
        "• If context about the user's dataset, logs, tuning parameters, or evaluation results "
        "is provided below, reference it specifically to give personalised advice.\n"
        "• Always suggest a clear next step the user can take.\n"
    )

    @classmethod
    def chat(cls, messages: list[dict], context: dict | None = None) -> str:
        """
        Multi-turn chat.  *messages* is a list of {"role": "user"|"assistant", "content": str}.
        *context* is an optional dict with keys like data_summary, recent_logs,
        tuning_params, evaluation_results, user_context.
        """
        logger.info(
            "LLM chat: %d messages, context_keys=%s",
            len(messages),
            list(context.keys()) if context else [],
        )

        # Build a system message enriched with whatever context the caller included.
        system = cls.CHAT_SYSTEM_PROMPT
        if context:
            system += "\n--- SESSION CONTEXT ---\n"
            for key, value in context.items():
                if value:
                    system += f"\n**{key.replace('_', ' ').title()}:**\n"
                    if isinstance(value, (dict, list)):
                        system += f"```json\n{json.dumps(value, indent=2, default=str)}\n```\n"
                    else:
                        system += f"{value}\n"

        provider = Config.LLM_PROVIDER.lower()
        logger.debug("Chat provider=%s, system_len=%d", provider, len(system))

        try:
            t0 = time.time()
            if provider == "anthropic":
                result = cls._chat_anthropic(system, messages)
            else:
                # OpenAI-compatible path (openai, azure_openai, deepseek)
                result = cls._chat_openai_compat(system, messages, provider)
            logger.info("Chat response in %.2fs (provider=%s, len=%d)",
                        time.time() - t0, provider, len(result))
            return result
        except Exception as e:
            logger.error("Chat call failed (provider=%s): %s", provider, e, exc_info=True)
            return f"[Chat Error] {e}\n\nMake sure your API key is set correctly in config.py."

    # ── OpenAI-compatible chat (OpenAI / Azure / DeepSeek) ────────
    @classmethod
    def _chat_openai_compat(cls, system: str, messages: list[dict], provider: str) -> str:
        from openai import OpenAI, AzureOpenAI

        if provider == "azure_openai":
            client = AzureOpenAI(
                api_key=Config.AZURE_OPENAI_KEY,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
            model = Config.AZURE_OPENAI_DEPLOYMENT
        elif provider == "deepseek":
            client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)
            model = Config.DEEPSEEK_MODEL
        else:
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            model = Config.OPENAI_MODEL

        oai_messages = [{"role": "system", "content": system}]  # type: ignore
        for m in messages:
            oai_messages.append({"role": m["role"], "content": m["content"]})  # type: ignore

        response = client.chat.completions.create(
            model=model,
            messages=oai_messages,  # type: ignore
            max_tokens=Config.OPENAI_MAX_TOKENS,
            temperature=Config.OPENAI_TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    # ── Anthropic chat ────────────────────────────────────────────
    @classmethod
    def _chat_anthropic(cls, system: str, messages: list[dict]) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        anth_messages = [{"role": m["role"], "content": m["content"]} for m in messages]  # type: ignore
        response = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=Config.OPENAI_MAX_TOKENS,
            system=system,
            messages=anth_messages,  # type: ignore
        )
        # Extract text from response content
        if hasattr(response.content[0], 'text'):
            return response.content[0].text  # type: ignore
        return str(response.content[0])

    # ------------------------------------------------------------------ #
    #  LLM backend dispatcher
    # ------------------------------------------------------------------ #
    @classmethod
    def _call_llm(cls, user_prompt: str) -> str:
        provider = Config.LLM_PROVIDER.lower()
        logger.debug("LLM call: provider=%s, prompt_len=%d", provider, len(user_prompt))
        try:
            t0 = time.time()
            if provider == "openai":
                result = cls._call_openai(user_prompt)
            elif provider == "azure_openai":
                result = cls._call_azure_openai(user_prompt)
            elif provider == "anthropic":
                result = cls._call_anthropic(user_prompt)
            elif provider == "deepseek":
                result = cls._call_deepseek(user_prompt)
            else:
                logger.error("Unknown LLM provider: %s", provider)
                return f"[LLM Error] Unknown provider: {provider}. Set LLM_PROVIDER in config.py."
            logger.info("LLM response received in %.2fs (provider=%s, response_len=%d)",
                        time.time() - t0, provider, len(result))
            return result
        except Exception as e:
            logger.error("LLM call failed (provider=%s): %s", provider, e, exc_info=True)
            return f"[LLM Error] {e}\n\nMake sure your API key is set correctly in config.py."

    # ── OpenAI ────────────────────────────────────────────────────────
    @classmethod
    def _call_openai(cls, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=Config.OPENAI_MAX_TOKENS,
            temperature=Config.OPENAI_TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    # ── Azure OpenAI ──────────────────────────────────────────────────
    @classmethod
    def _call_azure_openai(cls, prompt: str) -> str:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=Config.OPENAI_MAX_TOKENS,
            temperature=Config.OPENAI_TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    # ── Anthropic ─────────────────────────────────────────────────────
    @classmethod
    def _call_anthropic(cls, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=Config.OPENAI_MAX_TOKENS,
            system=cls.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from response content
        if hasattr(response.content[0], 'text'):
            return response.content[0].text  # type: ignore
        return str(response.content[0])

    # ── DeepSeek ──────────────────────────────────────────────────────
    @classmethod
    def _call_deepseek(cls, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=Config.DEEPSEEK_MAX_TOKENS,
            temperature=Config.DEEPSEEK_TEMPERATURE,
        )
        return response.choices[0].message.content or ""
