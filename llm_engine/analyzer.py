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
    def analyze_tuning_error(cls, model_key: str, method: str, error_message: str, 
                            data_info: dict, task: str) -> str:
        """Analyze why hyperparameter tuning failed and suggest fixes."""
        logger.info("LLM: analyzing tuning error for model='%s', method=%s", model_key, method)
        prompt = (
            f"Hyperparameter tuning failed for model '{model_key}' using {method} method. "
            "Analyze the error and provide:\n"
            "1. A clear explanation of what went wrong in plain language\n"
            "2. The likely root cause of the failure\n"
            "3. Step-by-step instructions to fix the issue\n"
            "4. Alternative approaches if the standard fix won't work\n\n"
            f"**Error Message:**\n```\n{error_message}\n```\n\n"
            f"**Model:** {model_key}\n"
            f"**Task:** {task}\n"
            f"**Tuning Method:** {method}\n\n"
            f"**Dataset Info:**\n```json\n{json.dumps(data_info, indent=2, default=str)}\n```\n\n"
            "Be specific about which parameters to adjust, data preprocessing steps needed, "
            "or configuration changes required."
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
    #  Iterative Workflow: Plan, Evaluate Step, Evaluate Iteration
    # ------------------------------------------------------------------ #

    @classmethod
    def plan_workflow_iteration(
        cls,
        data_snapshot: dict,
        quality_metrics: dict,
        target_column: str,
        task_type: str,
        objectives: str,
        iteration_number: int,
        max_iterations: int,
        enabled_steps: list[str],
        history_summary: str = "",
    ) -> dict:
        """
        Ask the LLM to produce a concrete iteration plan.
        Returns a dict with keys: plan_summary, steps (list of step dicts).
        """
        logger.info("LLM: planning workflow iteration %d (target=%s, task=%s)",
                     iteration_number, target_column, task_type)

        prompt = (
            "You are the AceML Studio Workflow Planner.\n"
            "Your job is to plan ONE iteration of data-preparation steps for an ML pipeline.\n\n"
            "RULES:\n"
            "• Return ONLY valid JSON (no markdown fences, no explanation outside JSON).\n"
            "• Each step must use operations compatible with AceML Studio's existing engine.\n"
            "• Be conservative — only include steps that will measurably improve the data.\n"
            "• If the data is already clean and well-prepared, return fewer or no steps.\n"
            "• Respect the enabled_steps list — only use step types from that list.\n\n"
            "AVAILABLE STEP TYPES AND THEIR OPERATIONS:\n"
            "1. data_analysis — no operations needed (runs quality analysis automatically)\n"
            "2. data_cleaning — operations list, each: {action, params}\n"
            "   Actions: drop_missing (params: {columns?, how?}), impute (params: {strategy, columns?}),\n"
            "            drop_duplicates (params: {subset?, keep?}), clip_outliers (params: {columns?, multiplier?}),\n"
            "            remove_outliers (params: {columns?, multiplier?}), drop_columns (params: {columns}),\n"
            "            convert_to_numeric (params: {columns}), convert_to_datetime (params: {columns}),\n"
            "            convert_to_category (params: {columns})\n"
            "3. feature_engineering — operations list, each: {action, params}\n"
            "   Actions: extract_date_features (params: {columns}),\n"
            "            create_interactions (params: {column_pairs}),\n"
            "            create_polynomial (params: {columns, degree?}),\n"
            "            create_bins (params: {column, bins, labels?, new_col_name}),\n"
            "            create_ratio (params: {numerator, denominator, new_col_name}),\n"
            "            create_aggregate (params: {column, group_by, agg_func, new_col_name})\n"
            "4. transformations — operations list, each: {action, params}\n"
            "   Actions: scale (params: {columns, method: standard|minmax|robust}),\n"
            "            one_hot_encode (params: {columns, drop_first?, max_cardinality?}),\n"
            "            label_encode (params: {columns}),\n"
            "            target_encode (params: {columns, target}),\n"
            "            log_transform (params: {columns}),\n"
            "            power_transform (params: {columns})\n"
            "5. dimensionality_reduction — operations list, each: {method, params}\n"
            "   Methods: pca (params: {n_components?}), variance_threshold (params: {threshold?}),\n"
            "            correlation_filter (params: {threshold?}),\n"
            "            feature_importance (params: {target, task?, top_k?})\n\n"
            f"ITERATION: {iteration_number} of {max_iterations}\n"
            f"TARGET COLUMN: {target_column}\n"
            f"TASK TYPE: {task_type}\n"
            f"USER OBJECTIVES: {objectives or 'General data preparation for ML training'}\n"
            f"ENABLED STEPS: {json.dumps(enabled_steps)}\n\n"
            f"CURRENT DATA SNAPSHOT:\n```json\n{json.dumps(data_snapshot, indent=2, default=str)}\n```\n\n"
            f"QUALITY METRICS:\n```json\n{json.dumps(quality_metrics, indent=2, default=str)}\n```\n\n"
        )
        if history_summary:
            prompt += f"PREVIOUS ITERATION HISTORY:\n{history_summary}\n\n"

        prompt += (
            "RESPOND WITH THIS EXACT JSON STRUCTURE:\n"
            "{\n"
            '  "plan_summary": "Brief description of what this iteration will do",\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_type": "data_cleaning",\n'
            '      "title": "Human-readable step title",\n'
            '      "description": "What this step does and why",\n'
            '      "rationale": "Why this step is needed based on the data",\n'
            '      "operations": [{"action": "...", "params": {...}}]\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        raw = cls._call_llm(prompt)
        return cls._parse_json_response(raw, fallback={
            "plan_summary": raw[:300],
            "steps": [{
                "step_type": "data_analysis",
                "title": "Analyze Data Quality",
                "description": "Run quality analysis on the current dataset",
                "rationale": "Baseline assessment",
                "operations": [],
            }],
        })

    @classmethod
    def evaluate_workflow_step(
        cls,
        data_snapshot: dict,
        quality_metrics: dict,
        target_column: str,
        task_type: str,
        objectives: str,
    ) -> str:
        """Evaluate the current state of the data after a step."""
        logger.info("LLM: evaluating workflow step (target=%s)", target_column)
        prompt = (
            "You are the AceML Studio Workflow Evaluator.\n"
            "Evaluate the current state of the dataset and provide a brief assessment.\n\n"
            f"TARGET: {target_column} ({task_type})\n"
            f"OBJECTIVES: {objectives or 'General ML preparation'}\n\n"
            f"DATA SNAPSHOT:\n```json\n{json.dumps(data_snapshot, indent=2, default=str)}\n```\n\n"
            f"QUALITY METRICS:\n```json\n{json.dumps(quality_metrics, indent=2, default=str)}\n```\n\n"
            "Provide a brief (2-3 sentence) assessment of:\n"
            "1. Current data readiness for ML training\n"
            "2. Any remaining issues that need addressing\n"
            "3. Whether the data is ready for model training\n"
        )
        return cls._call_llm(prompt)

    @classmethod
    def evaluate_workflow_iteration(
        cls,
        data_snapshot: dict,
        quality_metrics: dict,
        target_column: str,
        task_type: str,
        objectives: str,
        iteration_number: int,
        max_iterations: int,
        step_summaries: list[dict],
        initial_quality: int,
    ) -> dict:
        """
        Evaluate a completed iteration and decide whether to continue.
        Returns dict with: evaluation, should_continue, improvement_summary.
        """
        logger.info("LLM: evaluating iteration %d (quality %d→%d)",
                     iteration_number, initial_quality,
                     quality_metrics.get("quality_score", 0))

        prompt = (
            "You are the AceML Studio Workflow Evaluator.\n"
            "An iteration of data preparation has completed. Evaluate the results and decide "
            "whether another iteration is needed.\n\n"
            "RULES:\n"
            "• Return ONLY valid JSON (no markdown fences).\n"
            "• Set should_continue to true ONLY if there are concrete, actionable improvements remaining.\n"
            "• If quality score is above 80 and no critical issues remain, recommend stopping.\n"
            "• If improvement from last iteration was minimal (<5 points), recommend stopping.\n\n"
            f"ITERATION: {iteration_number} of {max_iterations}\n"
            f"TARGET: {target_column} ({task_type})\n"
            f"OBJECTIVES: {objectives or 'General ML preparation'}\n"
            f"INITIAL QUALITY SCORE: {initial_quality}\n"
            f"CURRENT QUALITY SCORE: {quality_metrics.get('quality_score', 0)}\n\n"
            f"STEP RESULTS:\n```json\n{json.dumps(step_summaries, indent=2, default=str)}\n```\n\n"
            f"CURRENT DATA:\n```json\n{json.dumps(data_snapshot, indent=2, default=str)}\n```\n\n"
            f"QUALITY METRICS:\n```json\n{json.dumps(quality_metrics, indent=2, default=str)}\n```\n\n"
            "RESPOND WITH THIS EXACT JSON STRUCTURE:\n"
            "{\n"
            '  "evaluation": "Detailed evaluation of this iteration\'s results",\n'
            '  "should_continue": true/false,\n'
            '  "improvement_summary": "What improved and what still needs work",\n'
            '  "reason_to_continue_or_stop": "Why another iteration is/isn\'t needed"\n'
            "}\n"
        )

        raw = cls._call_llm(prompt)
        return cls._parse_json_response(raw, fallback={
            "evaluation": raw[:500],
            "should_continue": False,
            "improvement_summary": "Could not parse LLM response",
        })

    @classmethod
    def _parse_json_response(cls, raw: str, fallback: dict) -> dict:
        """Attempt to extract JSON from an LLM response string."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown fences
        import re
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                try:
                    candidate = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(candidate)
                except (json.JSONDecodeError, IndexError):
                    continue

        logger.warning("Could not parse JSON from LLM response (len=%d), using fallback", len(raw))
        return fallback

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
