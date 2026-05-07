from __future__ import annotations

import argparse
from pathlib import Path

from aprompt.llm import DryRunClient, OpenAIResponsesClient
from aprompt.orchestrator import TopicFormationOrchestrator
from aprompt.topic_to_code_orchestrator import TopicToCodeOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topic formation and topic-to-code prompting.")
    parser.add_argument("--template", required=True, help="Path to prompt template YAML.")
    parser.add_argument("--task-vars", required=True, help="Path to task variable YAML.")
    parser.add_argument("--run-dir", default="runs/run_001", help="Directory for logs and final output.")
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-refine-count", type=int, default=1)
    parser.add_argument(
        "--model-meta",
        default="gpt5.1",
        choices=["gpt5.1", "gpt4.1"],
        help="Simple model profile. gpt5.1 uses reasoning effort medium; gpt4.1 omits reasoning.",
    )
    parser.add_argument(
        "--feature-plan-web-search",
        action="store_true",
        help="Enable OpenAI web_search tool for expert scenario-specific feature-plan prompts.",
    )
    parser.add_argument("--model", default=None, help="Optional exact OpenAI model override.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--non-interactive", action="store_true", help="Do not stop for manual review.")
    parser.add_argument("--dry-run", action="store_true", help="Use no-cost mock LLM responses to test control flow.")
    args = parser.parse_args()

    if args.dry_run:
        llm = DryRunClient()
    else:
        llm = OpenAIResponsesClient(
            model=args.model,
            model_meta=args.model_meta,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            store=False,
        )

    topic_orchestrator = TopicFormationOrchestrator(
        template_path=args.template,
        task_variables_path=args.task_vars,
        llm_client=llm,
        run_dir=Path(args.run_dir),
        num_experts=args.num_experts,
        max_rounds=args.max_rounds,
        interactive=not args.non_interactive,
        max_refine_count=args.max_refine_count,
    )
    topic_orchestrator.run()

    code_orchestrator = TopicToCodeOrchestrator(
        experts=topic_orchestrator.experts,
        template_store=topic_orchestrator.template_store,
        task_variables=topic_orchestrator.task_variables,
        llm_client=llm,
        run_dir=Path(args.run_dir),
        interactive=not args.non_interactive,
        call_index=topic_orchestrator.call_index,
        enable_feature_plan_web_search=args.feature_plan_web_search,
    )
    code_orchestrator.run()


if __name__ == "__main__":
    main()
