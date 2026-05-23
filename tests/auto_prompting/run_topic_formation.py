from __future__ import annotations

import argparse
from pathlib import Path

from aprompt.llm import DryRunClient, OpenAIResponsesClient
from aprompt.orchestrator import TopicFormationOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic topic-set formation prompting.")
    parser.add_argument("--template", required=True, help="Path to prompt template YAML.")
    parser.add_argument("--task-vars", required=True, help="Path to task variable YAML.")
    parser.add_argument("--run-dir", default="runs/run_001", help="Directory for logs and final output.")
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--model", default=None, help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4.1-mini.")
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
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            store=False,
        )

    orchestrator = TopicFormationOrchestrator(
        template_path=args.template,
        task_variables_path=args.task_vars,
        llm_client=llm,
        run_dir=Path(args.run_dir),
        num_experts=args.num_experts,
        max_rounds=args.max_rounds,
        interactive=not args.non_interactive,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
