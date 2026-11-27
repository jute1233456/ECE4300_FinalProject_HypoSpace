import argparse
import sys
from .client import ArkCodingPlanClient, LLMError


def main():
    parser = argparse.ArgumentParser(description="火山方舟 Coding Plan 单次模型调用")
    parser.add_argument("prompt", help="要发送给模型的问题")
    parser.add_argument("--model", help="覆盖配置文件中的模型名称")
    parser.add_argument("--config", help="配置文件路径；默认 config/llm.json")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--provider", choices=("ark", "deepseek"), default="ark")
    args = parser.parse_args()
    try:
        if args.provider == "deepseek":
            from .deepseek_official import DeepSeekOfficialClient
            client = DeepSeekOfficialClient(model=args.model, config_path=args.config)
        else:
            client = ArkCodingPlanClient(model=args.model, config_path=args.config)
        result = client.generate(args.prompt, max_tokens=args.max_tokens)
    except (ValueError, LLMError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(result.text)
    print(f"model={result.model}; total_tokens={result.total_tokens}; "
          f"finish_reason={result.finish_reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
