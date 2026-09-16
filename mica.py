"""Mica command line interface."""
import argparse
import json
import os
import torch
from model import __version__, MicaConfig, MicaForCausalLM
from model.runtime import train, evaluate, import_legacy, load_model


def main():
    parser = argparse.ArgumentParser(prog="mica")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    p = commands.add_parser("train")
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p = commands.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cpu")
    p = commands.add_parser("data")
    p.add_argument("--source", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=768)
    p = commands.add_parser("import-legacy")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    for name in ("generate", "serve"):
        p = commands.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--device", default="cpu")
        if name == "generate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--max-new-tokens", type=int, default=32)
        else:
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "doctor":
        import transformers
        result = {"mica": __version__, "torch": torch.__version__, "transformers": transformers.__version__,
                  "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()}
    elif command == "train":
        args["recipe_path"] = args.pop("recipe")
        result = train(**args)
    elif command == "evaluate":
        args["model_path"] = args.pop("model")
        result = evaluate(**args)
    elif command == "data":
        from dataset.prepare import prepare
        args["tokenizer_path"] = args.pop("tokenizer")
        result = prepare(**args)
    elif command == "import-legacy":
        args["config_path"] = args.pop("config")
        result = import_legacy(**args)
    elif command == "generate":
        from inference.generation import Generator
        generator = Generator(args["model"], args["tokenizer"], args["device"])
        result = generator.chat([{"role": "user", "content": args["prompt"]}], args["max_new_tokens"])
    else:
        from inference.generation import serve
        serve(**args)
        return
    if int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
