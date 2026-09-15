"""Greedy inference and a minimal non-streaming chat completions endpoint."""
import threading
import time
import uuid
import torch
from transformers import AutoTokenizer
from .runtime import load_model


class Generator:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = load_model(model, device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.device = device
        self.lock = threading.Lock()

    @torch.inference_mode()
    def chat(self, messages, max_tokens=32):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("max_tokens must be an integer between 1 and 512")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str) for m in messages
        ):
            raise ValueError("messages must contain text roles and contents")
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("request exceeds model context")
        with self.lock:
            tokens = torch.tensor([ids], device=self.device)
            output = self.model.generate(tokens, max_new_tokens=max_tokens, do_sample=False,
                                         top_k=0, top_p=1.0, temperature=1.0,
                                         eos_token_id=self.tokenizer.eos_token_id)
        generated = output[0, len(ids):].tolist()
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "prompt_tokens": len(ids), "completion_tokens": len(generated)}


def serve(model, tokenizer, device="cpu", host="127.0.0.1", port=8000):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    generator = Generator(model, tokenizer, device)

    @app.get("/health")
    def health():
        return jsonify(status="ready")

    @app.post("/v1/chat/completions")
    def chat():
        body = request.get_json()
        if not isinstance(body, dict):
            return jsonify(error={"message": "JSON object required"}), 400
        if body.get("stream") or any(k in body for k in ("tools", "tool_choice", "temperature", "top_p", "n")):
            return jsonify(error={"message": "v0.1 supports text-only, non-streaming greedy decoding"}), 400
        try:
            result = generator.chat(body.get("messages"), body.get("max_tokens", 32))
        except ValueError as error:
            return jsonify(error={"message": str(error)}), 400
        return jsonify(id="chatcmpl-" + uuid.uuid4().hex, object="chat.completion", created=int(time.time()),
                       model="mica", choices=[{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                                              "finish_reason": "length" if result["completion_tokens"] == body.get("max_tokens", 32) else "stop"}],
                       usage={"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                              "total_tokens": result["prompt_tokens"] + result["completion_tokens"]})
    app.run(host=host, port=port, threaded=False)
