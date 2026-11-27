"""CPU regression checks for the production assistant loss and accumulation scaling."""
import ast
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Trainer, TrainingArguments


def main():
    source = Path(__file__).with_name("train_qlora.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "AssistantOnlyTrainer")
    namespace = {"torch": torch, "Trainer": Trainer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(source), "exec"), namespace)
    loss_trainer = namespace["AssistantOnlyTrainer"]

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 8)
            self.head = torch.nn.Linear(8, 7)

        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=0, **kwargs):
            hidden = self.embedding(input_ids)
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:]
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(4300)
    model = TinyLM()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
              "attention_mask": torch.ones(1, 6, dtype=torch.long),
              "labels": torch.tensor([[-100, -100, -100, 4, 5, 6]])}
    with tempfile.TemporaryDirectory(prefix="ece4300-loss-check-") as output:
        args = TrainingArguments(output_dir=output, use_cpu=True, report_to="none",
                                 gradient_accumulation_steps=16, disable_tqdm=True)
        trainer = loss_trainer(model=model, args=args)
        full = model(**inputs).logits
        reference = torch.nn.functional.cross_entropy(
            full[:, :-1].reshape(-1, 7), inputs["labels"][:, 1:].reshape(-1), ignore_index=-100)
        reference.backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        optimized = trainer.compute_loss(model, copy.deepcopy(inputs))
        optimized.backward()
        torch.testing.assert_close(optimized, reference)
        for parameter, gradient in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, gradient)
        print("PASS: assistant-only loss and gradients match full masked CE", flush=True)

        assert trainer.model_accepts_loss_kwargs, "Toy model must reproduce automatic kwargs detection"
        for count, fix in [(16, False), (16, True), (5, True)]:
            model.zero_grad()
            trainer.model_accepts_loss_kwargs = not fix
            trainer.current_gradient_accumulation_steps = count
            for _ in range(count):
                trainer.training_step(model, copy.deepcopy(inputs),
                                      num_items_in_batch=torch.tensor(count * 3))
            factor = 1 if fix else count
            for parameter, gradient in zip(model.parameters(), expected):
                torch.testing.assert_close(parameter.grad, gradient * factor, rtol=2e-5, atol=2e-6)
            print(f"PASS: accumulation={count}, fixed={fix}, gradient factor={factor}", flush=True)


if __name__ == "__main__":
    main()
