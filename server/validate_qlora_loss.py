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
    assert any(isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute)
                       and isinstance(t.value, ast.Name) and t.value.id == "trainer"
                       and t.attr == "model_accepts_loss_kwargs" for t in n.targets)
               and isinstance(n.value, ast.Constant) and n.value.value is False
               for n in ast.walk(tree)), "Production Trainer must disable automatic loss kwargs scaling"
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

        class RecordingTrainer(loss_trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                self.window_sizes.append(self.current_gradient_accumulation_steps)
                return super().compute_loss(model, inputs, return_outputs, **kwargs)

        integration_args = TrainingArguments(
            output_dir=output, use_cpu=True, report_to="none", disable_tqdm=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            num_train_epochs=1, learning_rate=0.0, save_strategy="no",
            logging_strategy="no", remove_unused_columns=False,
        )
        examples = [{key: value[0].clone() for key, value in inputs.items()} for _ in range(21)]
        integration = RecordingTrainer(model=TinyLM(), args=integration_args, train_dataset=examples)
        integration.model_accepts_loss_kwargs = False
        integration.window_sizes = []
        integration.train()
        assert integration.window_sizes == [16] * 16 + [5] * 5, integration.window_sizes
        assert integration.state.global_step == 2 and integration.state.epoch == 1.0
        print("PASS: real Trainer.train automatically uses accumulation windows 16 and 5", flush=True)


if __name__ == "__main__":
    main()
