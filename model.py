from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_default_config
from speech_encoder import build_speech_encoder


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        # Step 4: Connector / projection speech encoder output ko LLM embedding space me map karta hai.
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PreNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        # Step 8: Training ke time ground-truth continuation spectrogram ko LLM side tokens me badalne ke liye.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PostNet(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        # Step 9/10: LLM hidden states ko spectrogram space me map karta hai.
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpokenQAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = get_default_config()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.model.llm_name,
            local_files_only=True,
            attn_implementation="eager",
        )

        # Step 3: Speech encoder
        hidden_dim = self.lm.config.n_embd
        self.encoder = build_speech_encoder()
        # Step 4: Connector / projection
        self.projection = ProjectionLayer(
            input_dim=cfg.model.speech_hidden_size,
            output_dim=hidden_dim,
        )
        # Step 8: PreNet
        self.prenet = PreNet(
            input_dim=cfg.audio.n_mels,
            hidden_dim=hidden_dim,
            dropout=cfg.model.dropout,
        )
        # Step 9/10: PostNet
        self.postnet = PostNet(
            hidden_dim=hidden_dim,
            output_dim=cfg.audio.n_mels,
            dropout=cfg.model.dropout,
        )
        self.speech_prompt_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def _normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return (mel - mean) / std

    def _build_prompt_mask(self, prompt_padding_mask: torch.Tensor | None, batch_size: int, prompt_length: int, device: torch.device) -> torch.Tensor:
        if prompt_padding_mask is None:
            return torch.ones(batch_size, prompt_length, dtype=torch.long, device=device)
        return (~prompt_padding_mask).long()

    def encode_text_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Step 5/6: Same LLM ke text embeddings se text-side semantic representation nikalna.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        text_embeds = self.lm.transformer.wte(input_ids)
        hidden = self._run_transformer(text_embeds, attention_mask)
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return hidden, pooled

    def encode_prompt(
        self,
        prompt_mel: torch.Tensor,
        prompt_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Step 2 -> 3 -> 4: Speech features ko normalize karke encoder aur connector se LLM-space prompt tokens banana.
        normalized_prompt_mel = self._normalize_mel(prompt_mel)
        prompt_hidden, prompt_padding_mask = self.encoder(normalized_prompt_mel, prompt_lengths)
        prompt_tokens = self.projection(prompt_hidden) + self.speech_prompt_embedding
        prompt_mask = self._build_prompt_mask(
            prompt_padding_mask,
            batch_size=prompt_mel.size(0),
            prompt_length=prompt_tokens.size(1),
            device=prompt_tokens.device,
        )
        pooled_prompt = (prompt_tokens * prompt_mask.unsqueeze(-1)).sum(dim=1)
        pooled_prompt = pooled_prompt / prompt_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return prompt_tokens, prompt_mask, pooled_prompt

    def _run_transformer(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Step 5: Main LLM forward pass.
        outputs = self.lm.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.last_hidden_state

    @torch.no_grad()
    def _generate_from_text_prefix(
        self,
        prompt_mel: torch.Tensor,
        prompt_lengths: torch.Tensor | None,
        prefix_text: str,
        max_new_tokens: int,
        stop_on_newline: bool = False,
    ) -> str:
        self.eval()
        # Step 6: Speech-conditioned text generation.
        prompt_tokens, prompt_mask, _ = self.encode_prompt(prompt_mel, prompt_lengths)
        input_ids = self.tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(prompt_mel.device)
        input_ids = input_ids.repeat(prompt_mel.size(0), 1)
        generated_ids = []
        repetition_penalty = 1.1
        newline_token_ids = set(self.tokenizer("\n", add_special_tokens=False)["input_ids"])

        for _ in range(max_new_tokens):
            text_embeds = self.lm.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prompt_tokens, text_embeds], dim=1)
            text_mask = torch.ones(
                input_ids.size(0),
                input_ids.size(1),
                dtype=torch.long,
                device=prompt_mel.device,
            )
            attention_mask = torch.cat([prompt_mask, text_mask], dim=1)
            hidden = self._run_transformer(inputs_embeds, attention_mask)
            next_logits = self.lm.lm_head(hidden[:, -1:, :]).squeeze(1)
            for row_index in range(input_ids.size(0)):
                seen_tokens = torch.unique(input_ids[row_index])
                next_logits[row_index, seen_tokens] = next_logits[row_index, seen_tokens] / repetition_penalty
                if generated_ids:
                    next_logits[row_index, generated_ids[-1][row_index].item()] = -1e9

            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_ids.append(next_token)

            if torch.all(next_token.squeeze(1) == self.tokenizer.eos_token_id):
                break
            if stop_on_newline and all(token.item() in newline_token_ids for token in next_token.squeeze(1)):
                break

        if generated_ids:
            output_ids = torch.cat(generated_ids, dim=1)
        else:
            output_ids = input_ids[:, 0:0]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def generate_question_text(
        self,
        prompt_mel: torch.Tensor,
        prompt_lengths: torch.Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        generated = self._generate_from_text_prefix(
            prompt_mel,
            prompt_lengths=prompt_lengths,
            prefix_text="Question:",
            max_new_tokens=max_new_tokens,
            stop_on_newline=True,
        )
        generated = generated.replace("Question:", "").replace("Answer:", "").strip()
        return generated

    @torch.no_grad()
    def generate_answer_from_question_text(
        self,
        prompt_mel: torch.Tensor,
        prompt_lengths: torch.Tensor | None,
        question_text: str,
        max_new_tokens: int = 64,
    ) -> str:
        prefix = f"Question: {question_text}\nAnswer:"
        generated = self._generate_from_text_prefix(
            prompt_mel,
            prompt_lengths=prompt_lengths,
            prefix_text=prefix,
            max_new_tokens=max_new_tokens,
            stop_on_newline=True,
        )
        generated = generated.replace("Answer:", "").strip()
        return generated
   #7
    def forward(
        self,
        prompt_mel: torch.Tensor,
        prompt_lengths: torch.Tensor | None = None,
        continuation_mel: torch.Tensor | None = None,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Step 3/4: Spoken prompt ko LLM-space tokens me convert karna.
        prompt_tokens, prompt_mask, pooled_prompt = self.encode_prompt(prompt_mel, prompt_lengths)
        batch_size = prompt_mel.size(0)

        hidden_chunks = [prompt_tokens]
        mask_chunks = [prompt_mask]
        text_logits = None

        if text_input_ids is not None:
            # Step 6: Text continuation / QA text branch.
            text_embeds = self.lm.transformer.wte(text_input_ids)
            hidden_chunks.append(text_embeds)
            if text_attention_mask is None:
                text_attention_mask = torch.ones_like(text_input_ids)
            mask_chunks.append(text_attention_mask)

        if continuation_mel is not None:
            # Step 8: Teacher-forced acoustic continuation tokens.
            normalized_continuation_mel = self._normalize_mel(continuation_mel)
            continuation_tokens = self.prenet(normalized_continuation_mel)
            hidden_chunks.append(continuation_tokens)
            continuation_mask = torch.ones(
                batch_size,
                continuation_tokens.size(1),
                dtype=torch.long,
                device=prompt_tokens.device,
            )
            mask_chunks.append(continuation_mask)

        llm_input = torch.cat(hidden_chunks, dim=1)
        llm_mask = torch.cat(mask_chunks, dim=1)
        # Step 5: LLM ko speech + text + acoustic tokens dena.
        hidden = self._run_transformer(llm_input, llm_mask)

        offset = prompt_tokens.size(1)
        if text_input_ids is not None:
            # Step 6: Text logits nikalna.
            text_hidden = hidden[:, offset : offset + text_input_ids.size(1), :]
            text_logits = self.lm.lm_head(text_hidden)
            offset = offset + text_input_ids.size(1)

        # Step 7/9/10: Acoustic continuation hidden states se spectrogram output banana.
        acoustic_hidden = hidden[:, offset:, :] if continuation_mel is not None else hidden[:, -1:, :]
        acoustic_output = self.postnet(acoustic_hidden)

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_mask": prompt_mask,
            "pooled_prompt": pooled_prompt,
            "hidden": hidden,
            "text_logits": text_logits,
            "acoustic_output": acoustic_output,
        }

    @torch.no_grad()
    def generate_answer_text(
        self,
        prompt_mel: torch.Tensor,
        prompt_lengths: torch.Tensor | None = None,
        max_new_tokens: int = 96,
    ) -> str:
        question_text = self.generate_question_text(
            prompt_mel,
            prompt_lengths=prompt_lengths,
            max_new_tokens=max_new_tokens // 2,
        )
        if not question_text:
            question_text = "unknown question"
        return self.generate_answer_from_question_text(
            prompt_mel,
            prompt_lengths=prompt_lengths,
            question_text=question_text,
            max_new_tokens=max_new_tokens,
        )


if __name__ == "__main__":
    model = SpokenQAModel()
    prompt = torch.randn(2, 120, 128)
    prompt_lengths = torch.tensor([120, 90], dtype=torch.long)
    tokenized = model.tokenizer(["hello world", "test"], return_tensors="pt", padding=True)
    outputs = model(
        prompt,
        prompt_lengths=prompt_lengths,
        text_input_ids=tokenized["input_ids"],
        text_attention_mask=tokenized["attention_mask"],
    )
    print(f"Prompt tokens shape: {tuple(outputs['prompt_tokens'].shape)}")
    print(f"Hidden shape: {tuple(outputs['hidden'].shape)}")
    print(f"Text logits shape: {tuple(outputs['text_logits'].shape)}")
    print(f"Acoustic output shape: {tuple(outputs['acoustic_output'].shape)}")
