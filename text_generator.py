
"""
Модуль с реализацией архитектуры Transformer (Decoder-only).
Содержит основные компоненты модели.
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model.transformer_layers import MultiheadAttention, FeedForward, Decoder, DecoderLayer, Embedding


def create_padding_mask(input_tensor, pad_idx):
    return (input_tensor != pad_idx).unsqueeze(-2)


def create_lookahead_mask(input_tensor):
    batch_size, seq_len = input_tensor.size()
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class TextGenerator(torch.nn.Module):
    def __init__(
            self,
            d_model: int = 64,
            num_heads: int = 8,
            d_ff: int = 512,
            num_layers: int = 6,
            vocab_size: int = 1000,
            pad_index: int = 1,
            dropout: float = 0.1,
            max_len: int = 64,
            tokenizer: Tokenizer = None,
            device: str = 'cuda'
    ):
        super().__init__()
        attention_layer = MultiheadAttention(d_model, num_heads, dropout)
        ffn_layer = FeedForward(d_model, d_ff, dropout)
        self.decoder_stack = Decoder(DecoderLayer(attention_layer, ffn_layer, dropout), num_layers)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.embedding_layer = Embedding(d_model, vocab_size, pad_index)
        self.output_projection = torch.nn.Linear(d_model, vocab_size)

        self.pad_idx = pad_index
        self.compute_device = device
        self.max_seq_len = max_len
        self.text_tokenizer = tokenizer

    def process_input(self, x) -> torch.Tensor:
        mask = create_padding_mask(x, self.pad_idx) & create_lookahead_mask(x).to(self.compute_device)
        x = self.embedding_layer(x)
        x = self.decoder_stack.forward(x, mask)
        x = self.layer_norm(x)
        return self.output_projection(x)

    def forward(self, x):
        return self.process_input(x)

    def beam_search(
            self,
            prompt: str,
            beam_width: int = 5,
            max_length: int = 200,
            length_penalty: float = 0.6,
            temperature: float = 1.0,
            early_stopping: bool = True
    ) -> str:
        self.eval()
        eos_token_id = self.text_tokenizer.token_to_id('</s>')
        bos_token_id = self.text_tokenizer.token_to_id('<s>')

        input_ids = self.text_tokenizer.encode(prompt).ids
        input_ids = [bos_token_id] + input_ids if input_ids[0] != bos_token_id else input_ids
        input_tensor = torch.tensor([input_ids], device=self.compute_device)

        beam_scores = torch.zeros(beam_width, device=self.compute_device)
        beam_sequences = input_tensor.repeat(beam_width, 1)
        completed_sequences = []
        completed_scores = []

        for step in range(max_length):
            with torch.no_grad():
                logits = self(beam_sequences)[:, -1, :] / temperature
                vocab_size = logits.size(-1)

                probs = F.softmax(logits, dim=-1)
                next_scores = torch.log(probs) + beam_scores.unsqueeze(-1)

                next_scores = next_scores.view(-1)
                topk_scores, topk_indices = torch.topk(next_scores, beam_width * 2)

                beam_indices = topk_indices // vocab_size
                token_indices = topk_indices % vocab_size

                new_sequences = []
                new_scores = []

                for score, beam_idx, token_idx in zip(topk_scores, beam_indices, token_indices):
                    sequence = torch.cat([
                        beam_sequences[beam_idx],
                        token_idx.unsqueeze(0)
                    ], dim=-1)

                    if token_idx == eos_token_id:
                        completed_sequences.append(sequence)
                        completed_scores.append(score / (sequence.size(-1) ** length_penalty))
                        beam_width -= 1
                    else:
                        new_sequences.append(sequence)
                        new_scores.append(score)

                    if len(new_sequences) >= beam_width:
                        break

                if new_sequences:
                    beam_sequences = torch.stack(new_sequences)
                    beam_scores = torch.tensor(new_scores, device=self.compute_device)
                else:
                    break

                if early_stopping and len(completed_sequences) >= beam_width:
                    break

        if not completed_sequences or not early_stopping:
            for i in range(beam_sequences.size(0)):
                completed_sequences.append(beam_sequences[i])
                completed_scores.append(beam_scores[i] / (beam_sequences.size(-1) ** length_penalty))

        best_idx = torch.argmax(torch.tensor(completed_scores)).item()
        best_sequence = completed_sequences[best_idx].cpu().tolist()

        decoded = self.text_tokenizer.decode(best_sequence)
        return decoded.replace('<s>', '').replace('</s>', '').strip()

    def generate(
            self,
            prompt,
            context_size=50,
            temp=1.0,
            max_tokens=200,
            beam_size=1
    ):
        if beam_size > 1:
            return self.beam_search(
                prompt,
                beam_width=beam_size,
                max_length=max_tokens,
                temperature=temp
            )
        return self._greedy_generate(prompt, context_size, temp, max_tokens)

    def _greedy_generate(self, prompt, context_size, temp, max_tokens):
        self.eval()
        eos_id = self.text_tokenizer.token_to_id('</s>')

        with torch.no_grad():
            input_tokens = self.text_tokenizer.encode(prompt).ids
            input_tensor = torch.tensor([input_tokens]).to(self.compute_device)
            output_sequence = input_tensor.clone()

            for _ in range(max_tokens):
                model_output = self(input_tensor)
                next_token_logits = model_output[:, -1, :] / temp
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                output_sequence = torch.cat([output_sequence, next_token], dim=1)
                input_tensor = output_sequence[:, -context_size:]

                if next_token.item() == eos_id:
                    break

        return self.text_tokenizer.decode(output_sequence[0].tolist())

    @staticmethod
    def load_pretrained(checkpoint_path: str, tokenizer):
        model = TextGenerator(tokenizer=tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint_path, map_location=device)
        model.to(device)
        model.load_state_dict(state)
        return model