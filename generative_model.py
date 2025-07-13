
"""
Модуль для интерактивного взаимодействия с моделью Transformer.
Реализует чат-интерфейс для генерации текста.
"""

from model.transformer import TextGenerator
from tokenizers import Tokenizer
import torch
from data.dataset import TextDataset, create_dataloaders


def initialize_chat():
    model_config = {
        'batch_size': 1,
        'context_length': 128,
        'train_samples_limit': 5000000,
        'val_samples_limit': 3000,
        'model_dim': 256,
        'attention_heads': 8,
        'decoder_layers': 4,
        'ffn_dim': 1024,
        'dropout_prob': 0.1,
        'learning_rate': 1e-4,
        'total_epochs': 4,
        'save_frequency': 4,
        'checkpoint_dir': 'model_checkpoints',
    }

    text_tokenizer = Tokenizer.from_file("tokenizer.json")
    train_loader, val_loader, text_dataset = create_dataloaders(
        batch_size=model_config['batch_size'],
        max_seq_len=model_config['context_length'],
        max_train=model_config['train_samples_limit'],
        max_val=model_config['val_samples_limit'],
    )

    model_checkpoint = torch.load("model_checkpoints/epoch_2.pt", map_location='cuda')
    text_generator = TextGenerator(
        d_model=model_config['model_dim'],
        num_heads=model_config['attention_heads'],
        d_ff=model_config['ffn_dim'],
        num_layers=model_config['decoder_layers'],
        vocab_size=text_dataset.get_vocab_size(),
        pad_index=text_dataset.get_pad_token_id(),
        dropout=model_config['dropout_prob'],
        tokenizer=text_dataset.tokenizer,
        max_len=model_config['context_length'],
    ).to('cuda')

    text_generator.load_state_dict(model_checkpoint['model_state_dict'])
    text_generator.eval()
    return text_generator


def run_chat_interface():
    generator_model = initialize_chat()

    print("Система готова к диалогу. Введите 'exit' для завершения.")
    while True:
        user_input = input("Пользователь: ")
        if user_input.lower() == 'exit':
            break

        # Генерация с beam search (ширина 3)
        generated_response = generator_model.generate(
            user_input,
            context_size=50,
            max_tokens=200,
            temp=0.8,
            beam_size=3  # Включаем beam search
        )
        print(f"Модель: {generated_response}")


if __name__ == "__main__":
    run_chat_interface()