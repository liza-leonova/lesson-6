
"""
Модуль обучения генеративной модели Transformer.
Включает функции для тренировки и валидации модели.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from model.transformer import GeneratorTransformer
from data.dataset import ZaratustraDataset, create_dataloaders


class ModelTrainer:
    """Класс для управления процессом обучения модели"""

    def __init__(self, model, train_data, val_data, dataset, config, device):
        self.model = model.to(device)
        self.train_loader = train_data
        self.val_loader = val_data
        self.text_dataset = dataset
        self.train_config = config
        self.compute_device = device

        self.loss_function = nn.CrossEntropyLoss(ignore_index=dataset.get_pad_token_id())
        self.model_optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=0.95)

        self.checkpoint_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_bleu_score = 0.0

    def run_training_epoch(self, epoch_num):
        """Выполнение одной эпохи обучения"""
        self.model.train()
        epoch_loss = 0.0
        epoch_bleu = 0.0

        progress = tqdm(self.train_loader, desc=f'Эпоха {epoch_num}')
        gradient_scaler = GradScaler()

        for i, batch in enumerate(progress):
            input_ids = batch['src_ids'].to(self.compute_device)
            target_ids = batch['trgt_ids'].to(self.compute_device)

            self.model_optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                model_output = self.model.forward(input_ids)
                current_loss = self.loss_function(
                    model_output[:, :-1].reshape(-1, model_output.size(-1)),
                    target_ids[:, 1:].reshape(-1)

                gradient_scaler.scale(current_loss).backward()

                for param in self.model.parameters():
                    if
                torch.isnan(param.grad).any():
                param.grad.zero_()

                epoch_bleu += self.calculate_bleu(model_output.argmax(dim=-1), target_ids[:, 1:])

                gradient_scaler.step(self.model_optimizer)
                gradient_scaler.update()
                epoch_loss += current_loss.item()

                progress.set_postfix({
                    'loss': f'{(epoch_loss / (i + 1)):.4f}',
                    'bleu': f'{(epoch_bleu / (i + 1)):.4f}'
                })

        return {
            'loss': epoch_loss / (i + 1),
            'bleu': epoch_bleu / (i + 1)
        }

    def run_validation(self, epoch_num):
        """Проведение валидации модели"""
        self.model.eval()
        total_bleu = 0.0

        for i, batch in enumerate(tqdm(self.val_loader, desc='Валидация')):
            input_ids = batch['src_ids'].to(self.compute_device)
            target_ids = batch['trgt_ids'].to(self.compute_device)
            targets = target_ids[:, 1:]

            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                generated = self.model.generate_from_ids(input_ids)

            total_bleu += self.calculate_bleu(generated, targets)

        total_bleu /= (i + 1)

        if total_bleu > self.best_bleu_score:
            self.best_bleu_score = total_bleu
            self.save_model('best_model.pt')

        return {'bleu': total_bleu}

    def calculate_bleu(self, predictions, targets):
        """Вычисление BLEU score между предсказаниями и целями"""
        pred_texts = self.text_dataset.tokenizer.decode_batch(predictions.cpu().numpy())
        target_texts = self.text_dataset.tokenizer.decode_batch(targets.cpu().numpy())
        bleu_total = 0.0

        for pred, target in zip(pred_texts, target_texts):
            bleu_total += sentence_bleu(
                [target.split()],
                pred.split(),
                smoothing_function=SmoothingFunction().method4,
                auto_reweigh=True
            )
        return bleu_total / len(pred_texts)

    def save_model(self, filename):
        """Сохранение состояния модели"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model_optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.train_config,
            'best_bleu': self.best_bleu_score,
        }

        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
        print(f'Модель сохранена: {save_path}')

    def train_model(self):
        """Основной цикл обучения"""
        epochs = self.train_config.get('num_epochs', 100)

        for epoch in range(epochs):
            print(f'\nЭпоха {epoch + 1}/{epochs}')

            train_stats = self.run_training_epoch(epoch)
            print(f'Тренировка - Потери: {train_stats["loss"]:.4f}, BLEU: {train_stats["bleu"]:.4f}')

            val_stats = self.run_validation(epoch)
            print(f'Валидация - BLEU: {val_stats["bleu"]:.4f}')

            self.lr_scheduler.step()

            if (epoch + 1) % self.train_config.get('save_epochs', 5) == 0:
                self.save_model(f'epoch_{epoch + 1}.pt')


def setup_training():
    """Настройка и запуск процесса обучения"""
    training_config = {
        'batch_size': 1,
        'max_seq_length': 128,
        'max_train_samples': 5000000,
        'max_val_samples': 3000,
        'model_dim': 256,
        'attention_heads': 8,
        'decoder_layers': 4,
        'ffn_dim': 1024,
        'dropout_rate': 0.1,
        'learning_rate': 1e-4,
        'total_epochs': 3,
        'save_frequency': 2,
        'checkpoint_dir': 'model_checkpoints',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используемое устройство: {device}')

    train_loader, val_loader, dataset = create_dataloaders(
        batch_size=training_config['batch_size'],
        max_length=training_config['max_seq_length'],
        max_train_samples=training_config['max_train_samples'],
        max_val_samples=training_config['max_val_samples'],
    )

    transformer_model = GeneratorTransformer(
        d_model=training_config['model_dim'],
        num_heads=training_config['attention_heads'],
        d_ff=training_config['ffn_dim'],
        num_layers=training_config['decoder_layers'],
        vocab_size=dataset.get_vocab_size(),
        pad_index=dataset.get_pad_token_id(),
        dropout=training_config['dropout_rate'],
        tokenizer=dataset.tokenizer,
        max_len=training_config['max_seq_length'],
    ).to(device)

    print(f'Параметры модели: {sum(p.numel() for p in transformer_model.parameters()):,}')

    trainer = ModelTrainer(
        model=transformer_model,
        train_data=train_loader,
        val_data=val_loader,
        dataset=dataset,
        config=training_config,
        device=device,
    )

    trainer.train_model()


if __name__ == '__main__':
    setup_training()