# predictor/Base_Predictor.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from utils.metrics import calculate_metrics  # Assuming metrics.py exists
from utils.logger import SingleExpRecorder  # Assuming logger.py has this
import copy

class BasePredictor:
    def __init__(self, model_conf, dataset_info, device):
        self.best_model_state = None

        self.model_conf = model_conf
        self.training_conf = model_conf.training
        self.device = device
        self.dataset_info = dataset_info  # Pass info like n_classes, input shape etc.

        # Instantiate model (should be done in subclass)
        self.model = self._build_model().to(device)

        # Setup optimizer
        optimizer_name = self.training_conf.optimizer.lower()
        lr = self.training_conf.lr
        weight_decay = getattr(self.training_conf, 'weight_decay', 0)
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.training_conf, 'momentum', 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Supported: 'adam', 'adamw', 'sgd'")

        # Setup loss function
        loss_name = self.training_conf.loss.lower()
        if loss_name == 'crossentropyloss' or loss_name == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        # Add other loss functions if needed (e.g., Focal Loss)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        # Setup learning rate scheduler (optional)
        self.scheduler = None
        scheduler_conf = getattr(self.training_conf, 'scheduler', None)
        if scheduler_conf:
            if scheduler_conf.type.lower() == 'reducelronplateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode=getattr(scheduler_conf, 'mode', 'min'),
                    factor=getattr(scheduler_conf, 'factor', 0.1),
                    patience=getattr(scheduler_conf, 'patience', 10)
                )
            elif scheduler_conf.type.lower() == 'cosineannealing':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.training_conf.epochs  # Or use T_max from config
                )
            # Add other schedulers

        self.epochs = self.training_conf.epochs
        self.patience = getattr(self.training_conf, 'patience', 50)  # Early stopping patience
        self.print_freq = getattr(self.training_conf, 'print_freq', 10)
        self.debug = getattr(self.training_conf, 'debug', False)

        self.best_val_metric = -1  # Or high value if using loss
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.total_time = 0
        self.results = {}  # To store results from the best epoch

    def _build_model(self):
        # This method MUST be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _build_model()")

    def train(self, train_loader, val_loader,test_loader):
        """Trains the model for the specified number of epochs."""
        recorder = SingleExpRecorder(self.patience, criterion='metric')  # Use 'loss' or 'metric'
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc, _ = self.evaluate(val_loader)  # Evaluate on validation set

            epoch_time = time.time() - epoch_start_time

            if epoch % self.print_freq == 0 or self.debug:
                print(f'Epoch: {epoch + 1:03d}/{self.epochs:03d} | Time: {epoch_time:.2f}s | '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.1e} | '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%')

            test_loss, test_acc, test_metrics = self.test(test_loader)
            # Learning rate scheduler step (depends on scheduler type)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # or val_acc if mode='max'
                else:  # For schedulers like CosineAnnealingLR, StepLR
                    self.scheduler.step()

            # Early stopping and best model saving logic
            # Use validation accuracy as the primary metric here
            improved, stop = recorder.add(val_loss, val_acc)

            if improved:
                self.best_val_metric = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                self.results['train_loss'] = train_loss
                self.results['train_acc'] = train_acc
                self.results['valid_loss'] = val_loss
                self.results['valid_acc'] = val_acc
                self.results['best_epoch'] = epoch + 1

            if stop:
                print(f'Early stopping triggered at epoch {epoch + 1} (Patience: {self.patience})')
                break

        self.total_time = time.time() - start_time
        print(f'Training finished. Total time: {self.total_time:.2f}s')
        print(f'Best Validation Accuracy: {self.best_val_metric * 100:.2f}% at epoch {self.best_epoch + 1}')

        # Load best model weights if saved, or just report results from best epoch
        # if os.path.exists('best_model.pth'):
        #    self.model.load_state_dict(torch.load('best_model.pth'))

        # Return performance metrics from the best validation epoch
        # The logger expects 'train', 'valid', 'test' keys usually
        final_result = {
            # Use results saved when validation improved
            "train_acc": self.results.get('train_acc', 0.0),
            "valid_acc": self.results.get('valid_acc', 0.0),
            "test": 0.0,  # Test accuracy will be calculated separately
            "train_loss": self.results.get('train_loss', float('inf')),
            "valid_loss": self.results.get('valid_loss', float('inf')),
            "best_epoch": self.results.get('best_epoch', -1),
            "total_time": self.total_time,
            # Add other metrics if calculated during validation
        }
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {self.best_epoch + 1} for final testing.")
        else:
            print("Warning: No best model state was saved; using current model weights.")

        return final_result

    def _train_epoch(self, train_loader):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # Optional: Gradient clipping
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """Evaluates the model on a given dataset."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probabilities)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])

        avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), y_prob=all_probs)  # Get dict of metrics

        # Return loss, primary metric (e.g., accuracy), and all metrics
        return avg_loss, metrics.get('accuracy', 0.0), metrics

    def test(self, test_loader):
        """ Performs final evaluation on the test set."""
        print("Evaluating on Test Set...")
        # Ensure the best model state is loaded if checkpoints were saved,
        # otherwise uses the model state from the end of training (or best epoch if not overwritten).
        test_loss, test_acc, test_metrics = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')
        # Add other metrics like F1, AUC if available in test_metrics
        print(f'Test Metrics: {test_metrics}')
        return test_loss, test_acc, test_metrics
