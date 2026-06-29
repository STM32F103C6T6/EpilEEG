# predictor/MedFormer_Predictor.py
import numpy as np
import torch

from utils.metrics import calculate_metrics
from predictor.Base_Predictor  import BasePredictor
from models.medformer import Model as MedFormerModel  # 导入修改后的 MedFormer 模型类


class MedFormer_Predictor(BasePredictor):
    def _build_model(self):
        """
        构建 MedFormer 模型实例。
        """
        print(f"Building MedFormer model...")
        model_specific_conf = self.model_conf.model  # 获取 model: 下的配置
        print(f"  Passing model-specific config to MedFormer: {model_specific_conf}")
        print(f"  Dataset Info: {self.dataset_info}")

        # 将 model_specific_conf 和 dataset_info 传递给模型构造函数
        model = MedFormerModel(model_conf=model_specific_conf, dataset_info=self.dataset_info)

        return model

    # --- 可能需要重写 _train_epoch 或 evaluate ---
    # 因为 MedFormer 的 forward 可能只接受 x_enc
    def _train_epoch(self, train_loader):
        """ 重写以适应 MedFormer 的 forward 接口 """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)



            self.optimizer.zero_grad()
            # --- 调用 forward 时只传递 inputs (x_enc) ---
            outputs = self.model(inputs)
            # --- 修改结束 ---
            if outputs is None:
                continue  # Handle cases where forward might return None for wrong task

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """ 重写以适应 MedFormer 的 forward 接口，并计算概率 """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # --- 调用 forward 时只传递 inputs (x_enc) ---
                outputs = self.model(inputs)
                # --- 修改结束 ---
                if outputs is None:
                    continue

                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

                all_probs.append(probabilities)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0
        # --- 确保传递了概率 ---
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), y_prob=all_probs)

        return avg_loss, metrics.get('accuracy', 0.0), metrics
