# utils/logger.py
import os
import time
import datetime
import numpy as np
from prettytable import PrettyTable


class Logger:
    """训练和评估日志记录器"""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 生成日志文件名（包含时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")

    def log(self, message):
        """记录日志到文件和控制台"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def log_metrics(self, metrics, epoch, mode='Train'):
        """格式化记录评估指标"""
        table = PrettyTable()
        table.title = f"{mode} Metrics - Epoch {epoch}"
        table.field_names = ["Category", "F1_a", "F1_v", "F1_av", "F1"]

        for i, cat in enumerate(metrics['categories']):
            row = [
                cat,
                f"{metrics['F1_a'][i]:.4f}",
                f"{metrics['F1_v'][i]:.4f}",
                f"{metrics['F1_av'][i]:.4f}",
                f"{metrics['F1'][i]:.4f}"
            ]
            table.add_row(row)

        table.add_row(["-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10])
        row = [
            "Average",
            f"{metrics['avg_F1_a']:.4f}",
            f"{metrics['avg_F1_v']:.4f}",
            f"{metrics['avg_F1_av']:.4f}",
            f"{metrics['avg_F1']:.4f}"
        ]
        table.add_row(row)

        self.log(f"\n{table}")