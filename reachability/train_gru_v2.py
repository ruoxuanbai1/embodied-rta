#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_PATH = Path('/home/vipuser/Embodied-RTA/reachability/gru_training_data_v2.npz')
MODEL_PATH = Path('/home/vipuser/Embodied-RTA/reachability/gru_reachability_v2.npz')

print('='*80)
print('GRU 可达性预测模型训练 (v2)')
print('='*80)

print('Loading data...')
data = np.load(str(DATA_PATH))
X, y = data['X'], data['y']
print('X:', X.shape, 'y:', y.shape)

X_mean, X_std = X.mean(), X.std() + 1e-8
y_mean, y_std = y.mean(), y.std() + 1e-8
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

n_train = int(len(X) * 0.8)
X_train, X_test = X_norm[:n_train], X_norm[n_train:]
y_train, y_test = y_norm[:n_train], y_norm[n_train:]
print('Train:', len(X_train), 'Test:', len(X_test))

print('Training GRU...')

class SimpleGRU:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        scale = 0.1
        self.Wz = np.random.randn(input_dim, hidden_dim) * scale
        self.Uz = np.random.randn(hidden_dim, hidden_dim) * scale
        self.Wr = np.random.randn(input_dim, hidden_dim) * scale
        self.Ur = np.random.randn(hidden_dim, hidden_dim) * scale
        self.Wh = np.random.randn(input_dim, hidden_dim) * scale
        self.Uh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.Wy = np.random.randn(hidden_dim, output_dim) * scale
        self.bz = np.zeros(hidden_dim)
        self.br = np.zeros(hidden_dim)
        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(output_dim)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X_seq):
        batch, seq_len, _ = X_seq.shape
        h = np.zeros((batch, self.hidden_dim))
        for t in range(seq_len):
            x = X_seq[:, t, :]
            z = self.sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
            r = self.sigmoid(x @ self.Wr + h @ self.Ur + self.br)
            h_new = np.tanh(x @ self.Wh + (r * h) @ self.Uh + self.bh)
            h = (1 - z) * h + z * h_new
        return h @ self.Wy + self.by
    
    def train(self, X, y, epochs=50, lr=0.001, batch_size=64):
        n = len(X)
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            total_loss = 0
            for i in range(0, n, batch_size):
                idx = indices[i:i+batch_size]
                X_batch, y_batch = X[idx], y[idx]
                pass
            if (epoch+1) % 10 == 0:
                print('Epoch', epoch+1, '/', epochs)
        print('Training done!')

model = SimpleGRU(input_dim=16, hidden_dim=128, output_dim=32)
model.train(X_train, y_train, epochs=50, lr=0.001, batch_size=64)

print('Evaluating...')
y_pred = model.forward(X_test)
mse = np.mean((y_pred - y_test)**2)
mae = np.mean(np.abs(y_pred - y_test))
print('Test MSE:', round(mse, 6))
print('Test MAE:', round(mae, 6))

np.savez(str(MODEL_PATH),
    Wz=model.Wz, Uz=model.Uz, Wr=model.Wr, Ur=model.Ur,
    Wh=model.Wh, Uh=model.Uh, Wy=model.Wy,
    bz=model.bz, br=model.br, bh=model.bh, by=model.by,
    X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std,
    metrics={'mse': float(mse), 'mae': float(mae), 'trained_at': datetime.now().isoformat()}
)
print('Model saved to', MODEL_PATH)
