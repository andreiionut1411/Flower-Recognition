import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('detect/yolov8_flower_detector/results.csv')
epochs = df['epoch']
train_losses = df['train/cls_loss']
val_losses = df['val/cls_loss']

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()

model_1_train_losses = [3.6823, 3.2124, 2.9463, 2.7100, 2.4784, 2.2984, 2.1798, 2.0485, 1.9212, 1.8376, 1.7597, 1.6440, 1.5538, 1.4766, 1.4283, 1.3893, 1.3200, 1.2943, 1.2437, 1.2204, 1.1840, 1.1309, 1.1018, 1.0810, 1.0577, 1.0291, 0.8390, 0.7663, 0.7331, 0.7217, 0.7096, 0.6943, 0.6964, 0.6704, 0.6703, 0.6511, 0.6437, 0.6388, 0.6300, 0.6266]
model_1_val_losses = [4.0376, 3.5503, 3.1440, 3.5033, 2.8529, 2.3294, 2.4378, 2.0652, 1.8080, 1.9451, 1.8677, 1.6155, 1.6511, 1.7584, 1.4450, 1.4055, 1.4474, 1.3019, 1.3290, 1.2538, 1.2140, 1.1495, 1.2364, 1.1589, 1.1769, 1.2020, 0.9585, 0.9238, 0.9343, 0.9171, 0.9074, 0.9062, 0.9020, 0.8973, 0.8965, 0.8918, 0.8990, 0.9200, 0.9175, 0.8828]
model_2_train_losses = [1.6699, 0.7697, 0.6278, 0.5439, 0.4972, 0.4514, 0.4310, 0.3958, 0.3672, 0.3594, 0.3398, 0.3095, 0.2609, 0.2310, 0.2355, 0.2349, 0.2182, 0.2231, 0.2192, 0.2096]
model_2_val_losses = [0.6915, 0.5806, 0.5388, 0.5305, 0.4839, 0.4780, 0.4725, 0.4443, 0.5040, 0.4724, 0.4778, 0.4827, 0.4618, 0.4553, 0.4515, 0.4496, 0.4490, 0.4482, 0.4528, 0.4521]
model_3_train_losses = [2.3548, 1.3042, 1.1515, 1.0392, 0.9929, 0.9510, 0.9003, 0.8939, 0.8585, 0.8418, 0.8199, 0.7982, 0.7838, 0.7612, 0.7456, 0.7273, 0.7255, 0.7057, 0.6964, 0.6865]
model_3_val_losses = [1.1925, 0.9891, 0.8638, 0.8363, 0.7860, 0.7727, 0.7443, 0.7224, 0.7360, 0.7308, 0.7404, 0.7163, 0.7033, 0.7075, 0.7050, 0.7114, 0.6929, 0.6918, 0.7069, 0.6874]

epochs_1 = range(1, len(model_1_train_losses) + 1)
epochs_2 = range(1, len(model_2_train_losses) + 1)
epochs_3 = range(1, len(model_3_train_losses) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_1, model_1_train_losses, label='Training Loss', marker='o')
plt.plot(epochs_1, model_1_val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss for training from scratch', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_2, model_2_train_losses, label='Training Loss', marker='o')
plt.plot(epochs_2, model_2_val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss for training from pretrained weights', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_3, model_3_train_losses, label='Training Loss', marker='o')
plt.plot(epochs_3, model_3_val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss for training with freezed layers', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()


