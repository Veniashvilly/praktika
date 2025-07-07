import torch
import matplotlib.pyplot as plt


def plot_training_history(history,save_path):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def compare_models(fc_history, cnn_history,save_path):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

from sklearn.metrics import confusion_matrix

def print_confusion_matrix(model, dataloader, device, num_classes=10, title="Confusion Matrix"):
    """Построение confusion matrix """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad(): #отключаем градиенты, чтобы не тратить на них ресурсы
        for data, labels in dataloader:
            data = data.to(device)
            outputs = model(data) #получаем вероятности классов
            preds = torch.argmax(outputs, dim=1) #получаем предсказанный класс
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.tolist())

    cm = confusion_matrix(all_true, all_preds)
    class_names = [str(i) for i in range(num_classes)]

    print(f"\n{title}")
    header = "Pred → " + "  ".join([f"{name:>6}" for name in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:>6}" for val in row])
        print(f"True {class_names[i]:>3} {row_str}")


def plot_gradient_flow(model, save_path=None):
    """Показывает величину градиентов по слоям"""
    ave_grads = []
    layers = []
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            if param.grad is not None:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
    
    plt.figure(figsize=(10, 4))
    plt.plot(ave_grads, alpha=0.8, lw=2)
    plt.xticks(range(len(layers)), layers, rotation="vertical", fontsize=8)
    plt.ylabel("average gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def feature_maps(model, dataloader, device, model_name="model"):
    "построение feature maps"
    images, c = next(iter(dataloader))
    image = images[0].unsqueeze(0).to(device)  

    with torch.no_grad():
        x = model.conv1(image)
        feature_maps = x.squeeze(0).cpu()

    num_maps = min(8, feature_maps.size(0))
    plt.figure(figsize=(15, 5))
    for i in range(num_maps):
        plt.subplot(1, num_maps, i+1)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'FM {i}')
    plt.suptitle(f'Feature Maps: {model_name}')
    plt.tight_layout()
    plt.show()
