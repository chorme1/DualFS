import numpy as np
import torch


def _update_confusion(conf, preds, labels, num_classes):
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    valid = (labels >= 0) & (labels < num_classes)
    labels = labels[valid]
    preds = preds[valid]
    indices = labels * num_classes + preds
    bincount = np.bincount(indices, minlength=num_classes * num_classes)
    conf += bincount.reshape(num_classes, num_classes)


def metrics_from_confusion(conf):
    num_classes = conf.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    iou = np.zeros(num_classes, dtype=np.float64)

    for cls in range(num_classes):
        tp = float(conf[cls, cls])
        fp = float(conf[:, cls].sum() - conf[cls, cls])
        fn = float(conf[cls, :].sum() - conf[cls, cls])

        precision[cls] = tp / (tp + fp + 1e-12)
        recall[cls] = tp / (tp + fn + 1e-12)
        f1[cls] = 2.0 * precision[cls] * recall[cls] / (precision[cls] + recall[cls] + 1e-12)
        iou[cls] = tp / (tp + fp + fn + 1e-12)

    total = float(conf.sum())
    oa = float(np.trace(conf)) / (total + 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "m_f1": float(np.mean(f1)),
        "m_iou": float(np.mean(iou)),
        "oa": oa,
    }


def format_per_class(values, prefix="C"):
    return " ".join(f"{prefix}{idx + 1}:{float(value):.4f}" for idx, value in enumerate(values))


def evaluate_fusion_classifier(model, dataloader, num_classes, device, domain="source"):
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            _, logits = model.forward_fusion(data, domain=domain)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()
            _update_confusion(conf, preds, labels_np, num_classes)

    metrics = metrics_from_confusion(conf)
    return metrics["m_f1"], metrics["m_iou"], conf


def evaluate_iou(model, dataloader, num_classes, device):
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            _, logits = model.forward_fusion(data, domain="target")
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()
            _update_confusion(conf, preds, labels_np, num_classes)

    metrics = metrics_from_confusion(conf)
    return {cls: float(metrics["iou"][cls]) for cls in range(num_classes)}
