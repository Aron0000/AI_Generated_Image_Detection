from __future__ import annotations

import torch


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = correct = count = 0
    predictions, targets, probabilities = [], [], []

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        predicted = logits.argmax(dim=1)

        # Probability of class 1 = AI-generated image
        probs = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * labels.size(0)
        correct += (predicted == labels).sum().item()
        count += labels.size(0)

        predictions.extend(predicted.cpu().tolist())
        targets.extend(labels.cpu().tolist())
        probabilities.extend(probs.detach().cpu().tolist())

    return {
        "loss": total_loss / max(count, 1),
        "accuracy": correct / max(count, 1),
        "predictions": predictions,
        "targets": targets,
        "probabilities": probabilities,
        "count": count,
    }
