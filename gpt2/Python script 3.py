from dataclasses import dataclass
import math
import torch
import tqdm


class CustomLoader():
    def __init__(self, data, B, T):
        self.data = data
        self.B = B
        self.T = T

    def load(self):
        number = len(self.data) - 1
        total = (number // (self.B * self.T)) * (self.B * self.T)

        x = self.data[:total]
        y = self.data[1:total + 1]

        num_batches = total // (self.B * self.T)
        x = x.view(num_batches, self.B, self.T)
        y = y.view(num_batches, self.B, self.T)

        return x, y

class FullLoader():
    def __init__(self, dataset, train_test_split):
        split = int(len(dataset) * train_test_split)
        self.train_data = dataset[:split]
        self.val_data = dataset[split:]

    def load(self, B, T):
        train_x, train_y = CustomLoader(self.train_data, B, T).load()
        val_x, val_y = CustomLoader(self.val_data, B, T).load()

        return train_x, train_y, val_x, val_y
    
def get_lr(min_lr, max_lr, max_step, current_step):
    lr = min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos((current_step / max_step) * math.pi)))
    return lr

def val_acc(val_x, val_y, model):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in zip(val_x, val_y):
            logits, _ = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total


def val_loss(val_x, val_y, model):
    total_loss = 0.0

    with torch.no_grad():
        for x, y in zip(val_x, val_y):
            _, loss = model(x, y)
            total_loss += loss.item()

    return total_loss / len(val_x)

def hella_swag_eval(model, tokenizer, dataset, max_samples=100):
    model.eval()
    correct = 0

    for example in tqdm(dataset.select(range(max_samples))):
        context = example["ctx_a"] + " " + example["ctx_b"]
        choices = example["endings"]
        label = example["label"]

        losses = []
        for choice in choices:
            full_input = context + " " + choice
            tokens = tokenizer.encode(full_input)

            if len(tokens) > model.config.block_size:
                tokens = tokens[:model.config.block_size]

            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            input_ids = tokens_tensor[:, :-1]
            target_ids = tokens_tensor[:, 1:]

            with torch.no_grad():
                _, loss = model(input_ids, target_ids)
            losses.append(loss.item())

        prediction = torch.tensor(losses).argmin().item()
        if prediction == label:
            correct += 1

    accuracy = correct / max_samples
    return accuracy