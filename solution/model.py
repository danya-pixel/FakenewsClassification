import os
import numpy as np
from sklearn.metrics import f1_score
import torch
from tqdm import trange
from tqdm.notebook import tqdm


def evaluate(device, model, dataloader, loss_fn):
    """Evaluate model on given DataLoader

    Args:
        device (CUDA or CPU): Device to evaluation
        model: PyTorch model
        dataloader: PyTorch DataLoader
        loss_fn: Loss function

    Returns:
        Tuple: eval_loss, correct_labels, predicted_labels
    """
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for _, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask,
                            token_type_ids=segment_ids, labels=label_ids)
        logits = outputs[1]

        eval_loss = loss_fn(outputs.logits, label_ids)

        label_ids = label_ids.to('cpu').numpy()
        outputs = np.argmax(logits.to('cpu'), axis=1)

        predicted_labels += list(outputs)
        correct_labels += list(label_ids)

        eval_loss += eval_loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)

    return eval_loss, correct_labels, predicted_labels


def train_model(device, model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, config) -> None:
    """Train model on given Dataloder

    Args:
        device (CUDA or CPU): Device to train
        model: PyTorch model
        train_dataloader: PyTorch DataLoader
        val_dataloader: PyTorch DataLoader
        loss_fn: Loss function
        optimizer: any optimizer
        scheduler: any scheduler
        config: NUM_TRAIN_EPOCHS, GRADIENT_ACCUMULATION_STEPS, MAX_GRAD_NORM, SAVED_MODEL_NAME
    """
    NUM_TRAIN_EPOCHS, GRADIENT_ACCUMULATION_STEPS, MAX_GRAD_NORM, SAVED_MODEL_NAME = config
    OUTPUT_DIR = "trained_models/"
    PATIENCE = 4
    loss_history = []
    F1_history = []
    no_improvement = 0
    for _ in trange(NUM_TRAIN_EPOCHS, desc="Epoch"):
        model.train()
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask,
                            token_type_ids=segment_ids, labels=label_ids)
            loss = outputs[0]

            _ = torch.argmax(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, label_ids)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        dev_loss, dev_correct, dev_predicted = evaluate(device,
                                                        model, val_dataloader, loss_fn)
        dev_F1 = f1_score(dev_predicted, dev_correct)

        if (len(loss_history) > 3):
            print(f"Loss history: {loss_history[-3:]}")
        else:
            print(f"Loss history: {loss_history}")
        print(f"Dev loss: {dev_loss}")
        print(f"Dev F1: {dev_F1}")

        if len(F1_history) == 0 or dev_F1 > max(F1_history):
            no_improvement = 0
            output_model_file = os.path.join(OUTPUT_DIR, SAVED_MODEL_NAME)

            print(f'New record, model saved to {output_model_file}')
            model.save_pretrained(output_model_file)

        elif dev_F1 < F1_history[-1]:
            no_improvement += 1

        if no_improvement > PATIENCE:
            print("No improvement on development set. Finish training.")
            break

        loss_history.append(dev_loss.item())
        F1_history.append(dev_F1)
