import os
import numpy as np
import torch
from tqdm import trange
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support


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


def train(device, model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, config) -> None:
    """Train model on given Dataloder

    Args:
        device (CUDA or CPU): Device to train
        model: PyTorch model
        train_dataloader: PyTorch DataLoader
        val_dataloader: PyTorch DataLoader
        loss_fn: Loss function
        optimizer: _description_
        scheduler: _description_
        config: NUM_TRAIN_EPOCHS, GRADIENT_ACCUMULATION_STEPS, MAX_GRAD_NORM, SAVED_MODEL_NAME
    """
    NUM_TRAIN_EPOCHS, GRADIENT_ACCUMULATION_STEPS, MAX_GRAD_NORM, SAVED_MODEL_NAME = config
    OUTPUT_DIR = "trained_models/"
    PATIENCE = 4
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_history = []
    acc_history = []
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
        dev_acc = np.mean(dev_predicted == dev_correct)

        if (len(loss_history) > 3):
            print(f"Loss history: {loss_history[-3:]}")
        else:
            print(f"Loss history: {loss_history}")
        print(f"Dev loss: {dev_loss}")
        print(f"Dev accuracy: {dev_acc}")

        if len(acc_history) == 0 or dev_acc > max(acc_history):
            print('New record, model saved')
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, SAVED_MODEL_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        elif dev_acc < acc_history[-1]:
            no_improvement += 1

        if no_improvement > PATIENCE:
            print("No improvement on development set. Finish training.")
            break

        loss_history.append(dev_loss.item())
        acc_history.append(dev_acc)


def val_report(device, model, train_dataloader, val_dataloader, loss_fn):
    """Get report for best model

    Args:
        device: Device to evaluation
        model: PyTorch model
        train_dataloader: PyTorch DataLoader
        val_dataloader: PyTorch DataLoader
        loss_fn: Loss function

    Returns:
        dev_correct, dev_predicted
    """
    model.to(device)
    model.eval()

    _, train_correct, train_predicted = evaluate(device,
                                                 model, train_dataloader, loss_fn)
    _, dev_correct, dev_predicted = evaluate(
        device, model, val_dataloader, loss_fn)

    print("Training performance:", precision_recall_fscore_support(
        train_correct, train_predicted, average="micro"))
    print("Development performance:", precision_recall_fscore_support(
        dev_correct, dev_predicted, average="micro"))

    np.mean(dev_predicted == dev_correct)

    print(classification_report(dev_correct, dev_predicted))
    return (dev_correct, dev_predicted)
