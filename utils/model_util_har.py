from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

def prune_model_train(num_epochs, model, optimizer, loss_func, train_loader, test_loader, device, adaptiveLR: bool = False):
    model.train()

    for param_group in optimizer.param_groups:
        init_lr = param_group['lr']

    for epoch in range(num_epochs):

        if adaptiveLR:
            update_learning_rate(init_lr, epoch, optimizer)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        for batch_idx, (data, labels) in pbar:
            b_x = data.type(torch.float)
            b_y = labels
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()

            done = batch_idx * len(data)
            percentage = 100. * batch_idx / len(train_loader)
            pbar.set_description(
                f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print(f'Epoch: {epoch}, Learning Rate: {lr},', end=' ')
        _ = prune_model_eval(model, loss_func, test_loader, device)


def prune_model_eval(model, loss_func, test_loader, device):
    model.eval()
    eval_acc = 0.0
    eval_loss = 0.0
    eval_top1 = 0.0
    eval_top2 = 0.0

    with torch.no_grad():
        for _, (data, labels) in enumerate(test_loader):
            b_x = data.type(torch.float)
            b_y = labels
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pred = output.argmax(dim=1)
            eval_loss += loss_func(output, b_y)
            eval_acc += torch.eq(pred, b_y).sum().float().item()

            x = topk_accuracy(output, b_y)
            eval_top1 += x[0].item()
            eval_top2 += x[1].item()

        eval_loss = eval_loss / len(test_loader.dataset)
        eval_acc = eval_acc * 100 / len(test_loader.dataset)
        eval_top1 = eval_top1 * 100 / len(test_loader.dataset)
        eval_top2 = eval_top2 * 100 / len(test_loader.dataset)
        print(f'Eval Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.3f}%')
        print(f'Top-1: {eval_top1:.3f}%, Top-2: {eval_top2:.3f}%')
        return eval_acc


def update_learning_rate(init_lr, epoch, optimizer):
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 2)) -> [torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk # / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]
