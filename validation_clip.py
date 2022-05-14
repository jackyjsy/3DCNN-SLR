import torch
from sklearn.metrics import accuracy_score
from tools import wer
import pickle
import numpy as np
# exp_name = 'swish_smooth_dropout_pretrained_test'
# exp_name = 'swish_smooth_dropout_pretrained_all_test'
def val_epoch(model, criterion, dataloader, device, epoch, logger, writer, phase='Train', exp_name = None):
    model.eval()
    losses = []
    all_label = []
    all_pred = []
    score_frag = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # print('batching')
            # get the inputs and labels
            inputs_clips, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs_clips = []
            for i_clip in range(inputs_clips.size(1)):
                inputs = inputs_clips[:,i_clip,:,:]
                outputs = model(inputs)
                # print(outputs.size())
                outputs_clips.append(outputs)
                # if isinstance(outputs, list):
                #     outputs = outputs[0]
            outputs = torch.mean(torch.stack(outputs_clips, dim=0), dim=0)
            if phase == 'Test':
                score_frag.append(outputs.data.cpu().numpy())
            else:
                # compute the loss
                loss = criterion(outputs, labels.squeeze())
                losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
            if phase == 'Test':
                score = np.concatenate(score_frag)
            # acc = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
            if (batch_idx + 1) % 5 == 0:
                logger.info("epoch {:3d} | iteration {:5d}".format(epoch+1, batch_idx+1))
        if phase == 'Test':
            validation_loss = 0
        else:
            # Compute the average loss & accuracy
            validation_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        validation_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    if phase == 'Test':
        with open('./results/{}/results_epoch{:03d}_{}.pkl'.format(exp_name, epoch+1, validation_acc), 'wb') as f:
            score_dict = dict(zip(dataloader.dataset.sample_names, score))
            pickle.dump(score_dict, f)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100))
    return validation_loss


def val_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs = imgs.to(device)
            target = target.to(device)

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target = target.permute(1,0)[1:].reshape(-1)

            # compute the loss
            loss = criterion(outputs, target)
            losses.append(loss.item())

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
            all_trg.extend(target)
            all_pred.extend(prediction)

            # compute wer
            # prediction: ((trg_len-1)*batch_size)
            # target: ((trg_len-1)*batch_size)
            batch_size = imgs.shape[0]
            prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
            target = target.view(-1, batch_size).permute(1,0).tolist()
            wers = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))
            all_wer.extend(wers)

    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100, validation_wer))