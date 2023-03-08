import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from resnet import ResNetModel, PreResNetModel, SL_ResModel
from sam import SAM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# Dataset Creat
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))  # Length of dataset

    def __getitem__(self, index):
        path, target = self.samples[index]
        t = transforms.ToTensor()  # Transform
        sample = self.loader(path)  # Load image


        if self.transform is not None:
            sample1 = self.transform(sample)
            sample2 = self.transform(sample)
            return sample1, sample2, t(sample), target
        else:
            return t(sample),  target

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()
    

# SimCLR Loss
def xt_xent(u, v, temperature=0.5):
    N = u.shape[0]

    z = torch.cat([u, v], dim=0)

    z = F.normalize(z, p=2, dim=1)
    s = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * N).bool().to(z.device)
    s = torch.masked_fill(s, mask, -float('inf'))
    label = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(N)]).to(z.device)

    loss = F.cross_entropy(s, label)

    return loss

# KNN
def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))

    for batch_x in torch.split(emb, batch_size):

        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")

        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)

    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]

    return max(accs)

# Train moudule
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)



# SSL Pretrain
def PreTrainSSL(train_dataloader, test_dataloader, num_epochs, criterion):
    best_acc = 0
    min_loss = 1
    t = 0.5

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Convert model parameters and buffers to CPU or Cuda
    model = ResNetModel('Resnet').to(device)

    # Set SGD as optimizer with ASAM
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=2,
                    adaptive=True, lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(
        train_dataloader), epochs=num_epochs, anneal_strategy='cos')

    #Training
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for _, (imgs1, imgs2, _, _) in enumerate(tqdm(train_dataloader)):

            # optimizer.zero_grad()
            enable_running_stats(model)
            # compute the loss based on model output and real labels

            outputs1 = model(imgs1.to(device))
            outputs2 = model(imgs2.to(device))

            loss = criterion(outputs1, outputs2, t)
            # backpropagate the loss
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)
            # nn.utils.clip_grad_norm_(model.parameters(), 1e-3, norm_type=2)
            # adjust parameters based on the calculated gradients
            disable_running_stats(model)

            criterion(model(imgs1.to(device)), model(
                imgs2.to(device)), t).mean().backward()
            optimizer.second_step(zero_grad=True)

            # optimizer.step()
            scheduler.step()

            running_loss += loss.item()     # extract the loss value

        # Caculate loss of each epoch
        curr_loss = running_loss / len(train_dataloader.dataset)

        # Dymanic rescale t
        if (curr_loss) < 0.045:
            t = 0.3
        if (curr_loss) < 0.03:
            t = 0.1
        if (curr_loss) < 0.02:
            t = 0.05

        # Save Model
        if curr_loss < min_loss:
            min_loss = curr_loss
            torch.save(model.state_dict(), 'best_cnn.pth')

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / len(train_dataloader.dataset)))

        torch.cuda.empty_cache()

        # Test with KNN
        model.eval()
        with torch.no_grad():

            embedding = torch.tensor([])
            classes = torch.tensor([])

            for _, _, imgs, label in test_dataloader:
                pred = model(imgs.float().to(device)).cpu()
                embedding = torch.cat(
                    [embedding, pred.view(imgs.shape[0], -1)])
                classes = torch.cat([classes, label], dim=0)

            acc = KNN(embedding.cpu(), classes.cpu(), batch_size=256)

            print("[%d] Accuracy: %.5f" % (epoch + 1, acc))

# Fine-tune (Semi-Supervised)
def FineTuneTrain(train_dataloader, test_dataloader, num_epochs, data_size, lr, mode, params_model):
    best_acc = 0
    path = None

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Load model with parameters
    model = None
    if mode == 'SL_without_pretrain':
        model = SL_ResModel('Resnet', params_model['Pretrain'][0]).to(
            device)  # for sl
        path = './SL weight/'

    # elif mode == 'SL_with_pretrain':
    #     model = SL_ResModel(
    #         'Resnet', params_model['Pretrain'][1]).to(device)
    #     path = './SL pretrain model weight/'

    elif mode == 'SSL_pretrain':
        model = PreResNetModel(
            'resnet50', params_model['SSL_Pretrain_Weight']).to(device)  # for ssl
        path = './SSL model weight/'

    print(mode)

    # Set optimizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-8)

    # Training

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        accuracy = 0.0
        total = 0.0
        criterion = nn.CrossEntropyLoss()

        for _, (_, _, img, labels) in enumerate(tqdm(train_dataloader)):
            img = img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(img)
            # compute the loss based on model output and real labels
            loss = criterion(outputs,  labels)
            # backpropagate the loss
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1e-3, norm_type=2)
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # extract the loss value
            running_loss += loss.item()     
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

        print('[%d] loss: %.3f, acc : %.3f' %
              (epoch + 1, running_loss / total, accuracy / total))

        # torch.cuda.empty_cache()
        
        
        # Test Evaluation

        model.eval()

        with torch.no_grad():
            test_loss = 0.0
            test_accuracy = 0.0
            test_total = 0

            for _, (_, _, img, labels) in enumerate(tqdm(test_dataloader)):
                img_test = img.to(device)
                labels_test = labels.to(device)
                outputs_test = model(img_test)

                # print(pred)
                # print(labels.to(device))
                loss = criterion(outputs_test, labels_test)
                test_loss += loss.item()
                _, predicted_test = torch.max(outputs_test.data, 1)
                test_total += labels_test.size(0)
                test_accuracy += (predicted_test == labels_test).sum().item()

            # print(accuracy)
            # print(len(test_loader.dataset))

            # Save Best model
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                model_name = path + 'best_cnn_' + str(data_size) + '.pth'
                torch.save(model.state_dict(), model_name)
            print('[%d] loss: %.3f, acc : %.3f' % (
                epoch + 1, test_loss / test_total, test_accuracy / test_total))

    print('Best acc : %.3f' % (best_acc/test_total))


def Prediction(pred_dataloader, mode, weight_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Load model
    model = None

    if mode == 'SL_without_pretrain':
        weight_model_path = './SL weight/' + weight_model
        print('Load model from : ', weight_model_path)
        model = SL_ResModel('Resnet', False).to(
            device)  # for sl
        model.load_state_dict(torch.load(weight_model_path))


    elif mode == 'SSL_pretrain':
        weight_model_path = './SSL model weight/'+ weight_model
        print('Load model from : ',weight_model_path)
        model = PreResNetModel(
            'resnet50 SSL', 'best_cnn.pth').to(device)  # for ssl
        model.load_state_dict(torch.load(weight_model_path))


    # Predict
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pred_loss = 0.0
        pred_accuracy = 0.0
        pred_total = 0

        pred_labels = []

        target_labels = []

        for _, (img, labels) in enumerate(tqdm(pred_dataloader)):
            
            img_test = img.to(device)
            labels = labels.to(device)
            outputs_test = model(img_test)

            loss = criterion(outputs_test, labels)
            pred_loss += loss.item()
            _, preds = torch.max(outputs_test.data, 1)
            pred_total += labels.size(0)
            pred_accuracy += (preds == labels).sum().item()

            pred_labels.extend(preds.tolist())
            target_labels.extend(labels.tolist())


        print('Loss : %.3f' % (pred_loss/pred_total))
        print('Accuracy : %.3f' % (pred_accuracy/pred_total))


        cnf_matrix = confusion_matrix(target_labels, pred_labels, labels=[0, 1, 2, 3])
        np.set_printoptions(precision=2)
         

        # Plot non-normalized confusion matrix
        # plt.figure()
        plt.figure(dpi=120)
        plot_confusion_matrix(cnf_matrix, classes=['Mild_Demented',  'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented'],
                      title='Confusion matrix, without normalization')


