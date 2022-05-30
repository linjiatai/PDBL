import os
import argparse
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import numpy as np
from tool.dataset import ImageDataset
from tool.PDBL import PDBL_net
from tool.eff import EfficientNet
from tool.resnet import resnet50
from tool.shufflenet import shufflenet_v2_x1_0
from sklearn.metrics import accuracy_score,f1_score

def create_model(model_name, n_class):
    if model_name == 'shuff':
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model._stage_out_channels[-1], n_class)
    elif model_name == 'eff':
        model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=n_class)
    elif model_name == 'r50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(512 * model.block.expansion, n_class)
    return model

def save_PDBL(PDBL, save_path, fname):
    with open(os.path.join(save_path, fname), 'wb') as f:
        pickle.dump(PDBL, f)
def load_PDBL(file_dir):
    with open(file_dir, 'rb') as f:
        PDBL = pickle.load(f)
    return PDBL

def train_PDBL(model,dataloader_train,args,PDBL):
    model.eval()
    steps = len(dataloader_train)
    dataiter_train = iter(dataloader_train)
    print('Training phase ---> number of training items is: ', args.n_item_train)
    work_space_in = np.zeros((args.n_item_train,args.n_feature))
    work_space_out = np.zeros((args.n_item_train,args.n_class))
    progress = 0
    for step in tqdm(range(steps)):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        img224, img160, img112, target_train = next(dataiter_train)
        len_batch = len(target_train)
        with torch.no_grad():
            img224 = Variable(img224.float().cuda())
            img160 = Variable(img160.float().cuda())
            img112 = Variable(img112.float().cuda())
        feature1, _ = model(img224)
        feature2, _ = model(img160)
        feature3, _ = model(img112)

        feature = feature1
        feature = torch.cat((feature,feature2),1)
        feature = torch.cat((feature,feature3),1)
        
        work_space_in[  progress:(progress+len_batch),  :] = feature.detach().cpu().numpy()
        work_space_out[ progress:(progress+len_batch),  :] = target_train.detach().cpu().numpy()
        progress = progress+len_batch
    PDBL.train(work_space_in, work_space_out)
    return PDBL

def valid_epoch(model, dataloader_valid, args, PDBL):
    model.eval()
    steps = len(dataloader_valid)
    dataiter_valid = iter(dataloader_valid)
    print('Validation phase ---> number of val items is: ', args.n_item_val)
    work_space_in = np.zeros((args.n_item_val,args.n_feature))
    work_space_out = np.zeros((args.n_item_val,args.n_class))
    progress = 0
    for step in range(steps):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        img224, img160, img112, target_valid = next(dataiter_valid)
        len_batch = len(target_valid)

        with torch.no_grad():
            img224 = Variable(img224.float().cuda())
            img160 = Variable(img160.float().cuda())
            img112 = Variable(img112.float().cuda())
        feature1, _ = model(img224)
        feature2, _ = model(img160)
        feature3, _ = model(img112)
        feature = feature1
        feature = torch.cat((feature,feature2),1)
        feature = torch.cat((feature,feature3),1)
        work_space_in[progress:(progress+len_batch),:] = feature.cpu().detach().numpy()
        work_space_out[progress:(progress+len_batch),:] = target_valid.cpu().detach().numpy()
        progress = progress+len_batch
    ##  PDBL prediction
    ##  In the prediction phase, PDBL can also make predictions sample by sample.
    PDBL_output =PDBL.predict(work_space_in)
    PDBL_pred = np.zeros(PDBL_output.shape[0])
    PDBL_lab = np.zeros(PDBL_output.shape[0])
    for i in range(len(PDBL_output)):
        PDBL_pred[i] = np.argmax(PDBL_output[i])
        PDBL_lab[i] = np.argmax(work_space_out[i])
    PDBL_acc = accuracy_score(PDBL_lab,PDBL_pred)
    PDBL_f1 = f1_score(PDBL_lab,PDBL_pred,average='macro')
    
    print('Accuracy of PDBL: ', PDBL_acc)
    print('F1 score of PDBL: ', PDBL_f1)
    
def main(args):
    ##  Training set and Validation set
    train = ImageDataset(data_path = args.traindir, n_class=args.n_class)
    val = ImageDataset(data_path = args.valdir, n_class=args.n_class)
    trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    valloader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    args.n_item_train = len(train)
    args.n_item_val = len(val)
    model_names = ['shuff','eff','r50']
    n_features = [812* 3, 1456* 3, 3840* 3]
    
    for model_index in range(len(model_names)):
        args.model_name = model_names[model_index]
        args.n_feature = n_features[model_index]

        print('<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('This model is', args.model_name)
        ##  Load PDBL model
        reg = 0.001
        n_components = min(int(args.n_item_train*0.9),  2000)
        PDBL = PDBL_net(isPCA=True, n_components = n_components, reg=reg)
        ##  Load CNN model
        ##  The first time you run this code will take some time to download the ImageNet pretrained models.
        model = create_model(model_name=args.model_name,n_class=args.n_class)
        model = model.cuda()

        ##  training and test phase
        PDBL = train_PDBL(model, trainloader, args, PDBL)
        valid_epoch(model, valloader, args, PDBL)

        ##  You can save or re-load the trained PDBL by following codes.
        if args.save_dir is not None:
            fname = 'PDBL_on_'+model_names[model_index]+'.pkl'
            save_PDBL(PDBL, args.save_dir, fname)
            ##  Try to reload the saved PDBL
            PDBL_reload = load_PDBL(os.path.join(args.save_dir, fname))
            # valid_epoch(model, valloader, args, PDBL_reload)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pyramidal Deep-Broad Learning')
    parser.add_argument('--device',     default='0',        type=str, help='index of GPU')
    parser.add_argument('--save_dir',   default='save/',    type=str, help='Save path of learned PDBL')
    parser.add_argument('--traindir',   default='dataset/KMI_001/',     type=str, help='Path of training set')
    parser.add_argument('--valdir',   default='dataset/KME/',           type=str, help='Path of validation set/test set')
    parser.add_argument('--batch_size', default=20,         type=int, help='Batch size of dataloaders')
    parser.add_argument('--n_class',    default=9,          type=int, help='Number of categories')
    parser.add_argument('--n_workers',  default=8,          type=int, help='Number of workers')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
