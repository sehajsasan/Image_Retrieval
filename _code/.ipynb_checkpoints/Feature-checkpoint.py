import torch
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F

def feature(dsets, model):
    Fvecs = []
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=128, sampler=SequentialSampler(dsets), num_workers=48)
    torch.set_grad_enabled(False)
    model.eval()
    model.cuda()
    for data in dataLoader:
        inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        fvec = model(inputs_bt.cuda())
        fvec = F.normalize(fvec, p = 2, dim = 1).cpu()
        Fvecs.append(fvec)
            
    return torch.cat(Fvecs,0)



# dataset = torch.load("data_dict_emb1.pth")
# dsets = ImageReader(dataset['test'], tra_transform)
# dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=48)
# torch.set_grad_enabled(False)
# model = models.resnet18(pretrained=True)
# print('Setting model: resnet18')
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, out_dim)
# checkpoint = torch.load("/notebooks/Image_Retrieval/EasyPositiveHardNegative-master/_result/EPSHN/HAR_R18/G16_lr0.03/model_state_dict.pth")
# model = model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# for data in dataLoader:
#     inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
#     fvec = model(inputs_bt.cuda())
#     fvec = F.normalize(fvec, p = 2, dim = 1).cpu()
#     Fvecs.append(fvec)


# def feature(dsets, model):
# Fvecs = []

# dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=48)
# torch.set_grad_enabled(False)
# model.eval()
# for data in dataLoader:
#     inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
#     fvec = model(inputs_bt.cuda())
#     fvec = F.normalize(fvec, p = 2, dim = 1).cpu()
#     Fvecs.append(fvec)

# return torch.cat(Fvecs,0)