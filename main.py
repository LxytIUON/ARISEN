from model.din import DeepInterestNetwork
from model.iv4rec import IV4Rec
from model.arisen import Arisen
from model.sequencemodel import SequenceModel
from model.arisen_sequencemodel import Arisen_SequenceModel
from model.caser import Caser
from model.arisen_caser import Arisen_Caser
from dataset.loader import get_dataloader
import config.config as config
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime
from utils import cal_metric_mind
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# din
#Config = config.DinConfig()
#model = DeepInterestNetwork()

# # iv4rec
# Config = config.IvDinConfig()
# model = IV4Rec()

# arisen
Config = config.autoIvDinConfig()
model = Arisen()

# sequence
# Config = config.SequenceModelConfig()
# model = SequenceModel()

# arisen_sequence
# Config = config.IV_SequenceModelConfig()
# model = Arisen_SequenceModel()

# # caser
# Config = config.CaserConfig()
# model = Caser()

# arisen_Caser
# Config = config.IV_CaserConfig()
# model = Arisen_Caser()

seed = 1024

if not os.path.isdir(Config.checkpoint_path[0:30]):
    os.makedirs(Config.checkpoint_path[0:30])
writer = SummaryWriter(Config.load_path)

def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

def train_step(model,dataloader):
    '''
    train step
    '''
    model.train()
    # 梯度清零
    model.optimizer.zero_grad()
    # 处理为model可以接受的格式
    if model.name == 'Din' or model.name == 'SequenceModel' or model.name =='Caser':
        data, label = dataloader[:3], dataloader[-1]
    else:
        data, label = dataloader[:-1], dataloader[-1]

    label = label.to(model.device)

    if model.name == 'Iv4Rec' or model.name == 'Din' or model.name == 'SequenceModel' or model.name =='Caser':
        pred = model(data)

    elif model.name == 'Arisen' or model.name == 'Arisen_SequenceModel' or model.name =='Arisen_Caser':
        pred, miloss = model(data)
        mi_loss = miloss * Config.mi_weight

    # 计算loss
    loss = model.loss_func(pred, label)
    reg_loss = model.get_regularization_loss()
    if model.name == 'Iv4Rec' or model.name == 'Din' or model.name == 'SequenceModel' or model.name =='Caser':
        all_loss = loss + reg_loss
    elif model.name == 'Arisen' or model.name == 'Arisen_SequenceModel' or model.name =='Arisen_Caser':
        if Config.mi_loss:
            all_loss = mi_loss + loss + reg_loss
        else:
            all_loss = loss + reg_loss

        if Config.connection_type == 'z':
            reg = 1e-6
            with torch.enable_grad():
                """正则化的手段"""
                orth_loss_A, orth_loss_B = torch.zeros(1).to(Config.device), torch.zeros(1).to(Config.device)
                for name, param in model.bridge1.named_parameters():
                    if 'bias' not in name:
                        param_flat = param.view(param.shape[0], -1)
                        sym = torch.mm(param_flat, torch.t(param_flat))
                        sym -= torch.eye(param_flat.shape[0]).to(Config.device)
                        orth_loss_A = orth_loss_A + (reg * sym.abs().sum())
                        orth_loss_A = Config.z_weight * orth_loss_A
                orth_loss_A.backward()
                for name, param in model.bridge2.named_parameters():
                    if 'bias' not in name:
                        param_flat = param.view(param.shape[0], -1)
                        sym = torch.mm(param_flat, torch.t(param_flat))
                        sym -= torch.eye(param_flat.shape[0]).to(Config.device)
                        orth_loss_B = orth_loss_B + (reg * sym.abs().sum())
                        orth_loss_B = Config.z_weight * orth_loss_B
                orth_loss_B.backward()

    all_loss.backward()
    model.optimizer.step()

    return all_loss

def valid_step(model,dataloader):
    '''
    test step
    '''
    model.eval()
    with torch.no_grad():
        # 处理为model可以接受的格式
        if model.name == 'Din' or model.name == 'SequenceModel' or model.name =='Caser':
            dataloader_data, dataloader_label,  dataloader_impression= dataloader[:3], dataloader[5], dataloader[-1]
        else:
            dataloader_data, dataloader_label, dataloader_impression = dataloader[:5], dataloader[5], dataloader[-1]
        dataloader_label = dataloader_label.to(model.device)

        if model.name == 'Iv4Rec' or model.name == 'Din' or model.name == 'SequenceModel' or model.name =='Caser':
            pred = model(dataloader_data)
        elif model.name == 'Arisen' or model.name == 'Arisen_SequenceModel'or model.name =='Arisen_Caser':
            pred, miloss = model(dataloader_data)
            mi_loss = miloss * Config.mi_weight

        label = dataloader_label
        # 计算loss
        loss = model.loss_func(pred, label)
        reg_loss = model.get_regularization_loss()
        if model.name == 'Iv4Rec' or model.name == 'Din' or model.name == 'SequenceModel' or model.name =='Caser':
            all_loss = loss + reg_loss
        elif model.name == 'Arisen'or model.name == 'Arisen_SequenceModel' or model.name =='Arisen_Caser':
            if Config.mi_loss:
                all_loss = mi_loss + loss + reg_loss
            else:
                all_loss = loss + reg_loss

        pred = pred.squeeze(-1).tolist()
        label = label.squeeze(-1).tolist()

    return pred,label,dataloader_impression,all_loss

def train(model,epochs,train_loader,val_loader):
    '''
    训练模型
    验证模型，计算评价指标【auc，hit@1, hit@5, hit@10, ndcg@1, ndcg@5，ndcg@10，mrr】
    '''
    model_name = Config.description
    dfhistory = pd.DataFrame(columns=['epoch', 'loss_total', 'val_loss_total', 'auc', 'hit@1', 'hit@5', 'hit@10', 'ndcg@1', 'ndcg@5','ndcg@10', 'mrr'])

    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 3 + "%s" % nowtime + "==========" * 3 )

    for epoch in range(Config.start_epoch, epochs):
        # 训练模型
        total_loss = 0

        for step, dataloader in enumerate(train_loader):
            # all_loss, mi_loss, loss ,reg_loss = train_step(model,dataloader)
            all_loss = train_step(model, dataloader)
            total_loss += all_loss.item()
            # 使用tensorboard记录数据
            writer.add_scalar('Train Total Loss', total_loss / (step + 1), (epoch - 1) * len(train_loader) + (step + 1))
            # 打印数据
            if step % Config.interval == 0 and step > 0:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # print('mi loss:',mi_loss, 'model loss:',loss, 'reg loss:',reg_loss)
                print( "%s" % nowtime+ "---", model_name+":","epoch {:d} , step {:d} , loss: {:.4f}".format(epoch+1, step, total_loss / (step + 1)))

        preds = {}
        labels = {}
        total_val_loss = 0
        # 验证模型
        with torch.no_grad():
            for val_step, dataloader in enumerate(val_loader):
                # pred, label,dataloader_impression,val_loss, mi_loss, loss ,reg_loss = valid_step(model, dataloader)
                pred, label, dataloader_impression, val_loss= valid_step(model, dataloader)
                total_val_loss += val_loss.item()
                for i, imp in enumerate(dataloader_impression):
                    if preds.__contains__(imp.item()) == False:
                        preds[imp.item()] = []
                        labels[imp.item()] = []
                    preds[imp.item()].append(pred[i])
                    labels[imp.item()].append(label[i])

        # 计算评价指标
        # res:{"auc':     ,"ndcg@5":     ,"ndcg@10":     ,"mrr":     }
        res = cal_metric_mind(preds, labels)
        print(res)

        info = (epoch + 1, total_loss / (step + 1), total_val_loss / (val_step + 1), res['auc'], res['hit@1'], res['hit@5'],res['hit@10'], res['ndcg@1'], res['ndcg@5'], res['ndcg@10'], res['mrr'])
        dfhistory.loc[epoch+1] = info
        nowtime = nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 6 + "%s" % nowtime)
        dfhistory.to_csv(Config.csv_path)

        # 保存模型
        torch.save(model.state_dict(), Config.checkpoint_path + "epoch_{}.pth".format(epoch + 1))

    writer.close()
    print("Finished Training...")
    return dfhistory

# import torchsnooper
# @torchsnooper.snoop()
def main():
    model.to(device=Config.device)
    model.optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    # model.loss_func = nn.BCELoss(reduction='sum')
    model.loss_func = nn.BCELoss()

    print("运行模型：",model.name,Config.description)
    if Config.description == 'AutoIV_Din':
        print('connection方式：', Config.connection_type)

    print("学习率：",Config.learning_rate)
    # 准备train数据和test数据
    train_dataloader = get_dataloader(Config.train_batch_size, 'train')
    print("Train Dataloader运行结束！")
    val_dataloader = get_dataloader(Config.test_batch_size, 'test')
    print("Valid Dataloader运行结束！")

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(Config.checkpoint_path + "epoch_{}.pth".format(Config.start_epoch)):
        checkpoint = torch.load(Config.checkpoint_path + "epoch_{}.pth".format(Config.start_epoch))
        model.load_state_dict(checkpoint)
        print('加载 epoch {} 成功！'.format(Config.start_epoch))
        dfhistory = train(model, epochs=Config.epochs, train_loader=train_dataloader, val_loader=val_dataloader)
    else:
        Config.start_epoch = 0
        print('无保存模型，将从头开始训练！')
        dfhistory = train(model, epochs=Config.epochs, train_loader=train_dataloader, val_loader=val_dataloader)

    dfhistory.to_csv(Config.csv_path)
    print("文件保存成功")

if __name__ == '__main__':
    '''
    在pycharm运行程序
    '''
    set_seed(seed)
    main()
    # model = DTCDR()
    # print(model(user = torch.tensor(1),item = torch.tensor(10),domain = 'A'))
