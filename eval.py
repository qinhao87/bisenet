from dataset.Crack import crack
import torch
import argparse
import os,cv2
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class test_dataset(crack):
    def __init__(self, image_path, label_path, loss='dice', mode='train'):
        super().__init__(image_path,label_path)
        # import ipdb;ipdb.set_trace()
        self.mode = mode
        self.image_list = [os.path.join(image_path,image_name) for image_name in os.listdir(image_path)]

        self.image_list.sort()
        if label_path is None:
            self.label_list = []
        else:
            self.label_list = [os.path.join(label_path,label_name) for label_name in os.listdir(label_path)]
            self.label_list.sort()

        self.T = A.Compose([
            A.PadIfNeeded(min_height=600,min_width=600,
                          border_mode=cv2.BORDER_CONSTANT,value=0),
                          ToTensorV2(),
        ])

        self.loss = loss
    
    def __getitem__(self, index):

        img = cv2.imread(self.image_list[index],cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_list[index],cv2.IMREAD_GRAYSCALE)

        # if self.mode == 'train':
        h,w = img.shape
        if h>600 or w>600:
            # import ipdb;ipdb.set_trace()
            resize_t = A.LongestMaxSize(max_size=600)
            img,label = resize_t(image=img,mask=label)['image'],resize_t(image=img,mask=label)['mask']
            label[label>0] = 1
        transformed = self.T(image=img,mask=label)
        img,label = transformed['image'] / 255.0,transformed['mask']
            

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            not_label = torch.logical_not(label)
            label = torch.stack([not_label,label],dim=0)

            return img, label.long() ,self.image_list[index].split('/')[-1]

        elif self.loss == 'crossentropy':

            return img, label.long() ,self.image_list[index].split('/')[-1]

    def __len__(self):
        return len(self.image_list)


def eval(model,dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict)
            # predict = colour_code_segmentation(np.array(predict), label_info)

            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)
            # label = colour_code_segmentation(np.array(label), label_info)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou_dict, miou = cal_miou(miou_list, csv_path)
        print('IoU for each class:')
        for key in miou_dict:
            print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision

def estimate(model,dataloader,args):
    # import ipdb;ipdb.set_trace()
    print('start test')
    color = np.array([[0,0,0],[0,0,255]])
    with torch.no_grad():
        model.eval()
        precision_rec = []
        min_pre,max_pre = 1,0
        min_name,max_name = '',''
        for i,(img,mask,name) in tqdm.tqdm(enumerate(dataloader)):
            if torch.cuda.is_available() and args.use_gpu:
                img,mask = img.cuda(),mask.cuda()
            predict = model(img).squeeze()
            # import ipdb;ipdb.set_trace()
            predict = reverse_one_hot(predict).cpu().detach().numpy()
            mask    = reverse_one_hot(mask).cpu().detach().numpy()
            b,h,w = predict.shape
            show_pic = np.zeros((b,h,w*2,3),dtype=np.uint8)
            
            res,label = color[predict],color[mask]
            show_pic[:,:,:w,:],show_pic[:,:,w:,:] = label,res
            for i in range(b):
                precision = compute_global_accuracy(predict,mask)
                # import ipdb;ipdb.set_trace()
                save_name = os.path.join(args.save_dir,name[i].split('\\')[-1])
                cv2.imwrite(save_name,show_pic[i])
                precision_rec.append(precision)
                if precision>max_pre:
                    max_pre = precision
                    max_name = name[i]
                if precision<min_pre:
                    min_pre = precision
                    min_name = name[i]
        
        mean_precision = np.mean(precision_rec)
        print('precision for test: %.3f' % mean_precision)
        print(f'min_name:{min_name} ,max_name:{max_name}')

            
            



def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--save_dir', type=str, default='/path/to/data', help='Path of save img')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # create dataset and dataloader
    # import ipdb;ipdb.set_trace()
    test_path = os.path.join(args.data, 'train')
    test_label_path = os.path.join(args.data, 'label')
    dataset = test_dataset(test_path, test_label_path, mode='test')
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # get label info
    # label_info = get_label_info(csv_path)
    # test
    estimate(model, dataloader, args)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', r'F:\py_pro\BiSeNet\checkpoints_18_sgd\latest_dice_loss.pth',
        '--data', r'F:\py_pro\BiSeNet\img',
        '--cuda', '0',
        '--context_path', 'resnet18',
        '--num_classes', '2',
        '--save_dir', r'F:\py_pro\BiSeNet\save_img'
    ]
    main(params)