import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import h5py

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# huggingface
from transformers import CLIPProcessor, CLIPModel

def ids_to_multinomial(ids):
    """
    Multi-label one-hot label encoding

    Argument
        ids (list): a list of class indices
    Output
        y (ndarray): (num of classes, ), multi-label one-hot labels, e.g. [1,0,1,0,0,...]
    """

    categories = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane', 'Race car, auto racing', 'Female speech, woman speaking',
              'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)',
              'Truck', 'Shofar', 'Motorcycle', 'Acoustic guitar',
              'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry',
              'Bus', 'Chainsaw', 'Cat', 'Horse',
              'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin']
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))

    index = id_to_idx[ids]
    y[index] = 1
    return y


class LLP_dataset(Dataset):

    def __init__(self, label_txt, video_frame_dir, clip_processor):

        with open(label_txt, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()

        self.video_frame_dir = video_frame_dir
        self.img_size = 224
        self.img_mean = clip_processor.image_processor.image_mean
        self.img_std = clip_processor.image_processor.image_std

        self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((self.img_size, self.img_size)),
                                                transforms.Normalize(self.img_mean, self.img_std)])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        lines = self.lines
        line = lines[idx]
        index_find = line.find("&")
        video_name = line[index_find+1:index_find+12]


        image_list = sorted(glob.glob(os.path.join(self.video_frame_dir, video_name, '*.jpg')))
        samples = np.round(np.linspace(0, len(image_list) - 1, 10))

        image_list = [image_list[int(sample)] for sample in samples]
        video_frames = []
        for iImg in range(len(image_list)):
            img = Image.open(image_list[iImg]).convert('RGB')
            video_frames.append(self.transform(img))
        video_frames = torch.stack(video_frames)


        ids = line[:index_find]
        label = ids_to_multinomial(ids)
        sample = {'label': label, 'video_name': video_name, 'video_frames': video_frames}

        return sample


def calculate_visual_logits(model, video_frames, event_captions, device):
    """
    Calculate the event logits of each image from the constructed event captions and raw images

    Arguments
        model (nn.Module): a pre-trained CLIP
        video_frames (tensor): size = (10, 3, H, W), video frames (1fps)
        event_captions (list): a list containing event captions in the same order as categories
        device (torch.device): gpu/cpu

    Output
        logits_image_text (tensor): size = (10, 25)
    """

    with torch.no_grad():

        inputs_img = {key: value for key, value in event_captions.items()}
        inputs_img['pixel_values'] = video_frames.to(device)
        img_features = model.get_image_features(pixel_values = video_frames.to(device))
        fusion_features = img_features
        text_features = model.get_text_features(**event_captions)

        fusion_features = fusion_features / fusion_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_image_text = torch.matmul(fusion_features, text_features.T).squeeze(0)
        # outputs = model(**inputs_img)
        # logits_image_text = outputs.logits_per_image

        return logits_image_text


def get_segment_pseudo_labels(logits_image_text, thresholds, labels, device):
    """
    Construct segment-level visual pseudo labels (multi-label one-hot pseudo labels)
    based on the pre-defined thresholds, logits, and the video-level labels

    Output
        Pv (tensor): size = (10, 25), segment-level visual pseudo labels
    """

    with torch.no_grad():
        occurred_event_idx = labels.nonzero().squeeze(dim=-1)
        occurred_events_logits = logits_image_text[:, occurred_event_idx]
        class_thresholds = thresholds[:, occurred_event_idx]
        segment_preds = torch.where(occurred_events_logits > class_thresholds, 1, 0).long()     # (10, k) in one-hot labels

        Pv = torch.zeros(10, 28).long().to(device)
        Pv[:, occurred_event_idx.long()] = segment_preds

    return Pv


def visual_label_elaboration(model, clip_processor, data_loader, event_captions, thresholds, save_path, device, print_progress):
    '''
    Visual label elaboration in VALOR

    Arguments
        model (nn.Module): a pre-trained CLAP model
        clip_processor:
        data_loader (DataLoader): a data loader for training, validation, or testing split
        event_captions (list): a list containing all event captions (with the prompt added) in the same order as in categories
        thresholds (list): a list containing the threshold for each event
        save_path (str): the directory where the pseudo labels are going to be saved
        device (torch.device): gpu/cpuc
    '''

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('# of data =', len(data_loader))

    event_captions = clip_processor(text=event_captions, images=None, return_tensors="pt", padding=True)
    event_captions = {key: value.to(device) for key, value in event_captions.items()}

    descrption = dict()
    with open("Text.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            v_name, desp = line.split('&')[0] ,line.split('&')[1]
            descrption[v_name] = desp

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            labels, video_frames = batch_data['label'].to(device).squeeze(), batch_data['video_frames'].to(device).squeeze()
            video_name = batch_data['video_name'][0]

            logits_image_text = calculate_visual_logits(model, video_frames, event_captions, device)
            seg_pseudo_labels = get_segment_pseudo_labels(logits_image_text, thresholds, labels, device)

            np.save(os.path.join(save_path, video_name +'.npy'), seg_pseudo_labels.cpu().numpy())

            if print_progress:
                print('Progress: {}/{}\r'.format(batch_idx+1, len(data_loader)), end='')
            if batch_idx%100==0:
                print(batch_idx)
        print('')

    return

def get_image_embedding(model, data_loader, save_path, device, print_progress, clip_processor):
    """
    Extract image embeddings from visual frames

    Arguments:
        model: a pre-trained CLIP
        data_loader (DataLoader):
        save_path (str): the directory where image features are going to be saved
        device (torch.device): gpu/cpu
    """

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('# of data =', len(data_loader))
    descrption = dict()
    with open("Text.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            v_name, desp = line.split('&')[0], line.split('&')[1]
            descrption[v_name] = desp
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):

            video_frames = batch_data['video_frames'].to(device).squeeze()
            video_name = batch_data['video_name'][0]

            inputs_img = {'pixel_values': video_frames.to(device)}
            image_features = model.get_image_features(**inputs_img)
            if len(descrption[video_name]) > 300:
                descrption[video_name] = descrption[video_name][:300]
            prompt = clip_processor(text = descrption[video_name], return_tensors="pt", padding=True).to(device)
            try:
                prompt_features = model.get_text_features(**prompt)
            except:
                print(video_name)
                prompt_features = torch.zeros((1, 768)).cuda()


            prompt_features = prompt_features / prompt_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            np.save(os.path.join(save_path, video_name[:11]+'.npy'), image_features.cpu().numpy())
            np.save(os.path.join(save_path, video_name[:11] + '_text.npy'), prompt_features.cpu().numpy())

            if print_progress:
                print('Progress: {}/{}\r'.format(batch_idx+1, len(data_loader)), end='')
        print('')


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='harvesting segment-level visual pseudo labels with CLIP')

    parser.add_argument("--video_frame_dir", type=str, default='data/video_frames')
    parser.add_argument("--label_all_dataset", type=str, default="data/Annotations.txt")

    parser.add_argument("--pseudo_labels_saved_dir", type=str, default='data/CLIP/segmcent_pseudo_labels')
    parser.add_argument("--visual_feats_saved_dir", type=str, default='data/CLIP/features')

    parser.add_argument('--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument('--print_progress', action='store_true')

    args = parser.parse_args()


    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    print('===> Prepare model ...')
    processor = CLIPProcessor.from_pretrained("clip/")
    model = CLIPModel.from_pretrained("clip/").to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print('===> Prepare dataloader ...')
    whole_dataset = LLP_dataset(label_txt=args.label_all_dataset, video_frame_dir=args.video_frame_dir, clip_processor=processor)
    whole_loader  = DataLoader(whole_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    event_captions = ['A photo of a Church bell.', 'A photo of a Male speech or man speaking.', 'A photo of Bark.', 'A photo of a Fixed-wing aircraft or airplane.', 'A photo of a Race car or auto racing.', 'A photo of a Female speech, woman speaking.',
              'A photo of a Helicopter.', 'A photo of a Violin or fiddle.', 'A photo of a Flute.', 'A photo of a Ukulele.', 'A photo of  Frying (food).',
              'A photo of a Truck.', 'A photo of a Shofar.', 'A photo of a Motorcycle.', 'A photo of a Acoustic guitar.',
              'A photo of a Train horn.', 'A photo of a Clock.', 'A photo of a Banjo.', 'A photo of a Goat.', 'A photo of a Baby cry or infant cry.',
              'A photo of a Bus.', 'A photo of a Chainsaw.', 'A photo of a Cat.', 'A photo of a Horse.',
              'A photo of Toilet flush.', 'A photo of Rodents or rats or mice.', 'A photo of a Accordion.', 'A photo of a Mandolin.']

    thresholds = np.array([20, 15, 18, 14, 15, 18, 18, 15, 15, 15,
                            15, 18, 15, 15, 15, 15, 15, 15, 15, 16,
                            14, 15, 15, 15, 18, 15, 15, 15 ])
    thresholds = torch.from_numpy(thresholds).to(device).unsqueeze(0).expand(10, -1)
    # .unsqueeze(0): 在张量的第0维度上添加一个维度，将张量形状从[25]变为[1, 25]。
    # .expand(10, -1): 将张量在第0维度上扩展10倍，使其形状变为[10, 25]，并且维持第二维度不变。
    # 该操作将维度为[25]的thresholds张量变成了维度为[10, 25]张量.这样是为了和logits_image_text的维度对齐, 因为logits_image_text的维度通常是[10, 25]

    print('===> Generate visual pseudo labels ...')
    print('(labels will be saved at {})'.format(args.pseudo_labels_saved_dir))
    visual_label_elaboration(model, processor, whole_loader, event_captions, thresholds,
                            args.pseudo_labels_saved_dir, device, True)
    print()


    '''
    Extract visual embedding from each segment
    '''
    print('===> Generate visual segment features ...')
    print('(features will be saved at {})'.format(args.visual_feats_saved_dir))
    get_image_embedding(model, whole_loader, args.visual_feats_saved_dir,
                        device, True, processor)
