import os
import torch
import torch.utils.data as utils

from utility_functions import audio_image_csv_to_dict, load_image

class CustomAudioVisualDataset(utils.Dataset):
    def __init__(self, audio_predictors, audio_target, image_path=None, image_audio_csv_path=None, transform_image=None):
        self.audio_predictors = audio_predictors[0]
        self.audio_target = audio_target
        self.audio_predictors_path = audio_predictors[1]
        self.image_path = image_path
        if image_path:
            print("AUDIOVISUAL ON")
            self.image_audio_dict = audio_image_csv_to_dict(image_audio_csv_path)
            self.transform = transform_image
        else:
            print("AUDIOVISUAL OFF")
    
    def __len__(self):
        return len(self.audio_predictors)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_pred = self.audio_predictors[idx]
        audio_trg = self.audio_target[idx]
        audio_pred_path = self.audio_predictors_path[idx]
        
        if self.image_path:
            image_name = self.image_audio_dict[audio_pred_path]
            img = load_image(os.path.join(self.image_path, image_name))
            
            if self.transform:
                img = self.transform(img)

            return (audio_pred, img), audio_trg
        
        return audio_pred, audio_trg

    
# class CustomBatch:
#     def __init__(self, data):
#         transposed_data = list(zip(*data))
#         transposed_data_0 = list(zip(*transposed_data[0]))
#         self.audio_pred = torch.stack(transposed_data_0[0], 0)
#         self.inp = list(zip(self.audio_pred, transposed_data_0[1]))
#         self.tgt = torch.stack(transposed_data[1], 0)

#     # custom memory pinning method on custom type
#     def pin_memory(self):
#         self.audio_pred = self.audio_pred.pin_memory()
#         self.tgt = self.tgt.pin_memory()
#         return self

# def collate_wrapper(batch):
#     return CustomBatch(batch)