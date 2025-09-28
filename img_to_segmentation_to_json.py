
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import json

import sys,shutil
sys.path.append('/content/floodnet-segmentation')
from core.model import set_model
from core.opt import Opt

class floodnet_colors:
  def __init__(self ):
   self.floodnet_colors = {
      0: [0, 0, 0], 1: [255, 0, 0], 2: [255, 191, 0], 3: [0, 77, 179],
      4: [153, 153, 153], 5: [0, 128, 255], 6: [0, 217, 0], 7: [255, 0, 255],
      8: [179, 179, 230], 9: [26, 140, 77]
  }

   self.class_names = {
      0: "background", 1: "building_flooded", 2: "building_non_flooded",
      3: "road_flooded", 4: "road_non_flooded", 5: "water", 6: "tree",
      7: "vehicle", 8: "pool", 9: "grass"
  }
   self.model_path="/content/drive/MyDrive/pspnet_best (4).pth"

  def class_mask_to_rgb(self,class_mask):
      h, w = class_mask.shape
      rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
      for class_id, color in self.floodnet_colors.items():
          mask = (class_mask == class_id)
          rgb_mask[mask] = color
      return rgb_mask

  def load_model(self):
      opt_instance = Opt()
      opt_instance.name_net = 'pspnet'
      model, _, _ = set_model(opt_instance)
      checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=False)
      model.load_state_dict(checkpoint['model'])
      model.eval()
      if torch.cuda.is_available():
          model = model.cuda()
      return model

  def count_objects_from_mask(self,pred_mask):
      object_counts = {}
      for class_id, color in self.floodnet_colors.items():
          if class_id == 0:
              continue
          class_name = self.class_names[class_id]
          class_mask = (pred_mask == class_id).astype(np.uint8)
          if class_mask.sum() == 0:
              continue
          kernel = np.ones((3,3), np.uint8)
          class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
          class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
          num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
          min_area = 50
          valid_components = 0
          for i in range(1, num_labels):
              area = stats[i, cv2.CC_STAT_AREA]
              if area >= min_area:
                  valid_components += 1
          if valid_components > 0:
              object_counts[class_name] = valid_components
      return object_counts

  def process_single_image(self,model, image_path, save_path=None):
      opt_instance = Opt()
      img_transform = T.Compose([
          T.Resize((opt_instance.resize_height, opt_instance.resize_width)),
          T.ToTensor(),
          T.Normalize(mean=opt_instance.mean, std=opt_instance.std)
      ])

      img = Image.open(image_path).convert("RGB")
      original_size = img.size
      img_tensor = img_transform(img).unsqueeze(0)

      if torch.cuda.is_available():
          img_tensor = img_tensor.cuda()

      with torch.no_grad():
          pred = model(img_tensor)
          pred_mask = pred.argmax(1).squeeze().cpu().numpy()

      pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
      object_counts = self.count_objects_from_mask(pred_mask_resized)

      result = {
          "image_path": image_path,
          "objects": object_counts,
          "total_objects": sum(object_counts.values())
      }

      if save_path:
          filename = os.path.splitext(os.path.basename(image_path))[0]
          rgb_mask = self.class_mask_to_rgb(pred_mask_resized)
          shutil.copy2(image_path, f"{save_path}/{filename}_image.jpg")
          cv2.imwrite(f"{save_path}/{filename}_mask.png", cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
          with open(f"{save_path}/{filename}_result.json", 'w') as f:
              json.dump(result, f, indent=2)

      return result

