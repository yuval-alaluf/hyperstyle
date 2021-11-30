import copy
import clip
import torch

from editing.styleclip.stylespace_utils import features_channels_to_s


class StyleCLIPGlobalDirection:

    def __init__(self, delta_i_c, s_std, text_prompts_templates):
        super(StyleCLIPGlobalDirection, self).__init__()
        self.delta_i_c = delta_i_c
        self.s_std = s_std
        self.text_prompts_templates = text_prompts_templates
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")

    def get_delta_s(self, neutral_text, target_text, beta):
        delta_i = self.get_delta_i([target_text, neutral_text]).float()
        r_c = torch.matmul(self.delta_i_c, delta_i)
        delta_s = copy.copy(r_c)
        channels_to_zero = torch.abs(r_c) < beta
        delta_s[channels_to_zero] = 0
        max_channel_value = torch.abs(delta_s).max()
        if max_channel_value > 0:
            delta_s /= max_channel_value
        direction = features_channels_to_s(delta_s, self.s_std)
        return direction

    def get_delta_i(self, text_prompts):
        text_features = self._get_averaged_text_features(text_prompts)
        delta_t = text_features[0] - text_features[1]
        delta_i = delta_t / torch.norm(delta_t)
        return delta_i

    def _get_averaged_text_features(self, text_prompts):
        with torch.no_grad():
            text_features_list = []
            for text_prompt in text_prompts:
                formatted_text_prompts = [template.format(text_prompt) for template in self.text_prompts_templates]  # format with class
                formatted_text_prompts = clip.tokenize(formatted_text_prompts).cuda()  # tokenize
                text_embeddings = self.clip_model.encode_text(formatted_text_prompts)  # embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                text_features_list.append(text_embedding)
            text_features = torch.stack(text_features_list, dim=1).cuda()
        return text_features.t()
