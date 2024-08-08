import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from easydict import EasyDict as edict
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
from tqdm import tqdm

class UNet_SD(nn.Module):
    def __init__(self, in_channels=4,
                 base_channels=320,
                 time_emb_dim=1280,
                 context_dim=768,
                 multipliers=(1, 2, 4, 4),
                 attn_levels=(0, 1, 2),
                 nResAttn_block=2,
                 cat_unet=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.out_channels = in_channels
        base_channels = base_channels
        time_emb_dim = time_emb_dim
        context_dim = context_dim
        multipliers = multipliers
        nlevel = len(multipliers)
        self.base_channels = base_channels
        level_channels = [base_channels * mult for mult in multipliers]
        self.time_embedding = nn.Sequential(OrderedDict({
            "linear_1": nn.Linear(base_channels, time_emb_dim, bias=True),
            "act": nn.SiLU(),
            "linear_2": nn.Linear(time_emb_dim, time_emb_dim, bias=True),
        })
        )  
        self.conv_in = nn.Conv2d(self.in_channels, base_channels, 3, stride=1, padding=1)

        nResAttn_block = nResAttn_block
        self.down_blocks = TimeModulatedSequential() 
        self.down_blocks_channels = [base_channels]
        cur_chan = base_channels
        for i in range(nlevel):
            for j in range(nResAttn_block):
                res_attn_sandwich = TimeModulatedSequential()
                res_attn_sandwich.append(ResBlock(in_channel=cur_chan, time_emb_dim=time_emb_dim, out_channel=level_channels[i]))
                if i in attn_levels:
                    res_attn_sandwich.append(SpatialTransformer(level_channels[i], context_dim=context_dim))
                cur_chan = level_channels[i]
                self.down_blocks.append(res_attn_sandwich)
                self.down_blocks_channels.append(cur_chan)
            if not i == nlevel - 1:
                self.down_blocks.append(TimeModulatedSequential(DownSample(level_channels[i])))
                self.down_blocks_channels.append(cur_chan)

        self.mid_block = TimeModulatedSequential(
            ResBlock(cur_chan, time_emb_dim),
            SpatialTransformer(cur_chan, context_dim=context_dim),
            ResBlock(cur_chan, time_emb_dim),
        )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(nlevel)):
            for j in range(nResAttn_block + 1):
                res_attn_sandwich = TimeModulatedSequential()
                res_attn_sandwich.append(ResBlock(in_channel=cur_chan + self.down_blocks_channels.pop(),
                                                  time_emb_dim=time_emb_dim, out_channel=level_channels[i]))
                if i in attn_levels:
                    res_attn_sandwich.append(SpatialTransformer(level_channels[i], context_dim=context_dim))
                cur_chan = level_channels[i]
                if j == nResAttn_block and i != 0:
                    res_attn_sandwich.append(UpSample(level_channels[i]))
                self.up_blocks.append(res_attn_sandwich)
        self.output = nn.Sequential(
            nn.GroupNorm(32, base_channels, ),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.out_channels, 3, padding=1),
        )
        self.to(self.device)
    def time_proj(self, time_steps, max_period: int = 10000):
        if time_steps.ndim == 0:
            time_steps = time_steps.unsqueeze(0)
        half = self.base_channels // 2
        frequencies = torch.exp(- math.log(max_period)
                                * torch.arange(start=0, end=half, dtype=torch.float32) / half
                                ).to(device=time_steps.device)
        angles = time_steps[:, None].float() * frequencies[None, :]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, x, time_steps, cond=None, encoder_hidden_states=None, output_dict=True):
        if cond is None and encoder_hidden_states is not None:
            cond = encoder_hidden_states
        t_emb = self.time_proj(time_steps)
        t_emb = self.time_embedding(t_emb)
        x = self.conv_in(x)
        down_x_cache = [x]
        for module in self.down_blocks:
            x = module(x, t_emb, cond)
            down_x_cache.append(x)
        x = self.mid_block(x, t_emb, cond)
        for module in self.up_blocks:
            x = module(torch.cat((x, down_x_cache.pop()), dim=1), t_emb, cond)
        x = self.output(x)
        if output_dict:
            return edict(sample=x)
        else:
            return x

class TimeModulatedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for module in self:
            if isinstance(module, TimeModulatedSequential):
                x = module(x, t_emb, cond)
            elif isinstance(module, ResBlock):
                x = module(x, t_emb)
            elif isinstance(module, SpatialTransformer):
                x = module(x, cond=cond)
            else:
                x = module(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel, time_emb_dim, out_channel=None, ):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        self.norm1 = nn.GroupNorm(32, in_channel, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(in_features=time_emb_dim, out_features=out_channel, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channel, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channel == out_channel:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x, t_emb, cond=None):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        if t_emb is not None:
            t_hidden = self.time_emb_proj(self.nonlinearity(t_emb))
            h = h + t_hidden[:, :, None, None]
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.conv_shortcut(x)


class UpSample(nn.Module):
    def __init__(self, channel, scale_factor=2, mode='nearest'):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channel, ):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, )

    def forward(self, x):
        return self.conv(x)  

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=8, ):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            self.to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.self_attn = True
        else:
            self.to_k = nn.Linear(context_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(context_dim, embed_dim, bias=False)
            self.self_attn = False
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True)
        )  

    def forward(self, tokens, context=None):
        Q = self.to_q(tokens)
        K = self.to_k(tokens) if self.self_attn else self.to_k(context)
        V = self.to_v(tokens) if self.self_attn else self.to_v(context)
        Q = rearrange(Q, 'B T (H D) -> (B H) T D', H=self.num_heads, D=self.head_dim)
        K = rearrange(K, 'B T (H D) -> (B H) T D', H=self.num_heads, D=self.head_dim)
        V = rearrange(V, 'B T (H D) -> (B H) T D', H=self.num_heads, D=self.head_dim)
        scoremats = torch.einsum("BTD,BSD->BTS", Q, K)
        attnmats = F.softmax(scoremats / math.sqrt(self.head_dim), dim=-1)
        ctx_vecs = torch.einsum("BTS,BSD->BTD", attnmats, V)
        ctx_vecs = rearrange(ctx_vecs, '(B H) T D -> B T (H D)', H=self.num_heads, D=self.head_dim)
        return self.to_out(ctx_vecs)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.attn1 = CrossAttention(hidden_dim, hidden_dim, num_heads=num_heads)  
        self.attn2 = CrossAttention(hidden_dim, hidden_dim, context_dim, num_heads=num_heads)  

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward_GEGLU(hidden_dim, )

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class GEGLU_proj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GEGLU_proj, self).__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward_GEGLU(nn.Module):
    def __init__(self, hidden_dim, mult=4):
        super(FeedForward_GEGLU, self).__init__()
        self.net = nn.Sequential(
            GEGLU_proj(hidden_dim, mult * hidden_dim),
            nn.Dropout(0.0),
            nn.Linear(mult * hidden_dim, hidden_dim)
        )

    def forward(self, x, ):
        return self.net(x)


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super(SpatialTransformer, self).__init__()
        self.norm = nn.GroupNorm(32, hidden_dim, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(hidden_dim, context_dim, num_heads=8)
        )  
        self.proj_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, "b c h w->b (h w) c")
        x = self.transformer_blocks[0](x, cond)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x) + x_in
    

class LinearLoRA(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.
    ):
        super(LinearLoRA, self).__init__()
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora_alpha = lora_alpha
        self.rank = rank
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.bias = linear_layer.bias.data if linear_layer.bias is not None else None
        self.scaling = self.lora_alpha / self.rank
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        output = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        if self.bias is not None:
            return output + self.bias.to(torch.float32)
        return output


class Conv2DLoRA(nn.Module):
    def __init__(
        self,
        conv_layer: nn.Conv2d,
        rank: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.
    ):
        super(Conv2DLoRA, self).__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size[0]
        self.bias = conv_layer.bias
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_channels * self.kernel_size * self.kernel_size)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_channels, rank)))
        self.scaling = self.lora_alpha / self.rank
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_weight = (self.lora_A.T @ self.lora_B.T) * self.scaling
        lora_weight = lora_weight.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(
            self.lora_dropout(x),
            lora_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

def replace_layers(model, rank=4):
    layers = []
    for name, module in model.named_modules():
        layers.append((name, module))
    for name, module in layers:
        if isinstance(module, nn.Linear):
            lora_layer = LinearLoRA(module, rank)
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, layer_name, lora_layer)
        elif isinstance(module, nn.Conv2d):
            if name in ["conv_in", "conv_out"]:
                continue
            lora_layer = Conv2DLoRA(module, rank)
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, layer_name, lora_layer)

    return model

BATCH_SIZE = 16
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_lora_diffusion(dataset):
    model_id = "runwayml/stable-diffusion-v1-5"
    
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    unet = UNet_SD().to(DEVICE)  
    unet = replace_layers(unet, rank=16) 
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=NUM_EPOCHS * len(dataset) // BATCH_SIZE
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        unet.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, captions = batch
            images = images.to(DEVICE)
            
            text_input = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]
            
            latents = vae.encode(images).latent_dist.sample().detach()
            latents = latents * 0.18215
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = unet(noisy_latents, timesteps, text_embeddings)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_loss,
        }, f"lora_diffusion_checkpoint_epoch_{epoch+1}.pth")
    
    print("Training completed!")
